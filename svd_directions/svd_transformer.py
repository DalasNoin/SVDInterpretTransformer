import torch
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gc
from copy import deepcopy
# from tqdm.auto import tqdm, trange
import re
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer, DecisionTransformerModel, GPTJForCausalLM
# utils
import json
from torch import nn
import torch.nn.functional as F
from datasets import load_dataset
from torch.nn import functional as F
from tabulate import tabulate
from tqdm import tqdm, trange
import functools
import math
import utils
from decision_transformers import decision_tokenizer

# this resets up the site so you don't have to restart the runtime to use pysvelte
import site
site.main()
import pysvelte




sns.set_palette('colorblind')
cmap = sns.color_palette('colorblind')


def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))

def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition('.')
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)

def get_model_tokenizer_embedding(model_name="gpt2-medium", embedding_name=None):
    """
    
    return: model, tokenizer, embedding (note: transposed), device
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    emb_bias = None
    # if device == 'cpu':
    #     print("WARNING: you should probably restart on a GPU runtime")
    if "edbeeching/decision-transformer" in model_name:
        # todo: check if layernorm is at the same positions
        model = DecisionTransformerModel.from_pretrained(model_name).to(device)
        assert embedding_name in [None, "state", "return", "action", "timestep"]
        if embedding_name is None or embedding_name == "state":
            emb = model.embed_state.weight.data.detach()
            # emb = model.predict_state.weight.data.T.detach()
        elif embedding_name == "return":
            emb = model.embed_return.weight.data.detach()
        elif embedding_name == "action":
            emb = model.embed_action.weight.data.detach()
            emb = model.predict_action[0].weight.data.T.detach()
        elif embedding_name == "timestep":
            emb = model.embed_timestep.weight.data.T.detach()
        # emb = model.predict_state.weight.data.T.detach()
        # model.predict_state[0].weight.data.T.detach()
        # emb = model.embed_state.weight.data.detach()
        # emb = model.embed_return.weight.data.detach()
        # normalize embedding emb
        emb = emb / torch.norm(emb, dim=0) * 3 # 3 for the color map
        transformer_module_name = "encoder"

        tokenizer = decision_tokenizer.DecisionTokenizer(int(emb.shape[1]), name=embedding_name[:3])
        # alternatively look at model._modules.keys()
        model.config.n_embd = model.config.hidden_size # I think there is an error in the config
        # https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/configuration_gpt2.py
        # n_embd is the embedding dimension, but the model has a hidden_size parameter they should match, 
        # but n_embd is set to the default 768
    elif model_name == "EleutherAI/gpt-j-6B":
        # mlp.c_fc is called mlp.fc_in and fc_out
        # there is no layer norm 2
        # c_proj is called out_proj
        # there is a q_proj, k_proj, v_proj instead of a single c_attn
        # position of layernorm might also be different
        model = GPTJForCausalLM.from_pretrained(
            pretrained_model_name_or_path=model_name,
            revision="float16",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
            ).to(device)
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
        transformer_module_name = "transformer"
        emb = model.get_output_embeddings().weight.data.T.detach()
        emb_bias = model.get_output_embeddings().bias.data.detach()
        transformer_module_name = "transformer"
    elif model_name == "pvduy/openai_summarize_sft_gptj":
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
        model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True).to(device)
        model.config.pad_token_id = tokenizer.bos_token_id
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.bos_token_id
        tokenizer.padding_side = "left"
        tokenizer.truncation_side = "left"
        emb = model.get_output_embeddings().weight.data.T.detach()
        transformer_module_name = "transformer"
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        emb = model.get_output_embeddings().weight.data.T.detach()
        transformer_module_name = "transformer"
    
    model.eval()
    return model, tokenizer, emb, emb_bias, device, transformer_module_name


class SVDTransformer:
    """Class that makes it easy to apply SVD to a transformer model"""
    def __init__(self, model_name="gpt2-medium", embedding_name=None):
        self.model_name = model_name
        self.model, self.tokenizer, self.emb, self.emb_bias, self.device, self.transformer_module_name = get_model_tokenizer_embedding(model_name, embedding_name)

        self.num_layers, self.num_heads, self.hidden_dim, self.head_size = self.get_model_info()
        self.K_heads, self.V_heads = self.get_mlp_weights()
        self.W_Q_heads, self.W_K_heads, self.W_V_heads, self.W_O_heads = self.get_attention_heads()
        self.all_tokens = [self.tokenizer.decode([i]) for i in range(len(self.tokenizer.vocab))]
        self.vocab_size = len(self.tokenizer.vocab)
        
        if self.emb_bias is None:
            self.emb_bias = torch.zeros(self.vocab_size)

    ## These functions prepare the data for SVD

    def get_model_info(self):
        num_layers = self.model.config.n_layer
        num_heads = self.model.config.n_head
        hidden_dim = self.model.config.n_embd
        head_size = hidden_dim // num_heads
        return num_layers, num_heads, hidden_dim, head_size

    def get_mlp_weights(self):
        Ks = []
        Vs = []
        for j in range(self.num_layers):
            K = self.model.get_parameter(f"{self.transformer_module_name}.h.{j}.mlp.c_fc.weight").T.detach()
            # fuse the layernorm
            ln_2_weight = self.model.get_parameter(f"{self.transformer_module_name}.h.{j}.ln_2.weight").detach()
            K = torch.einsum("oi,i -> oi", K, ln_2_weight)
            
            V = self.model.get_parameter(f"{self.transformer_module_name}.h.{j}.mlp.c_proj.weight").detach()
            Ks.append(K)
            Vs.append(V)
        
        Ks =  torch.cat(Ks)
        Vs = torch.cat(Vs)
        K_heads = Ks.reshape(self.num_layers, -1, self.hidden_dim)
        V_heads = Vs.reshape(self.num_layers, -1, self.hidden_dim)
        return K_heads, V_heads

    def get_attention_heads(self):
        qkvs = []
        for j in range(self.num_layers):
            qkv = self.model.get_parameter(f"{self.transformer_module_name}.h.{j}.attn.c_attn.weight").detach().T
            ln_weight_1 = self.model.get_parameter(f"{self.transformer_module_name}.h.{j}.ln_1.weight").detach()
            
            qkv = qkv - torch.mean(qkv, dim=0) 
            qkv = torch.einsum("oi,i -> oi", qkv, ln_weight_1)
            qkvs.append(qkv.T)

        W_Q, W_K, W_V = torch.cat(qkvs).chunk(3, dim=-1)
        W_O = torch.cat([self.model.get_parameter(f"{self.transformer_module_name}.h.{j}.attn.c_proj.weight") for j in range(self.num_layers)]).detach()

        # The following 4 reshapes create two separate dimenions for the head and head size, by default they are combined
        W_V_heads = W_V.reshape(self.num_layers, self.hidden_dim, self.num_heads, self.head_size).permute(0, 2, 1, 3)
        W_O_heads = W_O.reshape(self.num_layers, self.num_heads, self.head_size, self.hidden_dim)
        W_Q_heads = W_Q.reshape(self.num_layers, self.hidden_dim, self.num_heads, self.head_size).permute(0, 2, 1, 3)
        W_K_heads = W_K.reshape(self.num_layers, self.hidden_dim, self.num_heads, self.head_size).permute(0, 2, 1, 3)
        return W_Q_heads, W_K_heads, W_V_heads, W_O_heads

    ## These functions apply SVD to the data

    def top_singular_vectors(self, mat, k = 20, N_singular_vectors = 10, with_negative = False,use_visualization=True, filter="topk"):
        U,S,V = torch.linalg.svd(mat)
        Vs = []
        for i in range(N_singular_vectors):
            acts = V[i,:].float() @ self.emb.float()
            Vs.append(acts)
        if use_visualization:
            Vs = torch.stack(Vs, dim=1).unsqueeze(1) # n_tokens, n_layers (1), n_directions
            pysvelte.TopKTable(tokens=self.all_tokens, activations=Vs, obj_type="SVD direction", k=k, filter=filter).show()
        else:
            Vs = [utils.top_tokens(Vs[i].float().cpu(), k = k, pad_to_maxlen=True, tokenizer=self.tokenizer) for i in range(len(Vs))]
            print(tabulate([*zip(*Vs)]))
        if with_negative:
            Vs = []
            for i in range(N_singular_vectors):
                acts = -V[i,:].float() @ self.emb.float()
                Vs.append(acts)
            if use_visualization:
                Vs = torch.stack(Vs, dim=1).unsqueeze(1) # n_tokens, n_layers (1), n_directions
                pysvelte.TopKTable(tokens=self.all_tokens, activations=Vs, obj_type="SVD direction", k=k, filter=filter).show()
            else:
                Vs = [utils.top_tokens(Vs[i].float().cpu(), k = k, pad_to_maxlen=True, tokenizer=self.tokenizer) for i in range(len(Vs))]
                print(tabulate([*zip(*Vs)]))

    def OV_top_singular_vectors(self, layer_idx, head_idx, k=20, N_singular_vectors=10, use_visualization=True, with_negative=False, filter="topk", return_OV=False):
        W_V_tmp, W_O_tmp = self.W_V_heads[layer_idx, head_idx, :], self.W_O_heads[layer_idx, head_idx]
        if k > self.vocab_size:
            k = self.vocab_size
        OV = W_V_tmp.float() @ W_O_tmp.float()
        U,S,V = torch.linalg.svd(OV)
        Vs = []
        for i in range(N_singular_vectors):
            acts = V[i,:].float() @ self.emb.float()
            Vs.append(acts)
        if use_visualization:
            Vs = torch.stack(Vs, dim=1).unsqueeze(1) # n_tokens, n_layers (1), n_directions
            pysvelte.TopKTable(tokens=self.all_tokens, activations=Vs, layer_labels=[layer_idx], obj_type="SVD direction", k=k, filter=filter).show()
        else:
            Vs = [utils.top_tokens(Vs[i].float().cpu(), k = k, pad_to_maxlen=True, tokenizer=self.tokenizer) for i in range(len(Vs))]
            print(tabulate([*zip(*Vs)]))
        if with_negative:
            Vs = []
            for i in range(N_singular_vectors):
                acts = -V[i,:].float() @ self.emb.float()
                Vs.append(acts)
            if use_visualization:
                Vs = torch.stack(Vs, dim=1).unsqueeze(1) # n_tokens, n_layers (1), n_directions
                pysvelte.TopKTable(tokens=self.all_tokens, activations=Vs, layer_labels=[layer_idx], obj_type="SVD direction", k=k, filter=filter).show()
            else:
                Vs = [utils.top_tokens(Vs[i].float().cpu(), k = k, pad_to_maxlen=True, tokenizer=self.tokenizer) for i in range(len(Vs))]
                print(tabulate([*zip(*Vs)]))
        if return_OV:
            return OV

    def random_top_singular_vectors(self,k=20, N=10, use_visualization = True):
        """
        This function computes the top singular vectors of a random matrix and plots the top k tokens for each singular vector.
        The purpose is for comparison with the singular values of the weights of the model.
        """
        if k > self.vocab_size:
            k = self.vocab_size
        A = torch.randn(size=(self.hidden_dim,self.hidden_dim)).to(self.device)
        U,S,V = torch.linalg.svd(A)
        Vs = []
        for i in range(N):
            acts = V[i,:].float() @ self.emb.float()
            Vs.append(acts)
        if use_visualization:
            Vs = torch.stack(Vs, dim=1).unsqueeze(1) # n_tokens, n_layers (1), n_directions
            pysvelte.TopKTable(tokens=self.all_tokens, activations=Vs, obj_type="SVD direction", k=k, filter="topk").show()
        else:
            Vs = [utils.top_tokens(Vs[i].float().cpu(), k = k, pad_to_maxlen=True) for i in range(len(Vs))]
            print(tabulate([*zip(*Vs)]))

    def plot_singular_value_distribution(self, layer_idx, head_idx, max_rank=None, use_log_scale=False):

        if max_rank is None:
            max_rank = self.head_size
        W_V_tmp, W_O_tmp = self.W_V_heads[layer_idx, head_idx, :], self.W_O_heads[layer_idx, head_idx]
        OV = W_V_tmp.float() @ W_O_tmp.float()
        U,S,V = torch.linalg.svd(OV)
        if max_rank > len(S):
            max_rank = len(S) -1
        plt.plot(S[0:max_rank].detach().cpu().numpy())
        if use_log_scale:
            plt.yscale('log')
        plt.ylabel("Singular value")
        plt.xlabel("Rank")
        plt.title("Distribution of the singular values")
        plt.show()


    def MLP_K_top_singular_vectors(self, layer_idx, k=20, N_singular_vectors=10, with_negative = False, use_visualization = True):
        if k > self.vocab_size:
            k = self.vocab_size
        W_matrix = self.K_heads[layer_idx, :,:]
        U,S,V = torch.linalg.svd(W_matrix.float(),full_matrices=False)
        Vs = []
        for i in range(N_singular_vectors):
            acts = V[i,:].float() @ self.emb.float() # + self.emb_bias
            Vs.append(acts)
        if use_visualization:
            Vs = torch.stack(Vs, dim=1).unsqueeze(1) # n_tokens, n_layers (1), n_directions
            pysvelte.TopKTable(tokens=self.all_tokens, activations=Vs, layer_labels=[layer_idx], obj_type="SVD direction", k=k, filter="topk").show()
        else:
            Vs = [utils.top_tokens(Vs[i].float().cpu(), k = k, pad_to_maxlen=True) for i in range(len(Vs))]
            print(tabulate([*zip(*Vs)]))
        if with_negative:
            Vs = []
            for i in range(N_singular_vectors):
                acts = -V[i,:].float() @ self.emb.float()
                Vs.append(acts)
            if use_visualization:
                Vs = torch.stack(Vs, dim=1).unsqueeze(1) # n_tokens, n_layers (1), n_directions
                pysvelte.TopKTable(tokens=self.all_tokens, activations=Vs, layer_labels=[layer_idx], obj_type="SVD direction", k=k, filter="topk").show()
            else:
                Vs = [utils.top_tokens(Vs[i].float().cpu(), k = k, pad_to_maxlen=True) for i in range(len(Vs))]
                print(tabulate([*zip(*Vs)]))

    # TODO: add this also for the V matrix
    def plot_MLP_singular_values(self,layer_idx, max_rank=None, use_log_scale=True):
        W_matrix = self.K_heads[layer_idx, :,:]
        U,S,V = torch.linalg.svd(W_matrix.float(),full_matrices=False)
        if not max_rank:
            max_rank = len(S)
        if max_rank > len(S):
            max_rank = len(S) -1
        plt.plot(S[0:max_rank].detach().cpu().numpy())
        if use_log_scale:
            plt.yscale('log')
        plt.ylabel("Singular value")
        plt.xlabel("Rank")
        plt.title("Distribution of the singular values")
        plt.show()

    def plot_all_MLP_singular_values(self, max_rank=None, use_log_scale=True):
        """
        plots the singular value distribution of the mlps.
        """
        fig = plt.figure()
        ax = plt.subplot(111)
        for i in range(self.num_layers):
            W_matrix = self.K_heads[i, :,:]
            U,S,V = torch.linalg.svd(W_matrix.float(),full_matrices=False)
            if not max_rank:
                max_rank = len(S)
            if max_rank > len(S):
                max_rank = len(S) -1
            plt.plot(S[0:max_rank].detach().cpu().numpy(), label="Block " + str(i))
        if use_log_scale:
            plt.yscale('log')
        plt.ylabel("Singular value")
        plt.xlabel("Rank")
        plt.title("Distribution of the singular values")
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5),fontsize=10)
        plt.show()

    def MLP_V_top_singular_vectors(self, layer_idx, k=20, N_singular_vectors=10, with_negative = False, use_visualization = True):
        if k > self.vocab_size:
            k = self.vocab_size
        with torch.no_grad():
            # todo: doesn't it already have this info in self.V_heads?
            W_matrix = self.V_heads[layer_idx, ...]
            U,S,Vval = torch.linalg.svd(W_matrix.float())
            Vs = []
            for i in range(N_singular_vectors):
                acts = Vval.T[i,:].float() @ self.emb.float() # + self.emb_bias
                Vs.append(acts)
        if use_visualization:
            Vs = torch.stack(Vs, dim=1).unsqueeze(1) # n_tokens, n_layers (1), n_directions
            pysvelte.TopKTable(tokens=self.all_tokens, activations=Vs, layer_labels=[layer_idx], obj_type="SVD direction", k=k, filter="topk").show()
        else:
            Vs = [utils.top_tokens(Vs[i].float().cpu(), k = k, pad_to_maxlen=True) for i in range(len(Vs))]
            print(tabulate([*zip(*Vs)]))
        if with_negative:
            Vs = []
            for i in range(N_singular_vectors):
                acts = -Vval.T[i,:].float() @ self.emb.float()
                Vs.append(acts)
            if use_visualization:
                Vs = torch.stack(Vs, dim=1).unsqueeze(1) # n_tokens, n_layers (1), n_directions
                pysvelte.TopKTable(tokens=self.all_tokens, activations=Vs, layer_labels=[layer_idx], obj_type="SVD direction", k=k, filter="topk").show()
            else:
                Vs = [utils.top_tokens(Vs[i].float().cpu(), k = k, pad_to_maxlen=True) for i in range(len(Vs))]
                print(tabulate([*zip(*Vs)]))


