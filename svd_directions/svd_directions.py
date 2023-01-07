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
from transformers import AutoModelForCausalLM, AutoTokenizer
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

def get_model_tokenizer_embedding(model_name="gpt2-medium"):
    """
    
    return: model, tokenizer, embedding (note: transposed), device
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        print("WARNING: you should probably restart on a GPU runtime")

    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    emb = model.get_output_embeddings().weight.data.T.detach()
    return model, tokenizer, emb, device

#model, tokenizer, emb, device = get_model_tokenizer_embedding()
#my_tokenizer = tokenizer

class SVDTransformer:
    """Class that makes it easy to apply SVD to a transformer model"""
    def __init__(self, model_name="gpt2-medium"):
        self.model_name = model_name
        self.model, self.tokenizer, self.emb, self.device = get_model_tokenizer_embedding(model_name)
        self.num_layers, self.num_heads, self.hidden_dim, self.head_size = self.get_model_info()
        self.K_heads, self.V_heads = self.get_mlp_weights()
        self.W_Q_heads, self.W_K_heads, self.W_V_heads, self.W_O_heads = self.get_attention_heads()
        self.all_tokens = [self.tokenizer.decode([i]) for i in range(self.tokenizer.vocab_size)]


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
            K = self.model.get_parameter(f"transformer.h.{j}.mlp.c_fc.weight").T.detach()
            # fuse the layernorm
            ln_2_weight = self.model.get_parameter(f"transformer.h.{j}.ln_2.weight").detach()
            K = torch.einsum("oi,i -> oi", K, ln_2_weight)
            
            V = self.model.get_parameter(f"transformer.h.{j}.mlp.c_proj.weight")
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
            qkv = self.model.get_parameter(f"transformer.h.{j}.attn.c_attn.weight").detach().T
            ln_weight_1 = self.model.get_parameter(f"transformer.h.{j}.ln_1.weight").detach()
            
            qkv = qkv - torch.mean(qkv, dim=0) 
            qkv = torch.einsum("oi,i -> oi", qkv, ln_weight_1)
            qkvs.append(qkv.T)

        W_Q, W_K, W_V = torch.cat(qkvs).chunk(3, dim=-1)
        W_O = torch.cat([self.model.get_parameter(f"transformer.h.{j}.attn.c_proj.weight") for j in range(self.num_layers)]).detach()

        # The following 4 reshapes create two separate dimenions for the head and head size, by default they are combined
        W_V_heads = W_V.reshape(self.num_layers, self.hidden_dim, self.num_heads, self.head_size).permute(0, 2, 1, 3)
        W_O_heads = W_O.reshape(self.num_layers, self.num_heads, self.head_size, self.hidden_dim)
        W_Q_heads = W_Q.reshape(self.num_layers, self.hidden_dim, self.num_heads, self.head_size).permute(0, 2, 1, 3)
        W_K_heads = W_K.reshape(self.num_layers, self.hidden_dim, self.num_heads, self.head_size).permute(0, 2, 1, 3)
        return W_Q_heads, W_K_heads, W_V_heads, W_O_heads

    def top_singular_vectors(self, mat, k = 20, N_singular_vectors = 10, with_negative = False,use_visualization=True, filter="topk"):
        U,S,V = torch.linalg.svd(mat)
        Vs = []
        for i in range(N_singular_vectors):
            acts = V[i,:].float() @ self.emb
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
                acts = -V[i,:].float() @ self.emb
                Vs.append(acts)
            if use_visualization:
                Vs = torch.stack(Vs, dim=1).unsqueeze(1) # n_tokens, n_layers (1), n_directions
                pysvelte.TopKTable(tokens=self.all_tokens, activations=Vs, obj_type="SVD direction", k=k, filter=filter).show()
            else:
                Vs = [utils.top_tokens(Vs[i].float().cpu(), k = k, pad_to_maxlen=True, tokenizer=self.tokenizer) for i in range(len(Vs))]
                print(tabulate([*zip(*Vs)]))

    def plot_MLP_singular_vectors(self, K,layer_idx, max_rank=None):
        W_matrix = K[layer_idx, :,:]
        U,S,V = torch.linalg.svd(W_matrix,full_matrices=False)
        if not max_rank:
            max_rank = len(S)
        if max_rank > len(S):
            max_rank = len(S) -1
        plt.plot(S[0:max_rank].detach().cpu().numpy())
        plt.yscale('log')
        plt.ylabel("Singular value")
        plt.xlabel("Rank")
        plt.title("Distribution of the singular vectors")
        plt.show()

    def OV_top_singular_vectors(self, layer_idx, head_idx, k=20, N_singular_vectors=10, use_visualization=True, with_negative=False, filter="topk", return_OV=False):
        W_V_tmp, W_O_tmp = self.W_V_heads[layer_idx, head_idx, :], self.W_O_heads[layer_idx, head_idx]
        OV = W_V_tmp @ W_O_tmp
        U,S,V = torch.linalg.svd(OV)
        Vs = []
        for i in range(N_singular_vectors):
            acts = V[i,:].float() @ self.emb
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
                acts = -V[i,:].float() @ self.emb
                Vs.append(acts)
            if use_visualization:
                Vs = torch.stack(Vs, dim=1).unsqueeze(1) # n_tokens, n_layers (1), n_directions
                pysvelte.TopKTable(tokens=self.all_tokens, activations=Vs, obj_type="SVD direction", k=k, filter=filter).show()
            else:
                Vs = [utils.top_tokens(Vs[i].float().cpu(), k = k, pad_to_maxlen=True, tokenizer=self.tokenizer) for i in range(len(Vs))]
                print(tabulate([*zip(*Vs)]))
        if return_OV:
            return OV

    def random_top_singular_vectors(self,k=20, N=10, use_visualization = True):
        A = torch.randn(size=(1024,1024)).to(self.device)
        U,S,V = torch.linalg.svd(A)
        Vs = []
        for i in range(N):
            acts = V[i,:].float() @ self.emb
            Vs.append(acts)
        if use_visualization:
            Vs = torch.stack(Vs, dim=1).unsqueeze(1) # n_tokens, n_layers (1), n_directions
            pysvelte.TopKTable(tokens=self.all_tokens, activations=Vs, obj_type="SVD direction", k=k, filter="topk").show()
        else:
            Vs = [utils.top_tokens(Vs[i].float().cpu(), k = k, pad_to_maxlen=True) for i in range(len(Vs))]
            print(tabulate([*zip(*Vs)]))

    def plot_singular_value_distribution(self, layer_idx, head_idx, max_rank= 100):
        W_V_tmp, W_O_tmp = self.W_V_heads[layer_idx, head_idx, :], self.W_O_heads[layer_idx, head_idx]
        OV = W_V_tmp @ W_O_tmp
        U,S,V = torch.linalg.svd(OV)
        if max_rank > len(S):
            max_rank = len(S) -1
        plt.plot(S[0:max_rank].detach().cpu().numpy())
        plt.yscale('log')
        plt.ylabel("Singular value")
        plt.xlabel("Rank")
        plt.title("Distribution of the singular vectors")
        plt.show()

    def MLP_K_top_singular_vectors(self, K, layer_idx, k=20, N_singular_vectors=10, with_negative = False, use_visualization = True):
        W_matrix = K[layer_idx, :,:]
        U,S,V = torch.linalg.svd(W_matrix,full_matrices=False)
        Vs = []
        for i in range(N_singular_vectors):
            acts = V[i,:].float() @ self.emb
            Vs.append(acts)
        if use_visualization:
            Vs = torch.stack(Vs, dim=1).unsqueeze(1) # n_tokens, n_layers (1), n_directions
            pysvelte.TopKTable(tokens=self.all_tokens, activations=Vs, obj_type="SVD direction", k=k, filter="topk").show()
        else:
            Vs = [utils.top_tokens(Vs[i].float().cpu(), k = k, pad_to_maxlen=True) for i in range(len(Vs))]
            print(tabulate([*zip(*Vs)]))
        if with_negative:
            Vs = []
            for i in range(N_singular_vectors):
                acts = -V[i,:].float() @ self.emb
                Vs.append(acts)
            if use_visualization:
                Vs = torch.stack(Vs, dim=1).unsqueeze(1) # n_tokens, n_layers (1), n_directions
                pysvelte.TopKTable(tokens=self.all_tokens, activations=Vs, obj_type="SVD direction", k=k, filter="topk").show()
            else:
                Vs = [utils.top_tokens(Vs[i].float().cpu(), k = k, pad_to_maxlen=True) for i in range(len(Vs))]
                print(tabulate([*zip(*Vs)]))

    def plot_all_MLP_singular_vectors(self,K,max_rank=None):
        fig = plt.figure()
        ax = plt.subplot(111)
        for i in range(self.num_layers):
            W_matrix = K[i, :,:]
            U,S,V = torch.linalg.svd(W_matrix,full_matrices=False)
            if not max_rank:
                max_rank = len(S)
            if max_rank > len(S):
                max_rank = len(S) -1
            plt.plot(S[0:max_rank].detach().cpu().numpy(), label="Block " + str(i))
        plt.yscale('log')
        plt.ylabel("Singular value")
        plt.xlabel("Rank")
        plt.title("Distribution of the singular vectors")
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5),fontsize=10)
        plt.show()

    def MLP_V_top_singular_vectors(self, layer_idx, k=20, N_singular_vectors=10, with_negative = False, use_visualization = True):
        with torch.no_grad():
            W_matrix = self.model.get_parameter(f"transformer.h.{layer_idx}.mlp.c_proj.weight").detach()
            U,S,Vval = torch.svd(W_matrix)
            Vs = []
            for i in range(N_singular_vectors):
                acts = Vval.T[i,:].float() @ self.emb
                Vs.append(acts)
        if use_visualization:
            Vs = torch.stack(Vs, dim=1).unsqueeze(1) # n_tokens, n_layers (1), n_directions
            pysvelte.TopKTable(tokens=self.all_tokens, activations=Vs, obj_type="SVD direction", k=k, filter="topk").show()
        else:
            Vs = [utils.top_tokens(Vs[i].float().cpu(), k = k, pad_to_maxlen=True) for i in range(len(Vs))]
            print(tabulate([*zip(*Vs)]))
        if with_negative:
            Vs = []
            for i in range(N_singular_vectors):
                acts = -Vval.T[i,:].float() @ self.emb
                Vs.append(acts)
            if use_visualization:
                Vs = torch.stack(Vs, dim=1).unsqueeze(1) # n_tokens, n_layers (1), n_directions
                pysvelte.TopKTable(tokens=self.all_tokens, activations=Vs, obj_type="SVD direction", k=k, filter="topk").show()
            else:
                Vs = [utils.top_tokens(Vs[i].float().cpu(), k = k, pad_to_maxlen=True) for i in range(len(Vs))]
                print(tabulate([*zip(*Vs)]))


