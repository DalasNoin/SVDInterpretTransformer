import svd_transformer
import torch


class SVDGPTJ(svd_transformer.SVDTransformer):
    def get_mlp_weights(self):
        """
        GPTJ does not seem to use a layernorm in the mlp
        """
        Ks = []
        Vs = []
        for j in range(self.num_layers):
            K = self.model.get_parameter(f"{self.transformer_module_name}.h.{j}.mlp.fc_in.weight").T.detach()
            # fuse the layernorm
            # ln_2_weight = self.model.get_parameter(f"{self.transformer_module_name}.h.{j}.ln_2.weight").detach()
            # K = torch.einsum("oi,i -> oi", K, ln_2_weight)
            
            V = self.model.get_parameter(f"{self.transformer_module_name}.h.{j}.mlp.fc_out.weight").detach()
            Ks.append(K)
            Vs.append(V)
        
        Ks =  torch.cat(Ks)
        Vs = torch.cat(Vs)
        K_heads = Ks.reshape(self.num_layers, -1, self.hidden_dim)
        V_heads = Vs.reshape(self.num_layers, -1, self.hidden_dim)
        return K_heads, V_heads

    def get_attention_heads(self):
        qkvs = []
        for selection in "qkv":
            selection_all_layers = []
            for j in range(self.num_layers):
                # fuse the q_proj, k_proj, v_proj along the second dimension (dim=1)
                selected = self.model.get_parameter(f"{self.transformer_module_name}.h.{j}.attn.{selection}_proj.weight").detach().T
                # First concat then transpose
                ln_weight_1 = self.model.get_parameter(f"{self.transformer_module_name}.h.{j}.ln_1.weight").detach()

                selected = selected - torch.mean(selected, dim=0) 
                selected = torch.einsum("oi,i -> oi", selected, ln_weight_1)
                selection_all_layers.append(selected)
            selection_all_layers = torch.cat(selection_all_layers)
            qkvs.append(selection_all_layers)
        W_Q, W_K, W_V = qkvs
        W_O = torch.cat([self.model.get_parameter(f"{self.transformer_module_name}.h.{j}.attn.out_proj.weight") for j in range(self.num_layers)]).detach().T

        # The following 4 reshapes create two separate dimenions for the head and head size, by default they are combined
        W_V_heads = W_V.reshape(self.num_layers, self.hidden_dim, self.num_heads, self.head_size).permute(0, 2, 1, 3)
        W_O_heads = W_O.reshape(self.num_layers, self.num_heads, self.head_size, self.hidden_dim)
        W_Q_heads = W_Q.reshape(self.num_layers, self.hidden_dim, self.num_heads, self.head_size).permute(0, 2, 1, 3)
        W_K_heads = W_K.reshape(self.num_layers, self.hidden_dim, self.num_heads, self.head_size).permute(0, 2, 1, 3)
        return W_Q_heads, W_K_heads, W_V_heads, W_O_heads