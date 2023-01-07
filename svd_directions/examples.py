import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import svd_directions

# model, tokenizer, emb, device = svd_directions.get_model_tokenizer_embedding()
# my_tokenizer = tokenizer  # ?
# num_layers, num_heads, hidden_dim, head_size = svd_directions.get_model_info(model)
# all_tokens = [tokenizer.decode([i]) for i in range(tokenizer.vocab_size)]

# K,V = svd_directions.get_mlp_weights(model, num_layers = num_layers, hidden_dim = hidden_dim)
# W_Q_heads, W_K_heads, W_V_heads, W_O_heads = svd_directions.get_attention_heads(model, num_layers=num_layers, hidden_dim=hidden_dim, num_heads=num_heads, head_size = head_size)

svd_transformer = svd_directions.SVDTransformer(model_name="gpt2-medium")

svd_transformer.OV_top_singular_vectors(layer_idx=22, head_idx=10,k=20, N_singular_vectors=15, use_visualization=True)
svd_transformer.random_top_singular_vectors()
svd_transformer.plot_singular_value_distribution(layer_idx = 22, head_idx = 10)
svd_transformer.MLP_K_top_singular_vectors(layer_idx = 22, k=20, N_singular_vectors= 50)


svd_transformer.plot_all_MLP_singular_values(use_log_scale=False)

# OV_top_singular_vectors(W_V_heads, W_O_heads, emb, layer_idx=22, head_idx=15,N_singular_vectors=15,k=20, all_tokens = all_tokens)

# OV_top_singular_vectors(W_V_heads, W_O_heads, emb,layer_idx=22, head_idx=8,N_singular_vectors=15,k=20, all_tokens = all_tokens)

# OV_top_singular_vectors(W_V_heads, W_O_heads, emb,layer_idx=22, head_idx=3,N_singular_vectors=15,k=20, all_tokens = all_tokens)

# OV_top_singular_vectors(W_V_heads, W_O_heads, emb,layer_idx=22, head_idx=2,N_singular_vectors=15,k=20, all_tokens = all_tokens)


# random_top_singular_vectors(emb, N=20, k=20, all_tokens = all_tokens)

# OV_top_singular_vectors(W_V_heads, W_O_heads, emb,layer_idx=22, head_idx=10,N_singular_vectors=64,k=100, all_tokens = all_tokens)

# plot_singular_value_distribution(W_V_heads, W_O_heads,layer_idx = 22, head_idx = 10)
# plot_singular_value_distribution(W_V_heads, W_O_heads,layer_idx = 22, head_idx = 10,max_rank = 64)

# OV_top_singular_vectors(W_V_heads, W_O_heads, emb,layer_idx=22, head_idx=10,N_singular_vectors=15,k=20, with_negative=True, all_tokens = all_tokens)

# OV_top_singular_vectors(W_V_heads, W_O_heads, emb,layer_idx=22, head_idx=3,N_singular_vectors=15,k=20, with_negative=True, all_tokens = all_tokens)

# MLP_K_top_singular_vectors(K, emb,layer_idx = 22, k=20, N_singular_vectors= 50, all_tokens = all_tokens)


# plot_all_MLP_singular_vectors(K)

# MLP_K_top_singular_vectors(K, emb,layer_idx=20, k=20, N_singular_vectors= 50, with_negative=True, all_tokens = all_tokens)

# MLP_V_top_singular_vectors(layer_idx=16, k=20, N_singular_vectors= 50, all_tokens = all_tokens) 

# MLP_V_top_singular_vectors(layer_idx=17, k=20, N_singular_vectors= 50, all_tokens = all_tokens)

# MLP_V_top_singular_vectors(layer_idx=18, k=20, N_singular_vectors= 50, all_tokens = all_tokens)

# MLP_V_top_singular_vectors(layer_idx=15, k=20, N_singular_vectors= 50, with_negative=True, all_tokens = all_tokens)

# OV_top_singular_vectors(W_V_heads, W_O_heads, emb, layer_idx=19, head_idx=5,N_singular_vectors=15,k=20, with_negative=True, all_tokens = all_tokens)

# MLP_K_top_singular_vectors(K, emb,layer_idx=2, k=20, N_singular_vectors= 50, with_negative=True, all_tokens = all_tokens)