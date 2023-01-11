import svd_gptj
import svd_transformer

def apply_svd(model_name: str, *args, **kwargs) -> svd_transformer.SVDTransformer:
    """
    Returns the correct svd object, they differ because models have different layernames and structures. 
    E.g. layernorm can be at different positions in the transformer, which is important for "folding" it into the weights
    
    TODO: there is a doubling of logic here with svd_transformer.get_model_tokenizer_embedding, should be merged together
    there should be a more clever way of selecting the right class
    
    idea: def load_model function here for each model and give it as a parameter to SVDTransformer
    """
    if "gpt-j" in model_name or "gptj" in model_name:
        return svd_gptj.SVDGPTJ(model_name, *args, **kwargs)
    else:
        return svd_transformer.SVDTransformer(model_name, *args, **kwargs)