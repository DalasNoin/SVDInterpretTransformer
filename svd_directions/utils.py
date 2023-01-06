import torch
from copy import deepcopy
from functools import partial
import numpy as np


def keep_k(x, k=100, absolute=True, dim=-1):
    shape = x.shape
    x_ = x
    if absolute:
        x_ = abs(x)
    values, indices = torch.topk(x_, k=k, dim=dim)
    res = torch.zeros_like(x)
    res.scatter_(dim, indices, x.gather(dim, indices))
    return res

def get_max_token_length(tokens):
  maxlen = 0
  for t in tokens:
    l = len(t)
    if l > maxlen:
      maxlen = l
  return maxlen

def pad_with_space(t, maxlen):
  spaces_to_add = maxlen - len(t)
  for i in range(spaces_to_add):
    t += " "
  return t

def convert_to_tokens(indices, tokenizer, extended, extra_values_pos, strip=True, pad_to_maxlen=False):
    if extended:
        res = [tokenizer.convert_ids_to_tokens([idx])[0] if idx < len(tokenizer) else 
               (f"[pos{idx-len(tokenizer)}]" if idx < extra_values_pos else f"[val{idx-extra_values_pos}]") 
               for idx in indices]
    else:
        res = tokenizer.convert_ids_to_tokens(indices)
    if strip:
        res = list(map(lambda x: x[1:] if x[0] == 'Ġ' else "#" + x, res))
    if pad_to_maxlen:
      maxlen = get_max_token_length(res)
      res = list(map(lambda t: pad_with_space(t, maxlen), res))
    return res


def top_tokens(v_tok, k=100, tokenizer=None, only_english=False, only_ascii=True, with_values=False, 
               exclude_brackets=False, extended=True, extra_values=None, pad_to_maxlen=False):
    if tokenizer is None:
        tokenizer = my_tokenizer
    v_tok = deepcopy(v_tok)
    ignored_indices = []
    if only_ascii:
        ignored_indices = [key for val, key in tokenizer.vocab.items() if not val.strip('Ġ').isascii()]
    if only_english: 
        ignored_indices =[key for val, key in tokenizer.vocab.items() if not (val.strip('Ġ').isascii() and val.strip('Ġ[]').isalnum())]
    if exclude_brackets:
        ignored_indices = set(ignored_indices).intersection(
            {key for val, key in tokenizer.vocab.items() if not (val.isascii() and val.isalnum())})
        ignored_indices = list(ignored_indices)
    v_tok[ignored_indices] = -np.inf
    extra_values_pos = len(v_tok)
    if extra_values is not None:
        v_tok = torch.cat([v_tok, extra_values])
    values, indices = torch.topk(v_tok, k=k)
    res = convert_to_tokens(indices, tokenizer, extended=extended, extra_values_pos=extra_values_pos,pad_to_maxlen = pad_to_maxlen)
    if with_values:
        res = list(zip(res, values.cpu().numpy()))
    return res


def top_matrix_tokens(mat, k=100, tokenizer=None, rel_thresh=None, thresh=None, 
                      sample_entries=10000, alphabetical=True, only_english=False,
                      exclude_brackets=False, with_values=True, extended=True):
    if tokenizer is None:
        tokenizer = my_tokenizer
    mat = deepcopy(mat)
    ignored_indices = []
    if only_english:
        ignored_indices = [key for val, key in tokenizer.vocab.items() if not (val.isascii() and val.strip('[]').isalnum())]
    if exclude_brackets:
        ignored_indices = set(ignored_indices).intersection(
            {key for val, key in tokenizer.vocab.items() if not (val.isascii() and val.isalnum())})
        ignored_indices = list(ignored_indices)
    mat[ignored_indices, :] = -np.inf
    mat[:, ignored_indices] = -np.inf
    cond = torch.ones_like(mat).bool()
    if rel_thresh:
        cond &= (mat > torch.max(mat) * rel_thresh)
    if thresh:
        cond &= (mat > thresh)
    entries = torch.nonzero(cond)
    if sample_entries:
        entries = entries[np.random.randint(len(torch.nonzero(cond)), size=sample_entries)]
    res_indices = sorted(entries, 
                         key=lambda x: x[0] if alphabetical else -mat[x[0], x[1]])
    res = [*map(partial(convert_to_tokens, extended=extended, tokenizer=tokenizer), res_indices)]
            
    if with_values:
        res_ = []
        for (x1, x2), (i1, i2) in zip(res, res_indices):
            res_.append((x1, x2, mat[i1][i2].item()))
        res = res_    
    return res