import torch
import numpy as np
from typing import Union, Iterable


class Vocab:
    def __init__(self, vocab):
        self.vocab = vocab
    
    def items(self):
        return self.vocab

class DecisionTokenizer:
    def __init__(self, vocab_size, name):
        """
        Actually, it should just have one input vocab_size

        For some decision transformers:
        vocab_size:
        name: the name of the outputed strings
        """
        self.name = name
        self.act_dim = self.vocab_size = vocab_size
        self.vocab = Vocab(self.build_vocab())
    
    def build_vocab(self):
        return [f"{self.name} {i}" for i in np.arange(self.act_dim)]
    
    def decode(self, token_ids: Union[Iterable, int]) -> str:
        # check if token_ids is iterable
        if hasattr(token_ids, "__iter__"):
            return " ".join([self.vocab.vocab[token_id] for token_id in token_ids])
        else:
            return self.vocab.vocab[token_ids]

    def convert_ids_to_tokens(self, token_ids: Union[Iterable, int]) -> str:
        return self.decode(token_ids)


    # def vocab.items(self):


# Function that gets an action from the model using autoregressive prediction with a window of the previous 20 timesteps.
def get_action(model, states, actions, rewards, returns_to_go, timesteps):
    # This implementation does not condition on past rewards

    states = states.reshape(1, -1, model.config.state_dim)
    actions = actions.reshape(1, -1, model.config.act_dim)
    returns_to_go = returns_to_go.reshape(1, -1, 1)
    timesteps = timesteps.reshape(1, -1)

    states = states[:, -model.config.max_length :]
    actions = actions[:, -model.config.max_length :]
    returns_to_go = returns_to_go[:, -model.config.max_length :]
    timesteps = timesteps[:, -model.config.max_length :]
    padding = model.config.max_length - states.shape[1]
    # pad all tokens to sequence length
    attention_mask = torch.cat([torch.zeros(padding), torch.ones(states.shape[1])])
    attention_mask = attention_mask.to(dtype=torch.long).reshape(1, -1)
    states = torch.cat([torch.zeros((1, padding, model.config.state_dim)), states], dim=1).float()
    actions = torch.cat([torch.zeros((1, padding, model.config.act_dim)), actions], dim=1).float()
    returns_to_go = torch.cat([torch.zeros((1, padding, 1)), returns_to_go], dim=1).float()
    timesteps = torch.cat([torch.zeros((1, padding), dtype=torch.long), timesteps], dim=1)

    state_preds, action_preds, return_preds = model(
        states=states,
        actions=actions,
        rewards=rewards,
        returns_to_go=returns_to_go,
        timesteps=timesteps,
        attention_mask=attention_mask,
        return_dict=False,
    )

    return action_preds[0, -1]