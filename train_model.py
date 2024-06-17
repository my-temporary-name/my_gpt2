from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F

# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# From original transformer model gpt2 only have decoder part and also the cross-attention is not used.
# Also there's reshuffling layer-norms and Additional Layer normalization is added right before the soft-max layer.

class CausalSelfAttention(nn.Module): # this class combined the self-attention mechanism and multi-head attention mechanism in one class

    def __init__(self, config):
        super().__init__()

        assert config.n_emb % config.n_head == 0 # n_emb is the embedding size and n_head is the number of heads in the multi-head attention mechanism 
                                                 # (so the embedding size should be divisible by the number of heads)
        self.c_attn = nn.Linear(config.n_embd, 3*config.n_embd) # Linear layer for the query, key and value projections for all heads, but in batch
        self.c_proj = nn.Linear(config.n_embd, config.n_embd) # Linear layer for the final output projection
        # Regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd

        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1,1,config.block_size, config.block_size)) # Lower triangular matrix for masking future tokens
    
    def forward(self,x):
        B, T, C = x.size() # batch size, Sequence length, Embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dimension
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # eg: in GPT-2 (124M), n_head=12, hs=64, so nh*hs = C = 768 channels in Transformer (channels is also called as hidden size)
        


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__() # Inheriting from the parent class nn.Module
        self.c_fc = nn.Linear(config.n_embd, 4*config.n_embd) # Fully connected layer for the first part of the MLP which takes the input and projects it to 4 times the size of the input
        self.gelu = nn.GELU(approximate='tanh') # GELU activation function
        self.c_proj = nn.Linear(4*config.n_embd, config.n_embd) # Fully connected layer for the second part of the MLP which projects the output of the previous layer to the original size

    def forward(self,x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)

        return x


# Block is basically a transformer block which consists of a self-attention mechanism and a feed-forward neural network (decoder part)
class Block(nn.Module):

    def __init__(self,config):
        super().__init__()
        
        self.ln_1 = nn.LayerNorm(config.n_embd) # Layer normalization before the self-attention
        self.attn = CausalSelfAttention(config) # Self-attention mechanism
        self.ln_2 = nn.LayerNorm(config.n_embd) # Layer normalization after the self-attention
        self.mlp = MLP(config) # Multi-layer perceptron for each position
    
    # forward pass of the block, the input x is the sequence of embeddings and return is the updated sequence of embeddings
    def forward(self,x):
        x = x + self.attn(self.ln_1(x)) # residual connection followed by self-attention
        x = x + self.mlp(self.ln_2(x)) # residual connection followed by MLP (ffn)

        return x


@dataclass
class GPTConfig:
    block_size: int = 256
    vocab_size: int = 65
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 256

class GPT(nn.Module): # Kind of skeleton of the model
    
    def __init__(self,config):
        super().__init__()
        self.config = config

        # transformer is the main container and it have further sub-modules like wte, wpe, h, ln_f
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd), # token embedding weights
            wpe = nn.Embedding(config.block_size, config.n_embd), # positional embedding weights
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]), # transformer blocks as a list of n_layer (h is hidden layer)
            ln_f = nn.LayerNorm(config.n_embd), # final layer normalization before the softmax
        ))

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias = False) # language model head is a simple linear layer which projects the final hidden states to the vocab size


