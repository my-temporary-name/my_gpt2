from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import math

# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# From original transformer model gpt2 only have decoder part and also the cross-attention is not used.
# Also there's reshuffling layer-norms and Additional Layer normalization is added right before the soft-max layer.

class CausalSelfAttention(nn.Module): # this class combined the self-attention mechanism and multi-head attention mechanism in one class

    def __init__(self, config):
        super().__init__()

        assert config.n_embd % config.n_head == 0 # n_emb is the embedding size and n_head is the number of heads in the multi-head attention mechanism 
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
        qkv = self.c_attn(x) # qkv is the query, key and value projections for all heads
        q,k,v = qkv.split(self.n_embd, dim=2) # Splitting the qkv into query, key and value projections

        k = k.view(B,T,self.n_head, C//self.n_head).transpose(1,2) # Splitting the key into the number of heads and transposing it (B,nh,T,hs)
        q = q.view(B,T,self.n_head, C//self.n_head).transpose(1,2) # Splitting the key into the number of heads and transposing it (B,nh,T,hs)
        v = v.view(B,T,self.n_head, C//self.n_head).transpose(1,2) # Splitting the key into the number of heads and transposing it (B,nh,T,hs)

        # attention (materializes the large (T,T) matrix for all queries and keys)
        att = (q@k.transpose(-2,-1))*(1.0 / math.sqrt(k.size(-1))) # Multiplying the query and key and scaling it by the square root of the key size
        att = att.masked_fill(self.bias[:,:,:T,:T]==0, float('-inf')) # Masking the future tokens
        att = F.softmax(att, dim=-1) # Softmax over the last dimension

        y = att@v # Multiplying the attention weights with the values (B,nh,T,T) x (B,nh,T,hs) = (B,nh,T,hs)
        y = y.transpose(1,2).contiguous().view(B,T,C) # re-assemble all head outputs side by side

        # Output Projection
        y = self.c_proj(y) # Projecting the output to the original size
        return y



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
    block_size: int = 1024 # maximum sequence length
    vocab_size: int = 50257 # number of tokens in the vocabulary i.e. 50,000 BPE merges + 256 byte tokens + 1 <|endoftext|> token
    n_layer: int = 12 # number of transformer layers
    n_head: int = 12 # number of heads in the multi-head attention mechanism
    n_embd: int = 768 # embedding dimension of each token

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

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias = False) # language model head is a linear layer with vocab_size output
    
    def forward(self,idx): 
        # idx is of shape [batch_size, sequence_length] (B,T)
        B,T = idx.size() # batch size and sequence length
        assert T<=self.config.block_size ,f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"

        # forward the token and position embeddings
        pos = torch.arange(0, T, dtype = torch.long, device =idx.device) # tensor of shape [T]
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (B,T,n_embd)
        x =tok_emb + pos_emb 

        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        
        # Forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        return logits 



    @classmethod
    def from_pretrained(cls, model_type):
        """Load pretrained GPT2 model weights from huggingface"""

        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'} # Checking if the model type is valid
        from transformers import GPT2LMHeadModel
        print("Loading weights from pretrained gpt: %s" %model_type)

        # n_layer, n_head, and n_embd are determined by the model type

        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M parameters
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M parameters
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M parameters
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M parameters
        }[model_type]

        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024  # always 1024 for GPT model checkpoint

        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict() # state_dict is the model weights
        sd_keys = sd.keys() # keys are the names of the weights
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer key, not parameters of the model

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict() 

        # copy while ensuring all of the parameters are aligned correctly and matches in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them

        # print(f"sd_keys: {sd_keys}")
        # print(f"sd_keys_hf: {sd_keys_hf}")
        # check which key is not in the sd_keys_hf
        # for k in sd_keys_hf:
        #     if k not in sd_keys:
        #         print(f"missing in sd_keys: {k}")
        
        # missing in sd_keys: lm_head.weight

        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model # return the model with the pretrained weights

# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
num__return_sequences = 5
max_length = 30


model = GPT.from_pretrained('gpt2') # Load the pretrained GPT2 model
print("Model loaded successfully!")
# print(model)

model.eval() # Set the model to evaluation mode
model.to('cuda')

# prefix tokens
import tiktoken
enc = tiktoken.get_encoding('gpt2')
tokens = enc.encode("Hello, I am NLP student in IIT") 
tokens = torch.tensor(tokens, dtype = torch.long) # (8,) # Encoding the input text
tokens = tokens.unsqueeze(0).repeat(num__return_sequences, 1) # (5,8) # Repeating the input text for the number of sequences to generate
x = tokens.to('cuda') # Moving the input to the GPU

# Generate! right now x is (B,T) where B=5 and T=8
# set see to 42
torch.manual_seed(42)
torch.cuda.manual_seed(42) 

while x.size(1) < max_length: # Generate the tokens until the maximum length is reached
    # forward the model to get the logits
    with torch.no_grad():
        logits = model(x) # (5,8,50257) # Getting the logits from the model # (B,T, vocab_size)
        # take logits as the last position
        logits = logits[:, -1, :] # (5,50257) # Taking the logits of the last position # (B,vocab_size)
        # get the probabilities
        probs = F.softmax(logits, dim=-1) # (5,50257) # Getting the probabilities from the logits
        # do top-k sampling of 50 (huggingface pipeline default)'
        # topk_probs here becomes (5,50) and topk_indices becomes (5,50) where 50 is the number of tokens to sample from the vocabulary
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
        # select a token from the top-k probnabilities
        ix = torch.multinomial(topk_probs, 1) # (5,1) # Sampling a token from the top-k probabilities
        # gather the corresponding indices
        xcol = torch.gather(topk_indices, -1, ix) # (5,1) # Getting the token from the top-k indices (B,1)
        # append to the sequence
        x = torch.cat((x, xcol), dim=1) # (5,9) # Appending the sampled token to the sequence


# print the generated sequences
for i in range(num__return_sequences):
    tokens = x[i, :max_length].tolist() # Getting the tokens from the generated sequence
    decode = enc.decode(tokens) # Decoding the tokens to get the text
    print(">", decode)


