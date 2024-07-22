import os
import math
import time
import inspect
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
from hellaswag import render_example, iterate_examples
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# From original transformer model gpt2 only have decoder part and also the cross-attention is not used.
# Also there's reshuffling layer-norms and Additional Layer normalization is added right before the soft-max layer.

class CausalSelfAttention(nn.Module): # this class combined the self-attention mechanism and multi-head attention mechanism in one class

    def __init__(self, config):
        super().__init__()

        assert config.n_embd % config.n_head == 0 # n_emb is the embedding size and n_head is the number of heads in the multi-head attention mechanism 
                                                 # (so the embedding size should be divisible by the number of heads)
        self.c_attn = nn.Linear(config.n_embd, 3*config.n_embd) # Linear layer for the query, key and value projections for all heads, but in batch
        self.c_proj = nn.Linear(config.n_embd, config.n_embd) # Linear layer for the final output projection
        self.c_proj.NANOGPT_SCALE_INIT = 1 # Scaling the initialization of the output projection
        
        # Regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd

        # self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1,1,config.block_size, config.block_size)) # Lower triangular matrix for masking future tokens
    
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

        # att = (q@k.transpose(-2,-1))*(1.0 / math.sqrt(k.size(-1))) # Multiplying the query and key and scaling it by the square root of the key size
        # att = att.masked_fill(self.bias[:,:,:T,:T]==0, float('-inf')) # Masking the future tokens
        # att = F.softmax(att, dim=-1) # Softmax over the last dimension
        # y = att@v # Multiplying the attention weights with the values (B,nh,T,T) x (B,nh,T,hs) = (B,nh,T,hs)

        # Attention on GPT2: ( matmul + mask + softmax + dropout + matmul ) ==> 15ms
        # Flash Attention: Fused Kernel ==> 2.5ms

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

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
        self.c_proj.NANOGPT_SCALE_INIT = 1 # Scaling the initialization of the output projection


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
        # Our text first goes to ln_1, then to the self-attention mechanism, then to ln_2, and finally to the MLP
        x = x + self.mlp(self.ln_2(x)) # residual connection followed by MLP (ffn)
        # In attention 1024 sequence lined up communicated with each other & exchange info.
        # Whereas MLP happens to every single token individually and there's no communication between tokens or exchange of information between tokens.
        return x

@dataclass
class GPTConfig:
    # block_size: int = 256 # maximum sequence length
    # vocab_size: int = 50257 # number of tokens in the vocabulary i.e. 50,000 BPE merges + 256 byte tokens + 1 <|endoftext|> token
    # n_layer: int = 12 # number of transformer layers
    # n_head: int = 12 # number of heads in the multi-head attention mechanism
    # n_embd: int = 768 # embedding dimension of each token

    # # changed the default values of the parameters
    block_size: int = 256 # maximum sequence length
    vocab_size: int = 50257 # number of tokens in the vocabulary i.e. 50,000 BPE merges + 256 byte tokens + 1 <|endoftext|> token
    n_layer: int = 8 # number of transformer layers
    n_head: int = 8 # number of heads in the multi-head attention mechanism
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

        # Weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight # weight tying the token embeddings with the pre-softmax linear transformation, using this we saved 40m parameters

        # init parameters
        self.apply(self._init_weights) # initializing the weights of the model
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear): 
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2*self.config.n_layer)**-0.5 # scale by the number of layers
            torch.nn.init.normal_(module.weight, mean=0.0, std = std) # initializing the weights of the linear layer with normal distribution
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias) # initializing the bias of the linear layer with zeros
        elif isinstance(module, nn.Embedding): 
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02) 

    
    def forward(self,idx, targets= None): 
        # idx is of shape [batch_size, sequence_length] (B,T)
        B,T = idx.size() # batch size and sequence length
        assert T<=self.config.block_size ,f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"

        # forward the token and position embeddings
        pos = torch.arange(0, T, dtype = torch.long, device =idx.device) # tensor of shape [T]
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (B,T,n_embd)
        x = tok_emb + pos_emb 

        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        # Forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B,T,vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1)) # Cross-entropy flattens out the 3D (B,T,vocab_size) tensor to 2D 
                                                                                        # (B*T,vocab_size) tensor, It also flattens out the target tensor to 1D tensor
        return logits , loss


    @classmethod
    def from_pretrained(cls, model_type):
        """Load pretrained GPT2 model weights from huggingface"""

        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'} # Checking if the model type is valid
        
        print("Loading weights from pretrained gpt: %s" %model_type)
        from transformers import GPT2LMHeadModel
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

    def configure_optimizers(self, weight_decay, learning_rate, device_type):
        # start with all of the candidate parameters (that require gradients)
        param_dict = {pn: p for pn, p in self.named_parameters()} # named parameters
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad} # only parameters that require gradients

        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings, all biases and layernorm don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2] # weight tensors in matmuls + embeddings
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2] # biases and layernorm
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]

        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        if master_process:
            print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
            print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters # check if fused is available in AdamW
        use_fused = fused_available and device_type == "cuda"
        if master_process:
            print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer



# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
import tiktoken
import numpy as np


def load_tokens(filename):
    npt = np.load(filename)
    npt = npt.astype(np.int32)
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt

class DataLoaderLite:
    def __init__(self,B,T, process_rank, num_processes, split):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {'train', 'val'}

        # get the shard filenames
        data_root = "ultra2"
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no {split} shards found in {data_root}"
        if master_process:
            print(f"found {len(shards)} shards for split '{split}'")
        self.reset() # reset the state of the data loader

    def reset(self):
        # state, init at shard zero
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank # start at the beginning of the process's allocated tokens

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+B*T + 1] # +1 so we have both x and y
        x = (buf[:-1]).view(B,T) # Input is the first T tokens in the sequence
        y = (buf[1:]).view(B,T) # Labels are the next token in the sequence

        # advance the position in tensor
        self.current_position += B*T*self.num_processes
        # if loading the next batch would be out of bounds, reset
        if self.current_position + (B*T*self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = B * T * self.process_rank
        return x,y

# -----------------------------------------------------------------------------
# helper function for HellaSwag eval
# takes tokens, mask, and logits, returns the index of the completion with the lowest loss

def get_most_likely_row(tokens, mask, logits):
    # evaluate the auto-regressive loss at all positions
    shift_logits = (logits[..., :-1, :]).contiguous()
    shift_tokens = (tokens[..., 1:]).contiguous()
    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_shift_tokens = shift_tokens.view(-1)
    shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')
    shift_losses = shift_losses.view(tokens.size(0), -1)
    # now get the average loss just for the completion region (where mask == 1), in each row
    shift_mask = (mask[..., 1:]).contiguous() # we must shift mask, so we start at the last prompt token
    masked_shift_losses = shift_losses * shift_mask
    # sum and divide by the number of 1s in the mask
    sum_loss = masked_shift_losses.sum(dim=1)
    avg_loss = sum_loss / shift_mask.sum(dim=1)
    # now we have a loss for each of the 4 completions
    # the one with the lowest loss should be the most likely
    pred_norm = avg_loss.argmin().item()
    return pred_norm


# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# attempt to autodetect the device
# device = "cpu"
# if torch.cuda.is_available():
#     device = "cuda"
#     print("Using GPU")
# elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): # multi-precision support
#     device = "mps" # this is apple silicon (GPU)
#     print("Using MPS")
# print("Device: %s" %device)

# run the training loop
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

# -----------------------------------------------------------------------------
# simple launch:
# python train_gpt2.py
# DDP launch for e.g. 8 GPUs:
# torchrun --standalone --nproc_per_node=8 train_gpt2.py
# torchrun --standalone train_gpt2.py # when using the default 1 GPU
# 

# set up DDP (distributed data parallel).
# torchrun command sets the env variables RANK, LOCAL_RANK, and WORLD_SIZE
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    # use of DDP atm demands CUDA, we set the device appropriately according to rank
    assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
    init_process_group(backend='nccl') # initialize the process group for distributed training (using NCCL backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
else:
    # vanilla, non-DDP run
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    # attempt to autodetect device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"using device: {device}")

# device = 'cpu' # OVERRIDE

# added after video, pytorch can be serious about it's device vs. device_type distinction
device_type = "cuda" if device.startswith("cuda") else "cpu"

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

enc = tiktoken.get_encoding("gpt2")

# B = 8 # micro batch size
# T = 1024 # sequence length

total_batch_size = 524288 # 2**19, ~0.5M, in number of tokens as given in the paper of GPT-3 (0.5M tokens per batch)
# changed B,T
B = 32
T = 256

print(f"total batch size: {total_batch_size}, B: {B}, T: {T}")

assert total_batch_size % (B*T*ddp_world_size) == 0, "make sure total_batch_size is divisible by B*T"
grad_accum_steps = total_batch_size // (B*T*ddp_world_size) # number of steps to accumulate gradients over
if master_process:
    print(f"Total desired batch size: {total_batch_size}")
    print(f"==> Calculate gradient accumulation steps: {grad_accum_steps}")



train_loader = DataLoaderLite(B=B, T=T, process_rank = ddp_rank, num_processes = ddp_world_size, split="train") # DataLoaderLite object with batch size B and sequence length T
val_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="val") # DataLoaderLite object with batch size B and sequence length T

torch.set_float32_matmul_precision('high') # 'high' is the default


# get logits
# model = GPT(GPTConfig())
model = GPT(GPTConfig(vocab_size=50304)) # instead of 50257, we have 50304 because 50304 is even number and 50257 is odd number

model.to(device)
# model = torch.compile(model) # compile the model to TorchScript for faster execution ( it sees the entire code and optimizes it for the target device, in this case GPU)
use_compile = False # torch.compile interferes with HellaSwag eval and Generation. TODO fix

if use_compile:
    model = torch.compile(model)
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model # always contains the "raw" unwrapped model
# Speedup mainly comes from reducing Python overhead and GPU read//writes
# logits, loss = model(x, y)
# if ddp: # if using DDP, wrap the model in DDP
#     model = DDP(model, device_ids=[ddp_local_rank]) # DDP model with device id as ddp_local_rank
# raw_model = model.module if ddp else model # if using DDP, raw_model is the model.module, otherwise raw_model is the model (always contains the "raw" unwrapped model)

max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 715 # number of steps to linearly increase the learning rate
max_steps = 19073 # 19,073 steps is ~1 epoch, if data is 10B tokens and batch size 0.5M tokens , 7600

def get_lr(it):
    # 1. Linear warmup  for warmup_iters steps
    if it<warmup_steps:
        return max_lr * (it + 1 ) / warmup_steps
    # 2. if it>lr_decay_iters, return min learning rate
    if it>max_steps:
        return min_lr
    # 3. in between use cosine decay down to min learning rate
    decay_ratio = (it-warmup_steps)/(max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
    return min_lr + coeff*(max_lr - min_lr)


# optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps = 1e-8) # AdamW is better than Adam for training transformers
optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4,  device_type=device_type) 

# create the log directory we will write checkpoints to and log to
log_dir = "log"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"log.txt")
with open(log_file, "w") as f: # open for writing to clear the file
    pass

for step in range(max_steps):
    t0 = time.time()
    last_step = (step == max_steps - 1)

    # once in a while evaluate our validation loss
    if step % 250 == 0 or last_step:
        model.eval()
        val_loader.reset()
        with torch.no_grad():
            val_loss_accum = 0.0
            val_loss_steps = 20
            for _ in range(val_loss_steps):
                x, y = val_loader.next_batch()
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = model(x, y)
                loss = loss / val_loss_steps
                val_loss_accum += loss.detach()
        if ddp:
            dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
        if master_process:
            print(f"validation loss: {val_loss_accum.item():.4f}")
            with open(log_file, "a") as f:
                f.write(f"{step} val {val_loss_accum.item():.4f}\n")
            if step > 0 and (step % 5000 == 0 or last_step):
                # optionally write model checkpoints
                checkpoint_path = os.path.join(log_dir, f"model_{step:05d}.pt")
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'config': raw_model.config,
                    'step': step,
                    'val_loss': val_loss_accum.item()
                }
                # you might also want to add optimizer.state_dict() and
                # rng seeds etc., if you wanted to more exactly resume training
                torch.save(checkpoint, checkpoint_path)

    # once in a while evaluate hellaswag
    if (step % 250 == 0 or last_step) and (not use_compile):
        num_correct_norm = 0
        num_total = 0
        for i, example in enumerate(iterate_examples("val")):
            # only process examples where i % ddp_world_size == ddp_rank
            if i % ddp_world_size != ddp_rank:
                continue
            # render the example into tokens and labels
            _, tokens, mask, label = render_example(example)
            tokens = tokens.to(device)
            mask = mask.to(device)
            # get the logits
            with torch.no_grad():
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = model(tokens)
                pred_norm = get_most_likely_row(tokens, mask, logits)
            num_total += 1
            num_correct_norm += int(pred_norm == label)
        # reduce the stats across all processes
        if ddp:
            num_total = torch.tensor(num_total, dtype=torch.long, device=device)
            num_correct_norm = torch.tensor(num_correct_norm, dtype=torch.long, device=device)
            dist.all_reduce(num_total, op=dist.ReduceOp.SUM)
            dist.all_reduce(num_correct_norm, op=dist.ReduceOp.SUM)
            num_total = num_total.item()
            num_correct_norm = num_correct_norm.item()
        acc_norm = num_correct_norm / num_total
        if master_process:
            print(f"HellaSwag accuracy: {num_correct_norm}/{num_total}={acc_norm:.4f}")
            with open(log_file, "a") as f:
                f.write(f"{step} hella {acc_norm:.4f}\n")

    # once in a while generate from the model (except step 0, which is noise)
    if ((step > 0 and step % 250 == 0) or last_step) and (not use_compile):
        model.eval()
        num_return_sequences = 4
        max_length = 32
        tokens = enc.encode("Hello, I'm a language model,")
        tokens = torch.tensor(tokens, dtype=torch.long)
        tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
        xgen = tokens.to(device)
        sample_rng = torch.Generator(device=device)
        sample_rng.manual_seed(42 + ddp_rank)
        while xgen.size(1) < max_length:
            # forward the model to get the logits
            with torch.no_grad():
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = model(xgen) # (B, T, vocab_size)
                # take the logits at the last position
                logits = logits[:, -1, :] # (B, vocab_size)
                # get the probabilities
                probs = F.softmax(logits, dim=-1)
                # do top-k sampling of 50 (huggingface pipeline default)
                # topk_probs here becomes (5, 50), topk_indices is (5, 50)
                topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                # select a token from the top-k probabilities
                # note: multinomial does not demand the input to sum to 1
                ix = torch.multinomial(topk_probs, 1, generator=sample_rng) # (B, 1)
                # gather the corresponding indices
                xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
                # append to the sequence
                xgen = torch.cat((xgen, xcol), dim=1)
        # print the generated text
        for i in range(num_return_sequences):
            tokens = xgen[i, :max_length].tolist()
            decoded = enc.decode(tokens)
            print(f"rank {ddp_rank} sample {i}: {decoded}")

    # do one step of the optimization
    model.train()
    optimizer.zero_grad()
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        # added after video, this field is also used by the forward pass.
        if ddp:
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            logits, loss = model(x, y)
        # we have to scale the loss to account for gradient accumulation,
        # because the gradients just add on each successive backward().
        # addition of gradients corresponds to a SUM in the objective, but
        # instead of a SUM we want MEAN. Scale the loss here so it comes out right
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        loss.backward()
    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    # determine and set the learning rate for this iteration
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()
    if device_type == "cuda":
        torch.cuda.synchronize() # wait for the GPU to finish work
    t1 = time.time()
    dt = t1 - t0 # time difference in seconds
    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
    tokens_per_sec = tokens_processed / dt
    if master_process:
        print(f"step {step:5d} | loss: {loss_accum.item():.6f} | lr {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")
        with open(log_file, "a") as f:
            f.write(f"{step} train {loss_accum.item():.6f}\n")

if ddp:
    destroy_process_group()

# for step in range(max_steps):
#     t0 = time.time()
#     last_step = (step == max_steps - 1)

#     # once in a while evaluate our validation loss
#     if step%250 == 0 or last_step:
#         model.eval()
#         val_loader.reset()
#         with torch.no_grad(): # no need to compute gradients during validation
#             val_loss_accum = 0.0
#             val_loss_steps = 20
#             for _ in range(val_loss_steps):
#                 x, y = val_loader.next_batch()
#                 x, y = x.to(device) , y.to(device)
#                 with torch.autocast(device_type=device_type, dtype = torch.bfloat16):
#                     logits, loss = model(x , y)
#                 loss = loss / val_loss_steps
#                 val_loss_accum += loss.detach()
#         if ddp: # if using DDP, average the loss over all processes
#             dist.all_reduce(val_loss_accum , op = dist.ReduceOp.AVG)
#         if master_process:
#             print(f"validation loss: {val_loss_accum.item():.4f}")
#             with open(log_file, "a") as f:
#                 f.write(f"{step} val {val_loss_accum.item():.4f}\n")
#             if step > 0 and (step % 5000 == 0 or last_step):
#                 # optionally write model checkpoints
#                 checkpoint_path = os.path.join(log_dir, f"model_{step:05d}.pt")
#                 checkpoint = {
#                     'model': raw_model.state_dict(),
#                     'config': raw_model.config,
#                     'step': step,
#                     'val_loss': val_loss_accum.item()
#                 }
#                 # you might also want to add optimizer.state_dict() and
#                 # rng seeds etc., if you wanted to more exactly resume training
#                 torch.save(checkpoint, checkpoint_path)
    
#         # once in a while evaluate hellaswag
#     if (step % 250 == 0 or last_step) and (not use_compile):
#         num_correct_norm = 0
#         num_total = 0
#         for i, example in enumerate(iterate_examples("val")):
#             # only process examples where i % ddp_world_size == ddp_rank
#             if i % ddp_world_size != ddp_rank:
#                 continue
#             # render the example into tokens and labels
#             _, tokens, mask, label = render_example(example)
#             tokens = tokens.to(device)
#             mask = mask.to(device)
#             # get the logits
#             with torch.no_grad():
#                 with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
#                     logits, loss = model(tokens)
#                 pred_norm = get_most_likely_row(tokens, mask, logits)
#             num_total += 1
#             num_correct_norm += int(pred_norm == label)
#         # reduce the stats across all processes
#         if ddp:
#             num_total = torch.tensor(num_total, dtype=torch.long, device=device)
#             num_correct_norm = torch.tensor(num_correct_norm, dtype=torch.long, device=device)
#             dist.all_reduce(num_total, op=dist.ReduceOp.SUM)
#             dist.all_reduce(num_correct_norm, op=dist.ReduceOp.SUM)
#             num_total = num_total.item()
#             num_correct_norm = num_correct_norm.item()
#         acc_norm = num_correct_norm / num_total
#         if master_process:
#             print(f"HellaSwag accuracy: {num_correct_norm}/{num_total}={acc_norm:.4f}")
#             with open(log_file, "a") as f:
#                 f.write(f"{step} hella {acc_norm:.4f}\n")

#     # once in a while generate from the model (except step 0, which is noise)
#     if ((step > 0 and step % 250 == 0) or last_step) and (not use_compile):
#         model.eval()
#         num_return_sequences = 4
#         max_length = 32
#         tokens = enc.encode("Hello everyone")
#         tokens = torch.tensor(tokens, dtype=torch.long)
#         tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
#         xgen = tokens.to(device)
#         sample_rng = torch.Generator(device=device)
#         sample_rng.manual_seed(42 + ddp_rank)
#         while xgen.size(1) < max_length:
#             # forward the model to get the logits
#             with torch.no_grad():
#                 with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
#                     logits, loss = model(xgen) # (B, T, vocab_size)
#                 # take the logits at the last position
#                 logits = logits[:, -1, :] # (B, vocab_size)
#                 # get the probabilities
#                 probs = F.softmax(logits, dim=-1)
#                 # do top-k sampling of 50 (huggingface pipeline default)
#                 # topk_probs here becomes (5, 50), topk_indices is (5, 50)
#                 topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
#                 # select a token from the top-k probabilities
#                 # note: multinomial does not demand the input to sum to 1
#                 ix = torch.multinomial(topk_probs, 1, generator=sample_rng) # (B, 1)
#                 # gather the corresponding indices
#                 xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
#                 # append to the sequence
#                 xgen = torch.cat((xgen, xcol), dim=1)
#         # print the generated text
#         for i in range(num_return_sequences):
#             tokens = xgen[i, :max_length].tolist()
#             decoded = enc.decode(tokens)
#             print(f"rank {ddp_rank} sample {i}: {decoded}")

#     # set the model to training mode
#     model.train() 
#     optimizer.zero_grad() # always start with zero gradients (otherwise they accumulate)
#     loss_accum = 0.0
#     for micro_step in range(grad_acum_steps):
#         x,y = train_loader.next_batch()
#         x,y = x.to(device), y.to(device)

#         if ddp:
#             model.require_backward_grad_sync = (micro_step == grad_acum_steps - 1)
#         with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
#             logits, loss = model(x, y)
#             # import code; code.interact(local=locals())
#         loss = loss / grad_acum_steps # scale the loss to average over the batch
#         loss_accum += loss.detach() # accumulate the loss
#         loss.backward() # backpropagate to compute the gradients

#     if ddp:
#         dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG) # average the loss over all processes

#     norm = torch.nn.utils.clip_grad_norm_(model.parameters() ,1.0) # clip the gradients to prevent them from exploding (norm is the total norm of the gradients)

#     # determine and set the learning rate for this iteration
#     lr = get_lr(step)
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr

#     optimizer.step() # update the weights of the model using the computed gradients and the optimizer 

#     if device_type == "cuda":
#         torch.cuda.synchronize() # wait for the GPU to finish work

#     t1 = time.time()
#     dt = (t1 - t0) * 1000 # in milliseconds is time taken to process the batch in milliseconds
#     # tokens per second
#     tokens_processed = train_loader.B * train_loader.T * grad_acum_steps * ddp_world_size
#     tokens_per_sec = tokens_processed / dt  # number of tokens processed per second

#     if master_process:
#         print(f"step {step:5d} | loss: {loss_accum.item():.6f} | lr {lr:.4e} | norm: {norm:.4f} | dt: {dt/1000:.4f}s | tok/sec: {tokens_per_sec:.2f}")# loss is a single scalar tensor
#         with open(log_file, "a") as f:
#             f.write(f"{step} train {loss_accum.item():.6f}\n")

# if ddp:
#     destroy_process_group() # clean up the process group

# # print(logits.shape) # (4,32,50257) # (B,T,vocab_size)
# # print(loss) # -log(1/50257) = 10.82 --> we got tensor(10.8756, grad_fn=<NllLossBackward0>) which is close to 10.82


