import os
import multiprocessing as mp
import numpy as np
import pandas as pd
import tiktoken
from tqdm import tqdm

# Constants and Local Directory Setup
local_dir = "ultra_textbooks"
shard_size = int(1e8)  # 100M tokens per shard
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

# Load CSV File
data_path = os.path.join(os.path.dirname(__file__), 'train_1.csv')
df = pd.read_csv(data_path)

# Initialize the tokenizer
enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens['<|endoftext|>']  # end of text token

def tokenize(text):
    tokens = [eot]  # Start with the special EOT token
    tokens.extend(enc.encode_ordinary(text))
    tokens_np = np.array(tokens, dtype=np.uint16)
    return tokens_np

def write_datafile(filename, tokens_np):
    np.save(filename, tokens_np)

# Tokenize documents and write to output shards
nprocs = max(1, os.cpu_count() // 2)
with mp.Pool(nprocs) as pool:
    shard_index = 0
    all_tokens_np = np.empty((shard_size,), dtype=np.uint16)
    token_count = 0
    progress_bar = None

    for text in df['text']:  # Assuming 'text_column' holds the text data
        tokens = tokenize(text)
        if token_count + len(tokens) < shard_size:
            all_tokens_np[token_count:token_count+len(tokens)] = tokens
            token_count += len(tokens)
            if progress_bar is None:
                progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"Shard {shard_index}")
            progress_bar.update(len(tokens))
        else:
            split = "val" if shard_index == 0 else "train"
            filename = os.path.join(DATA_CACHE_DIR, f"shakespeare_{split}_{shard_index:06d}.npy")
            remainder = shard_size - token_count
            progress_bar.update(remainder)
            all_tokens_np[token_count:token_count+remainder] = tokens[:remainder]
            write_datafile(filename, all_tokens_np)
            shard_index += 1
            progress_bar = None
            all_tokens_np[0:len(tokens)-remainder] = tokens[remainder:]
            token_count = len(tokens) - remainder

    if token_count != 0:
        split = "val" if shard_index == 0 else "train"
        filename = os.path.join(DATA_CACHE_DIR, f"shakespeare_{split}_{shard_index:06d}.npy")
        write_datafile(filename, all_tokens_np[:token_count])

