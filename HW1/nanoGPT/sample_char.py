"""Sample from a character-level checkpoint (like sample.py, but without the tiktoken dependency)."""
import pickle, sys
import torch
from model import GPT, GPTConfig

out_dir = sys.argv[1] if len(sys.argv) > 1 else 'runs/baseline_lr1e-3'
torch.manual_seed(42)
ckpt = torch.load(f'{out_dir}/ckpt.pt', map_location='cuda')
model = GPT(GPTConfig(**ckpt['model_args']))
model.load_state_dict({k.removeprefix('_orig_mod.'): v for k, v in ckpt['model'].items()})
model.eval().cuda()
meta = pickle.load(open('data/shakespeare_char/meta.pkl', 'rb'))
stoi, itos = meta['stoi'], meta['itos']
x = torch.tensor([[stoi['\n']]], device='cuda')
with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
    y = model.generate(x, max_new_tokens=500, temperature=0.8, top_k=200)
print(f"(checkpoint from iter {ckpt['iter_num']}, val loss {ckpt['best_val_loss']:.4f})")
print(''.join(itos[i] for i in y[0].tolist()))
