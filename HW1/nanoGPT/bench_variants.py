"""Training-step throughput and parameter count of every variant (same shapes as config/train_shakespeare_char.py)."""
import json, time
import torch
from model import GPT, GPTConfig

VARIANTS = {'baseline': {}, 'rmsnorm': dict(norm_type='rmsnorm'), 'swiglu': dict(mlp_type='swiglu'),
            'nope': dict(pos_type='none'), 'rope': dict(pos_type='rope'), 'gqa': dict(n_kv_head=3)}
base = dict(n_layer=6, n_head=6, n_embd=384, block_size=256, vocab_size=65, bias=False, dropout=0.2)
torch.backends.cuda.matmul.allow_tf32 = True
res = {}
for name, kw in VARIANTS.items():
    torch.manual_seed(0)
    cfg = GPTConfig(**base, **kw)
    model = GPT(cfg).cuda()
    opt = model.configure_optimizers(0.1, 1e-3, (0.9, 0.99), 'cuda')
    x = torch.randint(0, 65, (64, 256), device='cuda')
    times = []
    for it in range(260):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            _, loss = model(x, x)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize(); times.append(time.perf_counter() - t0)
    n_kv = cfg.n_kv_head or cfg.n_head
    res[name] = dict(params=sum(p.numel() for p in model.parameters()), ms_per_iter=1000 * sorted(times[60:])[100],
                     kv_cache_values_per_token=2 * cfg.n_layer * n_kv * (cfg.n_embd // cfg.n_head))
    print(name, res[name], flush=True)
    del model, opt; torch.cuda.empty_cache()
json.dump(res, open('../results/bench.json', 'w'), indent=1)
