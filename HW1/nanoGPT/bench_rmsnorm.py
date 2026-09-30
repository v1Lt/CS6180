"""Is the RMSNorm slowdown only an implementation artifact? Compare LayerNorm, the unfused RMSNorm used in the
experiments, and PyTorch's fused F.rms_norm, all in the full training step."""
import time
import torch
import torch.nn.functional as F
import model as M
from model import GPT, GPTConfig

def unfused(self, x):
    xf = x.float()
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
    return xf.type_as(x) * self.weight

def fused(self, x):
    return F.rms_norm(x, (x.size(-1),), self.weight, self.eps)

base = dict(n_layer=6, n_head=6, n_embd=384, block_size=256, vocab_size=65, bias=False, dropout=0.2)
for name, norm, fwd in [('layernorm', 'layernorm', None), ('rmsnorm (unfused, used in experiments)', 'rmsnorm', unfused),
                        ('rmsnorm (fused F.rms_norm)', 'rmsnorm', fused)]:
    if fwd is not None:
        M.RMSNorm.forward = fwd
    torch.manual_seed(0)
    model = GPT(GPTConfig(**base, norm_type=norm)).cuda()
    opt = model.configure_optimizers(0.1, 1e-3, (0.9, 0.99), 'cuda')
    x = torch.randint(0, 65, (64, 256), device='cuda')
    ts = []
    for it in range(260):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            _, loss = model(x, x)
        loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize(); ts.append(time.perf_counter() - t0)
    print(f'{name:40s} {1000 * sorted(ts[60:])[100]:.1f} ms/iter', flush=True)
