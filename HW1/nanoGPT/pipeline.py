"""Remaining experiment phases, run automatically after the LR sweeps:
1) lr = 4e-3 for every variant whose best LR is still the largest one tried (3e-3),
2) two extra seeds (1338, 1339) at each variant's best LR,
3) the speed / parameter benchmark (GPU otherwise idle)."""
import json, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
from sweep import VARIANTS, run, run_name

def best_val(variant, lr, seed=1337):
    p = os.path.join('runs', run_name(variant, lr, seed), 'loss_log.json')
    if not os.path.exists(p):
        return None
    log = json.load(open(p))['log']
    return min(r['val'] for r in log) if log[-1]['iter'] >= 5000 else None

def count_finished(path):
    return sum('finished' in l for l in open(path)) if os.path.exists(path) else 0

def run_all(jobs, workers=3):
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for r in ex.map(run, jobs):
            print('finished', r, flush=True)

# wait for sweep.py lr_sweep (18 jobs) and sweep.py lrs 1 3e-3 (6 jobs)
while count_finished('logs/sweep_lr.log') < 18 or count_finished('logs/sweep_lr3.log') < 6:
    time.sleep(30)

lrs = [5e-4, 1e-3, 2e-3, 3e-3]
extra = [(v, 4e-3, 1337) for v in VARIANTS
         if min(lrs, key=lambda lr: best_val(v, lr) or 9e9) == 3e-3]
print('phase 1 (lr 4e-3):', extra, flush=True)
run_all(extra)

best = {}
for v in VARIANTS:
    cands = {lr: best_val(v, lr) for lr in lrs + [4e-3]}
    cands = {lr: b for lr, b in cands.items() if b is not None}
    best[v] = min(cands, key=cands.get)
print('phase 2 (seeds) best lr:', best, flush=True)
run_all([(v, best[v], s) for v in VARIANTS for s in (1338, 1339)])

print('phase 3 (benchmark)', flush=True)
subprocess.run([sys.executable, 'bench_variants.py'])
print('pipeline done', flush=True)
