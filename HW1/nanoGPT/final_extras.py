"""Last follow-ups once the GPU is free: fused-RMSNorm timing, NoPE at lr 2.5e-4 (its best LR was the smallest
one tried), and extra baseline seeds at lr 5e-3 if 5e-3 beat 4e-3."""
import json, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
from sweep import run, run_name

def done(path, marker):
    return os.path.exists(path) and marker in open(path).read()

while not (done('logs/extra_dropout.log', 'extra done') and done('logs/extra_lr5.log', 'finished')):
    time.sleep(20)

subprocess.run([sys.executable, 'bench_rmsnorm.py'])

def best_val(variant, lr, seed=1337):
    log = json.load(open(os.path.join('runs', run_name(variant, lr, seed), 'loss_log.json')))['log']
    return min(r['val'] for r in log)

jobs = [('nope', 2.5e-4, 1337)]
if best_val('baseline', 5e-3) < best_val('baseline', 4e-3):
    jobs += [('baseline', 5e-3, 1338), ('baseline', 5e-3, 1339)]
print('jobs:', jobs, flush=True)
with ThreadPoolExecutor(max_workers=3) as ex:
    for r in ex.map(run, jobs):
        print('finished', r, flush=True)
print('final extras done', flush=True)
