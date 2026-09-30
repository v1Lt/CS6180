"""Follow-up: does stronger regularization (dropout 0.3 instead of 0.2) change the SwiGLU vs. GELU comparison?
Runs after pipeline.py has finished; uses each variant's best LR from the sweep."""
import json, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
from sweep import run_name

def best_val(variant, lr, seed=1337):
    p = os.path.join('runs', run_name(variant, lr, seed), 'loss_log.json')
    if not os.path.exists(p):
        return None
    log = json.load(open(p))['log']
    return min(r['val'] for r in log) if log[-1]['iter'] >= 5000 else None

while 'pipeline done' not in (open('logs/pipeline.log').read() if os.path.exists('logs/pipeline.log') else ''):
    time.sleep(30)

FLAGS = {'baseline': [], 'swiglu': ['--mlp_type=swiglu']}

def job(variant):
    cands = {lr: best_val(variant, lr) for lr in (5e-4, 1e-3, 2e-3, 3e-3, 4e-3)}
    lr = min((l for l in cands if cands[l] is not None), key=lambda l: cands[l])
    out_dir = os.path.join('runs', run_name(variant, lr, 1337) + '_do0.3')
    os.makedirs(out_dir, exist_ok=True)
    cmd = [sys.executable, '-u', 'train.py', 'config/train_shakespeare_char.py', '--compile=False', f'--out_dir={out_dir}',
           f'--learning_rate={lr}', f'--min_lr={lr/10}', '--dropout=0.3', '--save_checkpoint=False'] + FLAGS[variant]
    with open(os.path.join(out_dir, 'train.log'), 'w') as f:
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    return out_dir

with ThreadPoolExecutor(max_workers=2) as ex:
    for r in ex.map(job, FLAGS):
        print('finished', r, flush=True)
print('extra done', flush=True)
