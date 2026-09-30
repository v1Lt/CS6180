"""Run the homework experiments: every variant x learning rate (x seed), 2 jobs at a time on one GPU."""
import json, os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor

PY = sys.executable
VARIANTS = {
    'baseline': [],
    'rmsnorm':  ['--norm_type=rmsnorm'],
    'swiglu':   ['--mlp_type=swiglu'],
    'nope':     ['--pos_type=none'],
    'rope':     ['--pos_type=rope'],
    'gqa':      ['--n_kv_head=3'],
}

def run_name(variant, lr, seed):
    lr_str = f"{lr:.1e}".replace('.0e', 'e').replace('e-0', 'e-')  # 1e-3, 2.5e-4, ...
    return f"{variant}_lr{lr_str}" + ("" if seed == 1337 else f"_s{seed}")

def done(out_dir):
    p = os.path.join(out_dir, 'loss_log.json')
    if not os.path.exists(p):
        return False
    with open(p) as f:
        return json.load(f)['log'][-1]['iter'] >= 5000

def run(job):
    variant, lr, seed = job
    out_dir = os.path.join('runs', run_name(variant, lr, seed))
    if done(out_dir):
        return out_dir + ' (cached)'
    os.makedirs(out_dir, exist_ok=True)
    cmd = [PY, '-u', 'train.py', 'config/train_shakespeare_char.py', '--compile=False',
           f'--out_dir={out_dir}', f'--learning_rate={lr}', f'--min_lr={lr/10}', f'--seed={seed}',
           '--save_checkpoint=False'] + VARIANTS[variant]
    with open(os.path.join(out_dir, 'train.log'), 'w') as f:
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    return out_dir

if __name__ == '__main__':
    # usage: python sweep.py lr_sweep  |  python sweep.py lrs <workers> <lr> ...  |  python sweep.py seeds <variant>=<lr> ...
    mode = sys.argv[1]
    workers = 2
    if mode == 'lr_sweep':
        jobs = [(v, lr, 1337) for v in VARIANTS for lr in (5e-4, 1e-3, 2e-3)]
        jobs.sort(key=lambda j: j == ('baseline', 1e-3, 1337)) # the first baseline run was launched by hand
    elif mode == 'lrs':
        workers = int(sys.argv[2])
        jobs = [(v, float(lr), 1337) for lr in sys.argv[3:] for v in VARIANTS]
    else:
        jobs = []
        for arg in sys.argv[2:]:
            v, lr = arg.split('=')
            jobs += [(v, float(lr), s) for s in (1338, 1339)]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for r in ex.map(run, jobs):
            print('finished', r, flush=True)
