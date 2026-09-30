"""Figures for Problem 2 (RoPE score vs. distance) and Problem 3 (GEC architecture sketch)."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / 'results'  # HW1/results

# ---------------- Problem 2: RoPE ----------------
d, b = 128, 10000.0
theta = b ** (-2 * np.arange(d // 2) / d)
delta = np.arange(0, 65537)
A = np.cos(np.outer(delta, theta)).mean(1)

fig, ax = plt.subplots(2, 1, figsize=(8, 6.6))
ax[0].plot(delta, A, lw=0.35, color='k')
ax[0].set_xlabel(r'$|i-j|$'); ax[0].set_ylabel(r'$A_{i,j}$'); ax[0].set_xlim(0, 65536)
ax[0].set_title(r'RoPE score for $q_i=k_j=\frac{1}{\sqrt{128}}\mathbf{1}$ ($d=128$, base $b=10000$): linear x-axis', fontsize=10)
ax[0].axhline(0, color='k', lw=0.5); ax[0].grid(alpha=.3)
ax[1].semilogx(delta[1:], A[1:], lw=0.45, color='k')
ax[1].set_xlabel(r'$|i-j|$ (log scale)'); ax[1].set_ylabel(r'$A_{i,j}$'); ax[1].set_title('Same curve, log x-axis', fontsize=10)
ax[1].axhline(0, color='k', lw=0.5); ax[1].grid(alpha=.3, which='both')
plt.tight_layout(); plt.savefig(OUT / 'rope_attention.png', dpi=160); plt.close()

print('A(1..5)', np.round(A[1:6], 4), ' half-point', np.argmax(A < 0.5), ' first zero', np.argmax(A < 0))
tail = A[3000:]
print(f'|i-j|>=3000: mean {tail.mean():.4f} std {tail.std():.4f} min {tail.min():.4f} max {tail.max():.4f}')

# ---------------- Problem 3: GEC architecture (black and white) ----------------
fig, ax = plt.subplots(figsize=(9, 7.2)); ax.set_xlim(0, 10); ax.set_ylim(0, 10.4); ax.axis('off')

def box(x, y, w, h, t, fs=8.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.03,rounding_size=0.08', fc='white', ec='k', lw=0.9))
    ax.text(x + w / 2, y + h / 2, t, ha='center', va='center', fontsize=fs, color='k')

def arr(x1, y1, x2, y2):
    ax.annotate('', (x2, y2), (x1, y1), arrowprops=dict(arrowstyle='->', lw=1, color='k'))

ax.text(2.5, 10.1, 'Encoder (N = 6 layers)', ha='center', fontsize=10, weight='bold')
box(0.6, 0.2, 3.8, 0.6, 'Source (erroneous): "She go to school yesterday."', fs=8)
box(0.6, 1.2, 3.8, 0.6, 'SentencePiece BPE tokenizer (32k, byte fallback)', fs=8)
box(0.6, 2.2, 3.8, 0.6, 'Token embedding (d = 512), shared')
ax.add_patch(FancyBboxPatch((0.4, 3.2), 4.2, 4.4, boxstyle='round,pad=0.05', fc='none', ec='k', lw=1.2, ls='--'))
box(0.7, 3.4, 3.6, 0.55, 'RMSNorm (pre-norm)')
box(0.7, 4.1, 3.6, 0.7, 'Bidirectional multi-head self-attention\n(8 heads x 64, RoPE on q/k)', fs=8)
box(0.7, 5.0, 3.6, 0.55, 'RMSNorm (pre-norm)')
box(0.7, 5.7, 3.6, 0.7, 'Feed-forward: SwiGLU (d_ff = 1408)\n+ dropout 0.1', fs=8)
box(0.6, 8.0, 3.8, 0.6, 'Final RMSNorm -> encoder states H (n x d)')
for y1, y2 in [(0.8, 1.2), (1.8, 2.2), (2.8, 3.4), (7.6, 8.0)]:
    arr(2.5, y1, 2.5, y2)

ax.text(7.5, 10.1, 'Decoder (N = 6 layers)', ha='center', fontsize=10, weight='bold')
box(5.6, 0.2, 3.8, 0.6, 'Target shifted right: <s> She went to school ...', fs=8)
box(5.6, 1.2, 3.8, 0.6, 'Token embedding (shared / tied with output layer)', fs=8)
ax.add_patch(FancyBboxPatch((5.4, 2.0), 4.2, 5.6, boxstyle='round,pad=0.05', fc='none', ec='k', lw=1.2, ls='--'))
box(5.7, 2.2, 3.6, 0.9, 'RMSNorm + causal (masked)\nmulti-head self-attention')
box(5.7, 3.4, 3.6, 1.0, 'RMSNorm + cross-attention\n(queries: decoder; keys/values: H)')
box(5.7, 4.7, 3.6, 0.9, 'RMSNorm + feed-forward (SwiGLU)')
box(5.7, 5.9, 3.6, 0.9, 'optional copy / pointer gate\n(copy source tokens directly)')
box(5.6, 8.0, 3.8, 0.6, 'Final RMSNorm -> Linear (d -> |V|) -> softmax', fs=8)
box(5.6, 9.0, 3.8, 0.6, 'Output: corrected sentence (beam search)')
for y1, y2 in [(0.8, 1.2), (1.8, 2.2), (7.6, 8.0), (8.6, 9.0)]:
    arr(7.5, y1, 7.5, y2)
ax.annotate('', (5.7, 3.9), (4.4, 8.3), arrowprops=dict(arrowstyle='->', lw=1.2, color='k', connectionstyle='arc3,rad=-0.25'))
ax.text(4.55, 5.2, 'H', color='k', fontsize=11, weight='bold')
plt.tight_layout(); plt.savefig(OUT / 'gec_architecture.png', dpi=170); plt.close()
