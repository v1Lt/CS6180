"""Numerical check of the Problem 1 answers (output size, im2col form, input gradient) against PyTorch autograd."""
import torch
import torch.nn.functional as F

torch.manual_seed(0)
X = torch.randn(4, 4, requires_grad=True)
K = torch.randn(3, 3)
G = torch.randn(2, 2)                       # dL/dY
positions = [(0, 0), (0, 1), (1, 0), (1, 1)]

Y = F.conv2d(X[None, None], K[None, None])[0, 0]
print('output size:', tuple(Y.shape))

# im2col: 9 x 4 matrix, one column per 3x3 patch
X_col = torch.stack([X[p:p + 3, q:q + 3].reshape(-1) for p, q in positions], dim=1)
print('im2col GEMM == conv2d:', torch.allclose((K.reshape(1, 9) @ X_col).reshape(2, 2), Y))

# equivalent 4 x 16 matrix M with vec(Y) = M vec(X)
M = torch.zeros(4, 16)
for r, (p, q) in enumerate(positions):
    for a in range(3):
        for b in range(3):
            M[r, 4 * (p + a) + (q + b)] = K[a, b]
print('M vec(X) == conv2d:', torch.allclose((M @ X.reshape(16)).reshape(2, 2), Y))

(Y * G).sum().backward()                    # autograd reference for dL/dX
full = F.conv2d(F.pad(G, (2, 2, 2, 2))[None, None], torch.flip(K, [0, 1])[None, None])[0, 0]
print('dL/dX == pad(G) * rot180(K):', torch.allclose(X.grad, full, atol=1e-6))
print('dL/dX == M^T vec(G):', torch.allclose(X.grad, (M.T @ G.reshape(4)).reshape(4, 4), atol=1e-6))

dX_col = K.reshape(9, 1) @ G.reshape(1, 4)  # im2col backward, then col2im (scatter-add)
dX = torch.zeros(4, 4)
for c, (p, q) in enumerate(positions):
    dX[p:p + 3, q:q + 3] += dX_col[:, c].reshape(3, 3)
print('dL/dX == col2im(k g^T):', torch.allclose(dX, X.grad, atol=1e-6))
