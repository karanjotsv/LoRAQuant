import torch
from utils import quant, complex_quant


# -- FFT-guided LoRA quantization ----------------------------------------------

def lora_quant_fft(B, A, bits_hi=2, gs=128, col_B=True, col_A=False,
                   ratio=None, rh=4):
    # 1D FFT along rank dim, mixed-precision complex channel quantization
    # key identity: ifft(B, dim=1, norm='ortho') @ fft(A, dim=0, norm='ortho') = B @ A
    m, r = B.shape
    n    = A.shape[1]

    ru = r // 2 + 1  # unique channels due to conjugate symmetry

    Bf = torch.fft.ifft(B.float(), norm='ortho', dim=1)  # (m, r) complex
    Af = torch.fft.fft (A.float(), norm='ortho', dim=0)  # (r, n) complex
    Bu, Au = Bf[:, :ru], Af[:ru, :]

    if ratio is not None:
        energy    = Bu.abs().norm(dim=0) * Au.abs().norm(dim=1)
        explained = energy.cumsum(0) / energy.sum()
        hits      = (explained >= ratio).nonzero(as_tuple=True)[0]
        rh        = max(hits[0].item() if len(hits) > 0 else ru - 1, 1)

    rh = min(rh, ru - 1)  # keep at least one channel in the low group

    Bh = complex_quant(Bu[:, :rh], gs, bits_hi, col_B, 'rtn')
    Ah = complex_quant(Au[:rh, :], gs, bits_hi, col_A, 'rtn')
    Bl = complex_quant(Bu[:, rh:], gs, 1,       col_B, 'bin')
    Al = complex_quant(Au[rh:, :], gs, 1,       col_A, 'bin')

    Bu_q = torch.cat([Bh, Bl], dim=1)
    Au_q = torch.cat([Ah, Al], dim=0)

    # imag parts at k=0 (and Nyquist when r is even) are structurally zero for
    # real inputs: force them clean and don't charge bits for them below
    real_ch = [0] + ([ru - 1] if r % 2 == 0 else [])
    for k in real_ch:
        Bu_q[:, k] = Bu_q[:, k].real
        Au_q[k, :] = Au_q[k, :].real

    Bff = torch.zeros(m, r, dtype=torch.complex64, device=B.device)
    Aff = torch.zeros(r, n, dtype=torch.complex64, device=A.device)
    Bff[:, :ru] = Bu_q
    Aff[:ru, :] = Au_q

    if r > 2:
        # X[r-k] = conj(X[k]) for k = 1 .. (r-1)//2
        src = torch.arange(1, (r + 1) // 2, device=B.device)
        dst = r - src
        Bff[:, dst] = Bff[:, src].conj()
        Aff[dst, :] = Aff[src, :].conj()

    B_q = torch.fft.fft (Bff, norm='ortho', dim=1).real.to(B.dtype)
    A_q = torch.fft.ifft(Aff, norm='ortho', dim=0).real.to(A.dtype)

    # 2 real components per complex channel, minus the structurally-zero imag
    # channels: k=0 sits in the high group, Nyquist in the low group
    nbits = rh * (m + n) * 2 * bits_hi + (ru - rh) * (m + n) * 2
    nbits -= (m + n) * bits_hi
    if r % 2 == 0:
        nbits -= (m + n)
    nparams = r * (m + n)

    return B_q, A_q, nbits, nparams


# -- reparameterization refinement ---------------------------------------------

def refine(B, A, num_bits=2, gs=128, col_B=False, col_A=False,
           steps=100, lr=1e-2, method='rtn'):
    # learnable reparameterizations initialized from originals
    Bp    = torch.nn.Parameter(B.clone())
    Ap    = torch.nn.Parameter(A.clone())
    optim = torch.optim.SGD([Bp, Ap], lr=lr)
    tgt   = B @ A

    best_loss      = float('inf')
    best_B, best_A = Bp.detach().clone(), Ap.detach().clone()

    for _ in range(steps):
        Qb   = quant(Bp, group_size=gs, num_bits=num_bits, along_column=col_B, method=method)
        Qa   = quant(Ap, group_size=gs, num_bits=num_bits, along_column=col_A, method=method)
        loss = torch.norm(tgt - Qb @ Qa, p='fro')
        # track best before updating; SGD can overshoot
        with torch.no_grad():
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_B    = Bp.detach().clone()
                best_A    = Ap.detach().clone()
        optim.zero_grad()
        loss.backward()
        optim.step()

    with torch.no_grad():
        ori_err = torch.norm(
            tgt - quant(B,      gs, num_bits, col_B, method)
                @ quant(A,      gs, num_bits, col_A, method), p='fro'
        )
        new_err = torch.norm(
            tgt - quant(best_B, gs, num_bits, col_B, method)
                @ quant(best_A, gs, num_bits, col_A, method), p='fro'
        )
    print(f"ori error = {ori_err:.6f}")
    print(f"new error = {new_err:.6f}")
    return best_B, best_A


# -- SVD-based LoRA quantization -----------------------------------------------

def lora_quant_svd(B, A, rh=4, bits_hi=2, gs=128,
                   col_B=True, col_A=False, decomp='svd',
                   split='sqrt', ratio=None, opt=False):
    m, r = B.shape
    n    = A.shape[1]

    if decomp == 'svd':
        # reparameterize BA via SVD, truncate to rank r
        U, S, Vh = torch.linalg.svd(B @ A)
        U, S, Vh = U[:, :r], S[:r], Vh[:r, :]

        if ratio is not None:
            # find smallest rh where top-rh singular values explain >= ratio of variance
            explained = torch.cumsum(S ** 2, dim=0) / S.pow(2).sum()
            hits      = (explained >= ratio).nonzero(as_tuple=True)[0]
            rh        = max(hits[0].item() if len(hits) > 0 else r - 1, 1)

        if split == 'sqrt':
            sB = sA = torch.sqrt(S)
        elif split == 'B':
            sB, sA = S, torch.ones_like(S)
        elif split == 'A':
            sB, sA = torch.ones_like(S), S
        else:
            raise ValueError(f"unknown split '{split}'. choose from 'sqrt', 'B', 'A'")

        Bh = U[:, :rh] * sB[:rh]
        Ah = sA[:rh].unsqueeze(1) * Vh[:rh, :]
        Bl = U[:, rh:] * sB[rh:]
        Al = sA[rh:].unsqueeze(1) * Vh[rh:, :]

    elif decomp == 'random':
        perm    = torch.randperm(r).tolist()
        Bh, Ah  = B[:, perm[:rh]], A[perm[:rh], :]
        Bl, Al  = B[:, perm[rh:]], A[perm[rh:], :]

    elif decomp == 'norm':
        # sort rank dims by frobenius norm of their outer product b_i a_i^T
        norms   = torch.stack([torch.norm(B[:, i].unsqueeze(1) @ A[i, :].unsqueeze(0)) for i in range(r)])
        perm    = norms.argsort(descending=True).tolist()
        Bh, Ah  = B[:, perm[:rh]], A[perm[:rh], :]
        Bl, Al  = B[:, perm[rh:]], A[perm[rh:], :]

    if opt:
        for i in range(rh):
            Bh[:, i:i+1], Ah[i:i+1, :] = refine(
                Bh[:, i:i+1], Ah[i:i+1, :],
                num_bits=bits_hi, gs=gs, col_B=col_B, col_A=col_A, steps=300, lr=1e-2
            )

    Bh_q = quant(Bh, gs, bits_hi, col_B, 'rtn')
    Ah_q = quant(Ah, gs, bits_hi, col_A, 'rtn')
    Bl_q = quant(Bl, gs, 1,       col_B, 'bin')
    Al_q = quant(Al, gs, 1,       col_A, 'bin')

    B_q   = torch.cat((Bh_q, Bl_q), dim=1)
    A_q   = torch.cat((Ah_q, Al_q), dim=0)
    nbits   = rh * (m + n) * bits_hi + (r - rh) * (m + n)
    nparams = r * (m + n)

    return B_q, A_q, nbits, nparams
