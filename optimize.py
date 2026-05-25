import torch
from utils import quantize


def refine(B, A, num_bits=2, group_size=128, along_column_B=False, along_column_A=False, steps=100, lr=1e-2, method='rtn'):
    """
    Find a reparameterization B*, A* of the original B, A such that quantization error || BA - D(Q(B*))D(Q(A*)) || is minimized    

    Returns:
        tuple: Optimized (B*, A*) as detached tensors.
    """
    # learnable reparameterizations, initialized from originals
    B_ = torch.nn.Parameter(B.clone())
    A_ = torch.nn.Parameter(A.clone())
    optimizer = torch.optim.SGD([B_, A_], lr=lr)

    # to approximate the original product
    target = B @ A

    for _ in range(steps):
        # quantize current reparameterization 
        Q_B = quantize(B_, group_size=group_size, num_bits=num_bits, along_column=along_column_B, method=method)
        ##
        Q_A = quantize(A_, group_size=group_size, num_bits=num_bits, along_column=along_column_A, method=method)
        # frobenius norm between original product and quantized product
        loss = torch.norm(target - Q_B @ Q_A, p='fro')
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # report 
    with torch.no_grad():
        original_error = torch.norm(
            target - quantize(B, group_size=group_size, num_bits=num_bits, along_column=along_column_B, method=method)
                   @ quantize(A, group_size=group_size, num_bits=num_bits, along_column=along_column_A, method=method),
            p='fro'
        )
        new_error = torch.norm(target - Q_B.detach() @ Q_A.detach(), p='fro')

    print(f"ori error = {original_error:.6f}")
    print(f"new error = {new_error:.6f}")

    # detach before returning 
    return B_.detach(), A_.detach()


def lora_quant(lora_B, lora_A, rank_high=4, num_bits_high=2, group_size=128,
               along_column_B=True, along_column_A=False, method='svd', split='sqrt', ratio=0.7, opt=False):
    """
    Mixed-precision quantization of a LoRA adapter via SVD-based splitting.

    Returns:
        tuple: (B, A, num_bits, num_params)
            B, A        - quantized matrices concatenated from both sub-LoRAs
            num_bits    - total bits used
            num_params  - total params at full precision
    """

    m, r = lora_B.shape
    n    = lora_A.shape[1]

    if method in ['svd', 'ratio']:
        # reparameterize BA via SVD, truncate to rank r
        U, S, Vh = torch.linalg.svd(lora_B @ lora_A)
        U, S, Vh = U[:, :r], S[:r], Vh[:r, :]

        if method == 'ratio':
            # find smallest h where top-h singular values explain >= ratio of variance
            explained = torch.cumsum(S ** 2, dim=0) / S.pow(2).sum()
            rank_high = max((explained >= ratio).nonzero(as_tuple=True)[0][0].item(), 1)

        # distribute singular values according to split strategy
        if split == 'sqrt':
            S_B = torch.sqrt(S)
            S_A = torch.sqrt(S)
        elif split == 'B':
            S_B = S
            S_A = torch.ones_like(S)
        elif split == 'A':
            S_B = torch.ones_like(S)
            S_A = S
        else:
            raise ValueError(f"unknown split: {split}. choose from 'sqrt', 'B', 'A'")

        # split into high and low sub-LoRAs
        B_high = U[:, :rank_high] * S_B[:rank_high]
        A_high = S_A[:rank_high].unsqueeze(1) * Vh[:rank_high, :]

        B_low  = U[:, rank_high:] * S_B[rank_high:]
        A_low  = S_A[rank_high:].unsqueeze(1) * Vh[rank_high:, :]

    elif method == 'random': 
        # baseline: randomly assign dimensions to sub-LoRAs
        perm   = torch.randperm(r).tolist()

        B_high = lora_B[:, perm[:rank_high]]
        A_high = lora_A[perm[:rank_high], :]

        B_low  = lora_B[:, perm[rank_high:]]
        A_low  = lora_A[perm[rank_high:], :]

    elif method == 'norm':
        # sort dimensions by frobenius norm of outer product b_i @ a_i
        norms = torch.stack([torch.norm(lora_B[:, i].unsqueeze(1) @ lora_A[i, :].unsqueeze(0)) for i in range(r)])
        perm  = norms.argsort(descending=True).tolist()

        B_high = lora_B[:, perm[:rank_high]]
        A_high = lora_A[perm[:rank_high], :]

        B_low  = lora_B[:, perm[rank_high:]]
        A_low  = lora_A[perm[rank_high:], :]

    if opt:
        # optimize one dimension at a time to avoid mixing singular dimensions
        for i in range(rank_high):
            B_high[:, i:i+1], A_high[i:i+1, :] = refine(B_high[:, i:i+1], A_high[i:i+1, :], num_bits=num_bits_high, group_size=group_size, along_column_B=along_column_B, along_column_A=along_column_A, steps=300, lr=1e-2)

    # high sub-LoRA: RTN at num_bits_high
    B_high_q = quantize(B_high, group_size=group_size, num_bits=num_bits_high, along_column=along_column_B, method='rtn')
    A_high_q = quantize(A_high, group_size=group_size, num_bits=num_bits_high, along_column=along_column_A, method='rtn')

    # low sub-LoRA: 1-bit binary
    B_low_q = quantize(B_low, group_size=group_size, num_bits=1, along_column=along_column_B, method='bin')
    A_low_q = quantize(A_low, group_size=group_size, num_bits=1, along_column=along_column_A, method='bin')

    # concatenate sub-LoRAs back into single matrices
    B = torch.cat((B_high_q, B_low_q), dim=1)
    A = torch.cat((A_high_q, A_low_q), dim=0)

    num_bits   = rank_high * (m + n) * num_bits_high + (r - rank_high) * (m + n)
    num_params = r * (m + n)

    return B, A, num_bits, num_params
