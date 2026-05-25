import math
import torch


def RTN_quant(w, num_bits=8, group_size=-1, along_column=False):
    """
    Round-to-nearest uniform quantization followed by dequantization
    
    Returns:
        torch.Tensor: Dequantized weights, same shape as input
    """
    
    if along_column:
        w = w.T
    org_w_shape = w.shape
    # form groups
    if group_size > 0:
        w = w.reshape(-1, group_size)
    # per-group scale and zero-point
    max_val = w.amax(dim=1, keepdim=True)
    min_val = w.amin(dim=1, keepdim=True)
    
    max_int = 2**num_bits - 1
    min_int = 0
    
    scales = (max_val - min_val).clamp(min=1e-5) / max_int
    zeros = (-torch.round(min_val / scales)).clamp_(min_int, max_int)

    # quantize
    w_int = torch.clamp(torch.round(w / scales) + zeros, min_int, max_int)
    # dequantize
    w_dequant = (w_int - zeros) * scales

    # restore shape
    w_dequant = w_dequant.reshape(org_w_shape)

    if along_column:
        w_dequant = w_dequant.T

    return w_dequant


def one_bit_quant(w, group_size, along_column=False):
    """
    Binary quantize-dequantize a weight matrix group-wise.
    
    Returns:
        torch.Tensor: Dequantized weights, same shape as input.
    """

    if along_column:
        w = w.T

    original_shape = w.shape
    total_elements = w.numel()

    # flatten and pad to make divisible by group_size
    flat = w.contiguous().view(-1)
    num_groups = math.ceil(total_elements / group_size)
    pad_size = num_groups * group_size - total_elements
    if pad_size > 0:
        flat = torch.cat([flat, torch.zeros(pad_size, device=w.device)])
    # form groups
    groups = flat.view(num_groups, group_size)
    # compute per-group scale and binarize
    scale = groups.abs().mean(dim=1, keepdim=True)
    w_dequant = torch.sign(groups) * scale
    # remove padding and restore shape
    w_dequant = w_dequant.view(-1)[:total_elements].view(original_shape)

    if along_column:
        w_dequant = w_dequant.T

    return w_dequant


def quantize(w, group_size, num_bits, along_column=False, method='rtn'):
    if method == 'bin':
        assert num_bits == 1, 'must quantize to 1 bit if using binary quantization'
        return one_bit_quant(w, group_size, along_column)
    elif method == 'rtn':
        return RTN_quant(w, num_bits, group_size, along_column)
