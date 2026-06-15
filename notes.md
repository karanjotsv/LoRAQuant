# Notes

## Quantization Basics

**What is quantization?**
Trading precision for memory. A 32-bit float like `0.7341928` gets snapped to the nearest point on a coarse grid defined by your bitwidth. You store the integer index instead of the full float, saving memory.

**The grid**
With `n` bits you have `2^n` levels, spread evenly across the weight range.
- 2-bit: 4 levels
- 4-bit: 16 levels
- 8-bit: 256 levels

**Why does it work?**
Neural networks are overparameterized and robust to small weight perturbations. Errors partially cancel out across millions of weights.

**The tradeoffs**
```
32-bit float:  0.7341928  (exact)
8-bit:         0.73       (barely noticeable)
4-bit:         0.75       (small error)
2-bit:         0.67       (visible error)
1-bit:         +S or -S   (just the sign)
```

---

## Range

The range is not fixed in advance - it is computed from the data itself:
```
min_val = minimum weight in the group
max_val = maximum weight in the group
```
The integer grid is stretched to fit exactly that range, so it is adaptive per group.

**Fixed-range quantization** (used in some hardware) assumes a fixed range like `[-1, 1]`. Simpler but worse accuracy.

---

## Scale

```
scale = (max_val - min_val) / (2^num_bits - 1)
```

Scale is the size of one interval on the grid - how much float range each integer step covers.

Example with 2-bit and range `[-0.8, 0.7]`:
```
scale = (0.7 - (-0.8)) / 3 = 1.5 / 3 = 0.5
```
Grid levels land at: -0.8, -0.3, 0.2, 0.7

Smaller scale = finer grid = less error. More bits → smaller scale → better accuracy.

---

## quantization_utils.py

### RTN(w, num_bits, group_size, along_column)

Uniform quantize-dequantize a weight matrix group-wise.

Snaps each weight to the nearest point on a uniform integer grid defined by [0, 2^num_bits - 1], then converts back to float. Each group of group_size consecutive elements gets its own scale and zero-point. group_size=-1 treats the entire matrix as one group.

**group_size significance:**
Without grouping, one scale covers the entire matrix. Outliers force a large scale, squashing small weights into few levels. Group-wise gives each chunk its own scale, limiting outlier impact. group_size=128 is the standard sweet spot used by GPTQ, AWQ, and this paper.

**along_column:**
Controls whether groups are formed row-wise or column-wise. Implemented via transpose trick - transpose, quantize row-wise, transpose back. For LoRA, B(col) + A(row) works best because SVD reparameterization concentrates singular value magnitudes into columns of B and rows of A.

**group_size=-1:**
Sentinel value meaning no grouping - entire matrix is one group. Just a convention, could have been None or 0.

**Why dequantize?**
This codebase simulates quantization to measure accuracy impact, rather than actually storing integers. Dequantizing back to float lets normal PyTorch ops run without custom CUDA kernels. The round-trip float→int→float is lossy - weights are snapped to grid points - which is where the error comes from.

**Asymmetric vs Symmetric quantization:**
- Asymmetric (used here): range is [min, max], needs zero-point to shift grid. Better accuracy.
- Symmetric: range is [-max_abs, +max_abs], zero-point always 0. Simpler, easier in hardware.

**RTN example (2-bit):**
```
w = [-0.8, -0.2, 0.3, 0.7]
scale = 0.5,  zero = 2

Quantize:   [0, 2, 3, 3]
Dequantize: [-1.0, 0.0, 0.5, 0.5]
Errors:     [0.2, 0.2, 0.2, 0.2]
```
0.3 and 0.7 both map to integer 3 - that's quantization error from having only 4 levels.

---

## Zero-point

```
zero = round(-min_val / scale)
```

Zero-point shifts the integer grid so it covers the actual weight range and maps to valid non-negative integers.

**Why we need it:**
Weights can be negative. Without zero-point, `round(w / scale)` anchors the grid at float `0.0`, pushing negative weights into negative integers which are outside the valid unsigned range `[0, 2^n - 1]`. After clamping, multiple weights collapse to the same bin - huge information loss.

Example without zero-point, weights `[-1.2, -0.4, 0.3, 0.9]`, scale=0.7:
```
round(w / scale): [-2, -1, 0, 1]  → clamped to [0,3] → [0, 0, 0, 1]
three weights collapsed to bin 0 - information destroyed
```

With zero-point=2:
```
round(w / scale) + 2: [0, 1, 2, 3]  → all valid, all distinct
```

**Why clamp zero-point?**
Numerical edge cases can push it outside `[0, 2^n-1]`. The clamp keeps it valid.

**Signed vs Unsigned integers:**
- Unsigned (used here): `min_int=0`, bins always non-negative, zero-point always needed
- Signed: range `[-2^(n-1), 2^(n-1)-1]`, bins can be negative, no zero-point needed, simpler math

---

## RTN full example (2-bit, with and without zero-point)

Weights: `[-1.2, -0.4, 0.3, 0.9]`, scale=0.7, zero=2

**Without zero-point:**
```
integers: [-2, -1, 0, 1] → clamped → [0, 0, 0, 1]
dequantized: [-1.4, -1.4, -1.4, -0.7]   ← three weights lost
```

**With zero-point:**
```
integers:    [0, 1, 2, 3]
dequantized: [-1.4, -0.7, 0.0, 0.7]
errors:      [0.2,  0.3,  0.3, 0.2]   ← small, all weights preserved
```

---

## Why RTN fails at 1-bit vs Binary quantization

**RTN at 1-bit**, weights `[-1.2, -0.4, 0.3, 0.9]`:
```
grid: {0, 1},  scale=2.1,  zero=1
integers:    [0, 1, 1, 1]
dequantized: [-2.1, 0.0, 0.0, 0.0]   ← three weights collapse to 0.0
```
Grid is `{-S, 0}` - near-zero weights vanish entirely.

**Binary quantization at 1-bit**, same weights:
```
scale = mean(abs(w)) = 0.7
grid: {-0.7, +0.7}
dequantized: [-0.7, -0.7, +0.7, +0.7]   ← sign preserved for all weights
```
Grid is `{-S, +S}` symmetric around zero - every weight survives as its sign.

**The core difference:** RTN at 1-bit is anchored at zero so near-zero weights collapse and vanish. Binary quantization is symmetric around zero so sign information is always preserved.

---

## Straight-Through Estimator (STE) & optimize()

### The Problem
After splitting a LoRA into sub-LoRAs, you want to quantize them with minimal error. Naively quantizing B and A directly gives whatever error it gives. But there are infinitely many B*, A* pairs that give the same product BA - some of those pairs happen to quantize more cleanly than others. The optimize() function searches for those better-conditioned pairs.

### The Objective
```
min over B*, A*:  || BA - Q(B*)Q(A*) ||_F

where Q = quantize-dequantize
initialized at:  B* = B,  A* = A
```
Target is always the original BA - we want to preserve the original LoRA's behavior.

### The STE Trick
Gradient descent needs gradients. But `round()` inside quantization has zero gradient almost everywhere and is undefined at integers - standard backprop can't flow through it.

STE approximation:
- **Forward pass:** use real `round()` - actual quantization happens
- **Backward pass:** pretend `round()` was identity - gradient flows straight through unchanged

It's technically incorrect but empirically works well. It's the standard approach for quantization-aware training.

### The Algorithm
```
initialize B* = B,  A* = A

for each step:
    1. forward:  compute Q(B*)Q(A*) using real quantization
    2. loss:     || BA - Q(B*)Q(A*) ||_F
    3. backward: STE lets gradient flow through round()
    4. update:   B* = B* - lr * grad,   A* = A* - lr * grad
```

### Why it works intuitively
Neural network weights aren't unique - many equivalent reparameterizations exist. Some happen to have distributions that sit naturally near quantization grid points. The optimization navigates that space to find a better-conditioned pair without changing the overall LoRA behavior.

---

## Command-line Arguments

### Model & Adapter
- `--model_name` - which base LLM to load. choices: `Llama-2-7b-hf`, `Mistral-7B-v0.1`, `Llama-2-13b-hf`
- `--adapter_path` - path to the pre-trained LoRA adapter directory on disk

### Dataset
- `--dataset` - which dataset to evaluate on. choices: `gsm8k` (math), `minerva_math` (harder math), `xsum` (summarization)
- `--num_fewshot` - number of in-context examples shown before each test question. paper uses 0 (no examples)

### Quantization Method
- `--method` - which quantization strategy to apply to LoRA weights:
  - `fp` - no quantization, full precision baseline
  - `rtn` - round-to-nearest uniform quantization
  - `bin` - 1-bit binary quantization
  - `loraq_ratio` - paper's main method, dynamic rank split by variance coverage
  - `loraq_svd` - SVD split with fixed rank_high
  - `loraq_random` - ablation baseline, random dimension assignment
  - `loraq_norm` - ablation baseline, norm-based dimension assignment

### Quantization Hyperparameters
- `--num_bits_high` - bitwidth for the high-precision sub-LoRA (2 or 3 in the paper)
- `--num_bits_low` - bitwidth for rtn/bin methods (not used by loraq methods, those always use 1-bit for low sub-LoRA)
- `--group_size` - elements per quantization group. default 128, standard across GPTQ/AWQ/this paper
- `--rank_high` - fixed number of dimensions for high-precision sub-LoRA. only used by `loraq_svd/random/norm`
- `--ratio` - minimum variance coverage for dynamic rank selection. only used by `loraq_ratio`. e.g. 0.8 means top-h dimensions must explain 80% of variance

### Quantization Direction
- `--along_column_B` - quantize B column-wise instead of row-wise. recommended, matches SVD reparameterization
- `--along_column_A` - quantize A column-wise instead of row-wise. not recommended, paper finds row-wise better for A

### Optimization
- `--opt` - enable STE gradient optimization before quantizing. improves accuracy at the cost of more compute

---

## Replication Results

### Setup
- Model: LLaMA 2-7B
- Dataset: GSM8K (strict-match accuracy)
- Adapter: pretrained weights from paper's Mega link
- All runs use `--along_column_B --opt --num_fewshot 0 --group_size 128`

### GSM8K Results - LLaMA 2-7B

| Method | GSM8K (replication) | GSM8K (original) | Diff | Avg Bits (replication) |
|:------:|:-------------------:|:----------------:|:----:|:---------------------:|
| FP16 | 58.91 | 58.53 | +0.38 | 16 |
| RTN (1 bit) | 0 | 0 | 0 | -- |
| RTN (2 bit) | 49.36 | 49.36 | 0 | -- |
| loraq 2@0.8 | 51.02 | 51.25 | -0.23 | 1.52 |
| loraq 2@0.9 | 51.78 | 52.16 | -0.38 | 1.68 |
| loraq 3@0.8 | 54.28 | 53.60 | +0.68 | 2.04 |
| loraq 3@0.9 | 57.16 | 56.86 | +0.30 | 2.36 |

### Observations
- All differences under 1
- FP16 baseline slightly higher than paper
- 3@0.9 recovers most of FP16 performance, matching the paper's finding
