# LABS Challenge Skills - CUDA-Q & Quantum-Enhanced Optimization

## Project Context
This is the iQuHACK 2026 NVIDIA Challenge: solving the Low Autocorrelation Binary Sequences (LABS) problem using a hybrid quantum-classical workflow, then GPU-accelerating it.

## The LABS Problem
- Given binary sequence $(s_1 \cdots s_N) \in \{+1,-1\}^N$, minimize:
  - $E(s) = \sum_{k=1}^{N-1} C_k^2$ where $C_k = \sum_{i=1}^{N-k} s_i s_{i+k}$
- Key symmetries: E(s) == E(-s) == E(reverse(s)) == E(-reverse(s))
- Known optimal energies: N=3:1, N=4:2, N=5:2, N=6:7, N=7:3

## CUDA-Q Syntax Reference (v0.13.0)
```python
import cudaq

@cudaq.kernel
def my_kernel(N: int, params: list[float]):
    reg = cudaq.qvector(N)
    h(reg)                        # Hadamard on all qubits
    rx(angle, reg[i])             # Rx rotation
    ry(angle, reg[i])             # Ry rotation
    rz(angle, reg[i])             # Rz rotation
    x(reg[i])                     # Pauli X
    x.ctrl(reg[i], reg[j])       # CNOT (control=i, target=j)

# Sampling
result = cudaq.sample(my_kernel, N, params, shots_count=1000)
for bitstring in result:
    count = result.count(bitstring)
```

### CUDA-Q Kernel Limitations
- **No nested list indexing**: `list[list[int]]` parameters cause CompilerError. Flatten to `list[int]` and use stride-based indexing.
- **No sub-kernel calls with qubit arguments**: Calling a kernel that takes `cudaq.qubit` from another kernel causes CompilerError. Inline all gate operations in one kernel.
- **Supported parameter types**: `int`, `float`, `list[int]`, `list[float]`
- **Constants**: Use `from math import pi` for pi inside kernels.

### Working Pattern for Multi-Qubit Rotations
```python
# Flatten nested index lists before passing to kernel
G2_flat = [idx for pair in G2 for idx in pair]      # stride 2
G4_flat = [idx for quartet in G4 for idx in quartet]  # stride 4

# Inside kernel, access with stride
for g in range(num_G2):
    idx = g * 2
    a = G2_flat[idx]
    b = G2_flat[idx + 1]
    # operate on reg[a], reg[b]
```

## Gate Decompositions

### RZZ gate: $R_{ZZ}(\theta) = e^{-i\theta ZZ/2}$
```
x.ctrl(q0, q1)    # CNOT
rz(theta, q1)
x.ctrl(q0, q1)    # CNOT
```

### R_YZ(theta): Y on qubit a, Z on qubit b
```
rx(-pi/2, reg[a])
x.ctrl(reg[a], reg[b])
rz(theta, reg[b])
x.ctrl(reg[a], reg[b])
rx(pi/2, reg[a])
```

### R_ZY(theta): Z on qubit a, Y on qubit b
```
rx(-pi/2, reg[b])
x.ctrl(reg[a], reg[b])
rz(theta, reg[b])
x.ctrl(reg[a], reg[b])
rx(pi/2, reg[b])
```

### R_ZZZZ(theta) via CNOT ladder
```
x.ctrl(reg[a], reg[b])
x.ctrl(reg[b], reg[c])
x.ctrl(reg[c], reg[d])
rz(theta, reg[d])
x.ctrl(reg[c], reg[d])
x.ctrl(reg[b], reg[c])
x.ctrl(reg[a], reg[b])
```

### R_YZZZ / R_ZYZZ / R_ZZYZ / R_ZZZY
Wrap the CNOT ladder with `rx(-pi/2)` and `rx(pi/2)` on whichever qubit has the Y:
```
rx(-pi/2, reg[y_qubit])
# ... CNOT ladder + rz ...
rx(pi/2, reg[y_qubit])
```

## Memetic Tabu Search (MTS) Algorithm

### Components
1. **Energy function**: `labs_energy(s)` — compute $\sum C_k^2$
2. **Combine**: single-point crossover — random cut point k, return `p1[:k] + p2[k:]`
3. **Mutate**: for each bit, flip with probability `p_mut`
4. **Tabu search**:
   - Iteration budget M: random in [N/2, 3N/2]
   - Tabu tenure: random in [M/50, M/10]
   - Best non-tabu move each iteration
   - Aspiration: override tabu if move yields new global best
5. **MTS loop**: init population → (make child → mutate → tabu search → update) repeat

### Typical Parameters
- Population size: 20
- Mutation probability: 0.05
- MTS iterations: 50-100
- Tabu tenure: M/50 to M/10

## Counterdiabatic Quantum Algorithm

### Theory
- Adiabatic approach: evolve from $H_i = \sum \sigma_x$ to $H_f$ (LABS Hamiltonian)
- Counterdiabatic term $H_{CD}$ corrects diabatic transitions
- Trotterized circuit applies $e^{-i\theta H_{CD}}$ in discrete steps

### Interaction Terms
- **G2 (2-body)**: indices [i, i+k] from $\prod_{i=1}^{N-2}\prod_{k=1}^{\lfloor(N-i)/2\rfloor}$
- **G4 (4-body)**: indices [i, i+t, i+k, i+k+t] from $\prod_{i=1}^{N-3}\prod_{t=1}^{\lfloor(N-i-1)/2\rfloor}\prod_{k=t+1}^{N-i-t}$
- Convert 1-based equation indices to 0-based Python indices

### Circuit Structure (Equation 15)
For each Trotter step:
1. Apply 2-body blocks with angle `4 * theta`
2. Apply 4-body blocks with angle `8 * theta`
- `theta` is precomputed via `utils.compute_theta(t, dt, T, N, G2, G4)`
- All $h_i^x = 1$ for standard LABS

## Quantum-Enhanced MTS (QE-MTS)
1. Run counterdiabatic circuit → sample bitstrings
2. Convert bitstrings ('0'→+1, '1'→-1) to form initial population
3. Run MTS with this quantum-seeded population
4. Compare against MTS with random population

## Key References
- [Scaling advantage with quantum-enhanced memetic tabu search for LABS](https://arxiv.org/html/2511.04553v1)
- [Parallel MTS by JPMorgan Chase](https://arxiv.org/pdf/2504.00987)

## Environment
- Phase 1: qBraid (CPU) with CUDA-Q v0.13.0, Python 3.11
- Phase 2: Brev (NVIDIA GPUs — L4, T4, A100)
- cudaq not available on native Windows; use qBraid, WSL2, or Docker
