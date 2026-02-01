"""
tests.py — Verification suite for the CVaR-VQA + MTS LABS solver.

Team LABSrats — iQuHACK 2026 NVIDIA Challenge, Phase 2

Main tests (presentation slide 07):
  1. Symmetry Check — E(s)==E(-s)==E(rev)==E(-rev)==E(alt), 100 random seqs
  2. Brute-Force Verification — exhaustive 2^N for N=3..20, MTS matches
  3. Convergence Consistency — 100 MTS runs, sigma/mean < 0.05

Extended tests (--extended flag, full code coverage):
  4. Energy hand-computed values
  5. Bitstring conversion roundtrips
  6. CVaR aggregation correctness
  7. Genetic operators (combine / mutate)
  8. CUDA-Q ansatz validity          [requires cudaq]
  9. CVaR-VQA training convergence   [requires cudaq]

Run:
    python tests.py                # main tests only
    python tests.py --extended     # main + extended
    pytest tests.py -v             # all (extended auto-included)
"""

import sys
import itertools
import numpy as np

try:
    import pytest
    HAS_PYTEST = True
except ImportError:
    HAS_PYTEST = False

try:
    import cudaq
    HAS_CUDAQ = True
except ImportError:
    HAS_CUDAQ = False

# ===================================================================
# Core functions (self-contained copy from the deliverable notebook)
# ===================================================================

def labs_energy(s) -> int:
    s = np.asarray(s, dtype=int)
    N = s.size
    C = [np.dot(s[:N - k], s[k:]) for k in range(1, N)]
    return int(np.sum(np.square(C)))


def random_sequence(N, rng):
    return rng.choice([-1, 1], size=N)


def combine(p1, p2, rng):
    k = rng.integers(1, len(p1))
    return np.concatenate([p1[:k], p2[k:]])


def mutate(s, p_mut, rng):
    s = s.copy()
    for i in range(len(s)):
        if rng.random() < p_mut:
            s[i] *= -1
    return s


def tabu_search(s0, max_iters=200, tabu_tenure=7):
    s = np.asarray(s0, dtype=int).copy()
    N = len(s)
    tabu = np.zeros(N, dtype=int)
    best = s.copy()
    best_energy = labs_energy(best)
    for _ in range(max_iters):
        best_move, best_move_energy = None, None
        for i in range(N):
            candidate = s.copy()
            candidate[i] *= -1
            cand_energy = labs_energy(candidate)
            if tabu[i] > 0 and cand_energy >= best_energy:
                continue
            if best_move_energy is None or cand_energy < best_move_energy:
                best_move = i
                best_move_energy = cand_energy
        if best_move is None:
            break
        s[best_move] *= -1
        tabu = np.maximum(tabu - 1, 0)
        tabu[best_move] = tabu_tenure
        if best_move_energy < best_energy:
            best_energy = best_move_energy
            best = s.copy()
    return best, best_energy


def mts(population, p_mut=0.02, max_gens=50, tabu_iters=200,
        tabu_tenure=7, seed=None):
    rng = np.random.default_rng(seed)
    pop_size = len(population)
    population = [np.array(s) for s in population]
    energies = []
    for i in range(pop_size):
        s, e = tabu_search(population[i], max_iters=tabu_iters,
                           tabu_tenure=tabu_tenure)
        population[i] = s
        energies.append(e)
    best_idx = int(np.argmin(energies))
    best_seq = population[best_idx].copy()
    best_energy = energies[best_idx]
    for _ in range(max_gens):
        i1, i2 = rng.integers(0, pop_size, size=2)
        child = combine(population[i1], population[i2], rng)
        child = mutate(child, p_mut, rng)
        child, child_energy = tabu_search(child, max_iters=tabu_iters,
                                          tabu_tenure=tabu_tenure)
        if child_energy < best_energy:
            idx = int(rng.integers(0, pop_size))
            population[idx] = child
            energies[idx] = child_energy
            best_seq = child.copy()
            best_energy = child_energy
    return best_seq, best_energy


def brute_force_labs(N):
    """Exhaustive search over all 2^N sequences. Ground truth by definition."""
    best = float("inf")
    for bits in itertools.product([-1, 1], repeat=N):
        e = labs_energy(np.array(bits))
        if e < best:
            best = e
    return best


# ===================================================================
# Config
# ===================================================================

# Brute-force up to this N (2^20 ~ 1M seqs, total runtime ~1 min)
MAX_N = 20


# ===================================================================
# Test 1: Symmetry Check — Energy Invariance
# ===================================================================

def test_symmetry(n_trials=100):
    """
    Verify E(s) == E(-s) == E(rev(s)) == E(-rev(s)) == E(alt(s))
    for random sequences of length N in [3, 25].
    """
    rng = np.random.default_rng(42)
    for _ in range(n_trials):
        N = rng.integers(3, 26)
        s = random_sequence(N, rng)
        E0 = labs_energy(s)
        assert E0 == labs_energy(-s), f"negation failed N={N}"
        assert E0 == labs_energy(s[::-1]), f"reversal failed N={N}"
        assert E0 == labs_energy(-s[::-1]), f"neg+rev failed N={N}"
        mod = np.array([(-1) ** i for i in range(N)], dtype=int)
        assert E0 == labs_energy(s * mod), f"alt-mod failed N={N}"
    return n_trials


# ===================================================================
# Test 2: Brute-Force Verification — Small N Validation
# ===================================================================

def test_brute_force(max_n=MAX_N):
    """
    For each N = 3..max_n:
      (a) Compute true optimal E* via exhaustive 2^N search.
      (b) Run MTS and assert it reaches E*.
    """
    results = {}
    for N in range(3, max_n + 1):
        bf_opt = brute_force_labs(N)
        rng = np.random.default_rng(0)
        pop = [random_sequence(N, rng) for _ in range(10)]
        _, mts_best = mts(pop, p_mut=0.05, max_gens=40,
                          tabu_iters=300, tabu_tenure=7, seed=0)
        ok = mts_best == bf_opt
        results[N] = (bf_opt, mts_best, ok)
        assert ok, f"N={N}: MTS found {mts_best}, brute-force optimum is {bf_opt}"
    return results


# ===================================================================
# Test 3: Convergence Consistency — Multi-Run Stability
# ===================================================================

def test_convergence(N=11, n_runs=100):
    """
    Run MTS n_runs times with independent random seeds.
    Assert sigma/mean < 0.05.
    """
    best_energies = []
    for i in range(n_runs):
        rng = np.random.default_rng(i)
        pop = [random_sequence(N, rng) for _ in range(8)]
        _, best_e = mts(pop, p_mut=0.03, max_gens=30,
                        tabu_iters=200, tabu_tenure=7, seed=i)
        best_energies.append(best_e)

    arr = np.array(best_energies, dtype=float)
    mean_e = np.mean(arr)
    std_e = np.std(arr)
    rel_sigma = std_e / mean_e if mean_e > 0 else 0.0

    assert rel_sigma < 0.05, (
        f"sigma/mean = {rel_sigma:.4f} >= 0.05"
    )
    return {"N": N, "runs": n_runs, "mean": mean_e, "std": std_e,
            "rel_sigma": rel_sigma, "min": int(arr.min()),
            "max": int(arr.max())}


# ===================================================================
# Extended helpers
# ===================================================================

def bits_to_pm1(bitstr):
    return np.array([1 if b == "1" else -1 for b in bitstr], dtype=int)

def pm1_to_bits(s):
    return "".join("1" if x == 1 else "0" for x in s)

def cvar_from_samples(energies, alpha=0.2):
    if alpha <= 0 or alpha > 1:
        raise ValueError("alpha must be in (0, 1]")
    sorted_e = np.sort(energies)
    k = max(1, int(np.ceil(alpha * len(energies))))
    return float(np.mean(sorted_e[:k]))

def compute_energies_from_counts(counts):
    energies, bitstrings = [], []
    for bitstr, count in counts.items():
        e = labs_energy(bits_to_pm1(bitstr))
        for _ in range(count):
            energies.append(e)
            bitstrings.append(bitstr)
    return np.array(energies), bitstrings


# ===================================================================
# Extended Test 4: Energy hand-computed values
# ===================================================================

def test_energy_hand_computed():
    """
    Verify labs_energy against hand-calculated values.
    Catches: wrong autocorrelation formula, off-by-one indexing,
    incorrect squaring, sign errors.
    """
    assert labs_energy([1]) == 0            # N=1: no lags
    assert labs_energy([-1]) == 0           # negation of N=1
    assert labs_energy([1, -1]) == 1        # C1=-1, E=1
    assert labs_energy([1, 1]) == 1         # C1=+1, E=1
    assert labs_energy([1, -1, 1]) == 5     # C1=-2, C2=1, E=4+1=5
    assert labs_energy([1, 1, 1, 1]) == 14  # C1=3,C2=2,C3=1, E=9+4+1
    rng = np.random.default_rng(0)
    for _ in range(50):                     # E is sum of squares => >= 0
        s = random_sequence(rng.integers(2, 15), rng)
        assert labs_energy(s) >= 0


# ===================================================================
# Extended Test 5: Bitstring conversions
# ===================================================================

def test_conversions():
    """
    Verify bits_to_pm1 / pm1_to_bits roundtrips and convention (0->-1, 1->+1).
    Catches: flipped convention (would silently invert all sequences),
    broken encoding that corrupts quantum sample -> MTS handoff.
    """
    for bitstr in ["00000", "11111", "01010", "10110"]:
        assert pm1_to_bits(bits_to_pm1(bitstr)) == bitstr
    for s in [[1, -1, 1], [-1, -1, 1, 1]]:
        arr = np.array(s)
        assert np.array_equal(bits_to_pm1(pm1_to_bits(arr)), arr)
    assert list(bits_to_pm1("01")) == [-1, 1]
    assert pm1_to_bits(np.array([-1, 1])) == "01"


# ===================================================================
# Extended Test 6: CVaR aggregation
# ===================================================================

def test_cvar():
    """
    Verify CVaR aggregation: alpha=1 gives mean, alpha->0 gives min,
    monotonicity in alpha, CVaR <= mean, invalid alpha raises.
    Catches: wrong quantile selection (would bias the VQA objective
    toward high-energy samples instead of low), silent numerical errors.
    """
    e = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    assert abs(cvar_from_samples(e, 1.0) - 30.0) < 1e-9       # alpha=1 = mean
    assert abs(cvar_from_samples(np.array([5., 100., 200., 300., 400.]), 0.2) - 5.0) < 1e-9
    # monotonicity: larger alpha includes more high-energy samples
    rng = np.random.default_rng(42)
    vals = rng.uniform(0, 100, size=100)
    prev = -np.inf
    for a in [0.1, 0.25, 0.5, 0.75, 1.0]:
        c = cvar_from_samples(vals, a)
        assert c >= prev - 1e-10
        prev = c
    assert cvar_from_samples(vals, 0.2) <= np.mean(vals) + 1e-10  # CVaR <= mean
    raised = False
    try:
        cvar_from_samples(e, 0.0)
    except ValueError:
        raised = True
    assert raised  # alpha=0 must raise


# ===================================================================
# Extended Test 7: Genetic operators
# ===================================================================

def test_genetic_operators():
    """
    Verify combine/mutate produce valid {-1,+1}^N sequences of correct length.
    Catches: crossover creating wrong-length children (breaks MTS),
    mutate corrupting values outside {-1,+1}, mutate modifying input in-place.
    """
    rng = np.random.default_rng(0)
    for N in [5, 10, 20]:
        p1 = random_sequence(N, rng)
        p2 = random_sequence(N, rng)
        child = combine(p1, p2, rng)
        assert len(child) == N
        assert set(child).issubset({-1, 1})
    s = random_sequence(10, rng)
    m = mutate(s, 0.0, rng)           # p=0 must be identity
    assert np.array_equal(s, m)
    original = s.copy()
    _ = mutate(s, 0.5, rng)           # must not modify original
    assert np.array_equal(s, original)


# ===================================================================
# Extended Test 8: CUDA-Q ansatz validity
# ===================================================================

def test_cudaq_ansatz():
    """
    Verify CUDA-Q hardware-efficient ansatz produces valid measurements:
    bitstring length == N, total counts == shots, chars in {'0','1'}.
    Catches: wrong qubit count, kernel compilation errors, gate mismatches.
    """
    if not HAS_CUDAQ:
        return "skipped"

    @cudaq.kernel
    def hea(n_qubits: int, n_layers: int, params: list[float]):
        qubits = cudaq.qvector(n_qubits)
        for i in range(n_qubits):
            h(qubits[i])
        idx = 0
        for layer in range(n_layers):
            for i in range(n_qubits):
                ry(params[idx], qubits[i]); idx += 1
            for i in range(n_qubits):
                rz(params[idx], qubits[i]); idx += 1
            for i in range(n_qubits - 1):
                x.ctrl(qubits[i], qubits[i + 1])

    N, L, shots = 5, 2, 500
    rng = np.random.default_rng(0)
    params = list(rng.uniform(0, 2 * np.pi, size=L * N * 2))
    counts = cudaq.sample(hea, N, L, params, shots_count=shots)
    total = sum(counts.values())
    assert total == shots
    for bitstr in counts:
        assert len(bitstr) == N
        assert all(c in "01" for c in bitstr)
    return "passed"


# ===================================================================
# Extended Test 9: CVaR-VQA convergence
# ===================================================================

def test_cvar_vqa_convergence():
    """
    Run a short CVaR-VQA optimization and check the objective does not diverge
    (mean of last 5 evals <= 1.5x mean of first 5 evals).
    Catches: broken gradient flow, parameter update errors, optimizer misconfiguration.
    """
    if not HAS_CUDAQ:
        return "skipped"
    from scipy.optimize import minimize

    @cudaq.kernel
    def hea(n_qubits: int, n_layers: int, params: list[float]):
        qubits = cudaq.qvector(n_qubits)
        for i in range(n_qubits):
            h(qubits[i])
        idx = 0
        for layer in range(n_layers):
            for i in range(n_qubits):
                ry(params[idx], qubits[i]); idx += 1
            for i in range(n_qubits):
                rz(params[idx], qubits[i]); idx += 1
            for i in range(n_qubits - 1):
                x.ctrl(qubits[i], qubits[i + 1])

    N, L = 5, 2
    n_params = L * N * 2
    rng = np.random.default_rng(42)
    history = []

    def obj(params):
        counts = cudaq.sample(hea, N, L, list(params), shots_count=256)
        energies, _ = compute_energies_from_counts(counts)
        val = cvar_from_samples(energies, 0.2)
        history.append(val)
        return val

    minimize(obj, rng.uniform(0, 2 * np.pi, size=n_params),
             method="COBYLA", options={"maxiter": 30})

    if len(history) >= 10:
        early = np.mean(history[:5])
        late = np.mean(history[-5:])
        assert late <= early * 1.5, (
            f"CVaR diverged: early={early:.1f}, late={late:.1f}"
        )
    return "passed"


# ===================================================================
# pytest wrappers
# ===================================================================

if HAS_PYTEST:
    needs_cudaq = pytest.mark.skipif(not HAS_CUDAQ, reason="cudaq not installed")

    # Main
    class TestSymmetry:
        def test_energy_invariance(self):
            test_symmetry(100)

    class TestBruteForce:
        def test_mts_matches_brute_force(self):
            test_brute_force(MAX_N)

    class TestConvergence:
        def test_multi_run_stability(self):
            test_convergence(N=11, n_runs=100)

    # Extended
    class TestExtendedEnergy:
        def test_hand_computed(self):
            test_energy_hand_computed()

    class TestExtendedConversions:
        def test_roundtrips(self):
            test_conversions()

    class TestExtendedCVaR:
        def test_aggregation(self):
            test_cvar()

    class TestExtendedGenetic:
        def test_operators(self):
            test_genetic_operators()

    @needs_cudaq
    class TestExtendedAnsatz:
        def test_cudaq_validity(self):
            test_cudaq_ansatz()

    @needs_cudaq
    class TestExtendedVQA:
        def test_convergence(self):
            test_cvar_vqa_convergence()


# ===================================================================
# Standalone runner
# ===================================================================

def _run(name, fn):
    try:
        result = fn()
        if result == "skipped":
            print(f"    SKIP  {name} (cudaq not installed)")
            return True
        print(f"    PASS  {name}")
        return True
    except (AssertionError, Exception) as e:
        print(f"    FAIL  {name}: {e}")
        return False


if __name__ == "__main__":
    extended = "--extended" in sys.argv

    print("=" * 62)
    print(f"  LABS Verification Suite{' (extended)' if extended else ''}")
    print("=" * 62)

    all_pass = True

    # --- 1. Symmetry ---
    print("\n[1] Symmetry Check")
    try:
        n = test_symmetry(100)
        print(f"    PASSED  {n} sequences, all 4 symmetries hold")
    except (AssertionError, Exception) as e:
        print(f"    FAILED  {e}")
        all_pass = False

    # --- 2. Brute-force ---
    print(f"\n[2] Brute-Force Verification (N=3..{MAX_N})")
    try:
        results = test_brute_force(MAX_N)
        for N in sorted(results):
            bf, mts_e, ok = results[N]
            tag = "PASS" if ok else "FAIL"
            print(f"    {tag}  N={N:2d}  E*={bf:3d}  MTS={mts_e:3d}")
        print(f"    PASSED  All N=3..{MAX_N} verified")
    except (AssertionError, Exception) as e:
        print(f"    FAILED  {e}")
        all_pass = False

    # --- 3. Convergence ---
    print("\n[3] Convergence Consistency (100 runs)")
    try:
        s = test_convergence(N=11, n_runs=100)
        print(f"    N={s['N']}, {s['runs']} runs")
        print(f"    mean={s['mean']:.2f}  std={s['std']:.2f}  "
              f"sigma/mean={s['rel_sigma']:.4f}")
        print(f"    range=[{s['min']}, {s['max']}]")
        print(f"    PASSED  sigma < 0.05")
    except (AssertionError, Exception) as e:
        print(f"    FAILED  {e}")
        all_pass = False

    # --- Extended ---
    if extended:
        print("\n--- Extended Tests ---")

        print("\n[4] Energy hand-computed values")
        all_pass &= _run("hand-computed energies", test_energy_hand_computed)

        print("\n[5] Bitstring conversions")
        all_pass &= _run("roundtrip conversions", test_conversions)

        print("\n[6] CVaR aggregation")
        all_pass &= _run("CVaR correctness", test_cvar)

        print("\n[7] Genetic operators")
        all_pass &= _run("combine / mutate", test_genetic_operators)

        print("\n[8] CUDA-Q ansatz validity")
        all_pass &= _run("ansatz outputs", test_cudaq_ansatz)

        print("\n[9] CVaR-VQA convergence")
        all_pass &= _run("VQA training", test_cvar_vqa_convergence)

    # --- Summary ---
    print(f"\n{'=' * 62}")
    print(f"  {'ALL CHECKS PASSED' if all_pass else 'SOME CHECKS FAILED'}")
    print(f"{'=' * 62}")
    if not all_pass:
        exit(1)
