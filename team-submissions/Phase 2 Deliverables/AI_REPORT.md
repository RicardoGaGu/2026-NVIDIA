# AI Report — Team LABSrats

## 1. The Workflow

Our team used AI agents throughout both phases with a deliberate division of labor:

- **Claude Code (Claude Opus 4.5)** — Primary coding and research assistant. Used by the QA PIC (Adrian) for implementing the Phase 1 tutorial notebook end-to-end: MTS solver, CUDA-Q kernel decompositions, Trotterized circuit, quantum-enhanced MTS, and self-validation tests. Also used for PRD drafting, literature review on CVaR-VQA and PCE approaches, and structuring the verification plan.

We adopted an **implement-then-verify** loop: AI would generate code, then we would immediately write independent verification (brute-force checks, symmetry tests, hand-computed values) before moving on. This prevented error accumulation across exercises.

For context management, we maintained a `skills.md` file that we fed to the AI agent at the start of each session, containing CUDA-Q syntax, known kernel compilation pitfalls, gate decompositions, and algorithm references. This significantly reduced repeated errors and hallucinations in later exercises.

## 2. Verification Strategy

We wrote specific unit tests to catch AI hallucinations and logic errors. These were designed so that **each test is independently verifiable** — either by hand calculation, exhaustive search, or mathematical symmetry — making it impossible for an AI-generated bug to pass undetected.

### Phase 1 Tests (implemented in the tutorial notebook)

| Test | Method | What it catches |
|------|--------|-----------------|
| Energy function correctness | Hand-computed: `E([1,-1,1])=5`, `E([1,1,1,1])=14`, `E([1,-1])=1` | Incorrect autocorrelation formula or off-by-one indexing |
| Brute-force optimality | Exhaustive search for N=3..7, compared against known optima (E*: 1, 2, 2, 7, 3) | Wrong energy function, wrong sign conventions |
| Symmetry invariance | For random sequences, assert `E(s)==E(-s)==E(rev)==E(-rev)` over 10+ trials | Broken symmetry in energy computation |
| MTS convergence | MTS must find brute-force optimum for N=5,6,7 | Bugs in tabu search, combine, or mutate |
| Quantum population quality | Mean energy of quantum-seeded population < random population | Incorrect bitstring-to-sequence conversion, broken circuit |

<!-- TODO: Add Phase 2 tests once custom quantum algorithm implementation is complete -->
### Phase 2 Tests (in `tests.py`)
TO-DO
### How we caught a real hallucination

During Phase 1 self-validation, the AI provided an incorrect dictionary of known optimal LABS energies: `{N=5: 3, N=6: 2, N=7: 5}`. Our brute-force exhaustive search — which is ground truth by definition — revealed the correct values are `{N=5: 2, N=6: 7, N=7: 3}`. Every single value was wrong. This confirmed our strategy: never trust reference values from AI without independent verification.

## 3. The "Vibe" Log

### Win: AI saved us hours on CUDA-Q gate decompositions

**Exercise 3** required translating the 2-qubit and 4-qubit rotation block circuit diagrams (Figures 3 and 4 from the paper) into CUDA-Q kernel code. This involves:
- Decomposing `R_YZ`, `R_ZY` into RZZ + Rx gates
- Decomposing `R_YZZZ`, `R_ZYZZ`, `R_ZZYZ`, `R_ZZZY` into CNOT ladders + Rz + Rx wrapping
- Getting the sign conventions and gate ordering correct

Claude Code produced working decompositions on the first attempt by reading the circuit diagrams from the paper and applying standard basis-change identities. Without AI, this would have required manually working through each decomposition and debugging sign errors — a process that can easily consume hours. The AI delivered correct, tested code in a single pass.

### Learn: Creating `skills.md` eliminated repeated compilation errors

In early sessions, the AI kept generating CUDA-Q code with patterns that fail compilation:
- `list[list[int]]` kernel parameters (causes `CompilerError`)
- Sub-kernel calls passing `cudaq.qubit` arguments (causes `CompilerError`)
- Using Python features not supported in the CUDA-Q kernel compiler

After hitting these errors multiple times, we created `skills.md` — a structured context file containing:
- CUDA-Q v0.13.0 syntax reference with working examples
- A list of known kernel limitations and their workarounds (flatten lists, inline gates)
- Gate decomposition patterns that compile successfully
- Algorithm parameters and references

Feeding `skills.md` at the start of each session eliminated these repeated failures entirely. The AI would immediately use the flatten-and-stride pattern for nested indices and inline all gate operations, producing compilable code on the first try.

### Fail: Wrong known optima for LABS

As described in the Verification Strategy, Claude confidently provided a dictionary of "known optimal" LABS energies that was completely wrong. The values `{N=5: 3, N=6: 2, N=7: 5}` were likely hallucinated or sourced from a confused memory of the problem.

**How we fixed it:** We had already planned brute-force verification as part of our test suite. The exhaustive search for N=3..7 takes under a second and produces ground truth. Once we saw the mismatch, we replaced the AI's values with the brute-force results: `{N=3: 1, N=4: 2, N=5: 2, N=6: 7, N=7: 3}`.

**Lesson:** For any optimization problem, always verify claimed optima with exhaustive search on small instances. AI models can hallucinate numerical values with high confidence.

<!-- TODO: Add Phase 2 fails if any occur during GPU migration or custom algorithm implementation -->

### Context Dump

#### `skills.md` (full file included in repository)

Our `skills.md` file contains:
- LABS problem definition and symmetries
- CUDA-Q kernel syntax reference with code examples
- Known compilation pitfalls and workarounds
- Gate decomposition patterns (RZZ, R_YZ, R_ZY, R_YZZZ, etc.)
- MTS algorithm components and parameters
- Counterdiabatic quantum algorithm theory and circuit structure
- QE-MTS pipeline description
- Key paper references
- Environment details (qBraid CPU, Brev GPU)

This file served as persistent memory across AI sessions, ensuring the agent never lost context about project-specific constraints.

<!-- TODO: Add any additional prompts, MCP configs, or other context management tools used in Phase 2 -->
