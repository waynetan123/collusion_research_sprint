# v2 Step 2: Build the patch pool (decisions log)

Status: **Done by reuse, 2026-10-16.** The v2 design doc says the pool pipeline is unchanged from v1, so v2 uses the pool built in v1 Step 1 (`docs/step1_decisions.md`) without rebuilding it. This step re-verified that pool and checked it against the v2 requirements. Result: **333/333 records pass the independent acceptance check.**

## 1. Decision: reuse, do not rebuild

- The v2 doc (§Dataset and §Steps 2) keeps the v1 pool, normalization and verification unchanged.
- Rebuilding would not reproduce the same pool on another machine: specs were drafted by Claude, and the replies are cached only on the Mac (`data/interim/llm_cache/`). Reuse keeps one fixed, already-reviewed pool.
- **Pool size: 333 pairs (100 HumanEval + 233 MBPP), not the doc's 400–500.** This was accepted by the user in v1 (`step1_decisions.md` §9a) and is still enough for v2. A 25-episode trajectory uses 50 functions, so the v1 round-cap problem (§14a there) no longer applies.

## 2. Files used (unchanged since 2026-10-09)

| File | sha256 |
|---|---|
| `data/pool/pool.jsonl` (333 records) | `f4827fcefa0484942ccfbece858ef4800e3b6642b9508488a64f38cc610245c8` |
| `data/pool/pool_tests.jsonl.gz` (hidden EvalPlus inputs and expected outputs) | `c84adb9f939b9410ed749cdeb6c805bd6b080b23debbff0e684adc05437dc667` |

`data/` is gitignored. Another machine must receive a copy of `pool.jsonl` and check this hash. Never rebuild the pool there (§1).

## 3. Re-verification (2026-10-16)

- **Stored verification:** every record was verified process-per-call. Clean passes every input in all 333; buggy fails at least one in all 333.
- **Independent acceptance check** (`python -m collusion.pool.check_pool`, reads only the files on disk): **333/333 pass, 0 failed** (100 HumanEval, 233 MBPP), in 1 min 54 s on the Mac. It re-runs clean and buggy on every stored input, checks the docstring examples on both versions, source/annotation leaks, the single-edit rule, and that `spec` equals the embedded docstring.
- **Buggy failure rate over test inputs:** quartiles 0.27 / 0.75 / 0.94, the same as in `step1_decisions.md` §11.
- **Bug types:**
  - HumanEval: missing logic 27, value misuse 20, excess logic 17, variable misuse 15, operator misuse 15, function misuse 6.
  - MBPP: operator misuse 102, value misuse 85, missing logic 23, function misuse 23.

## 4. v2 dataset requirements, checked

| v2 requirement | How it is met |
|---|---|
| Sources: MBPP+ with AST-mutated bugs; HumanEval with HumanEvalFix bugs; EvalPlus inputs as ground truth | The v1 pool, unchanged (`step1_decisions.md` §3–§8) |
| Normalization steps 1–9 | Unchanged from v1 (§4–§10 there; the deviations recorded there still apply) |
| One record per function with id, source, entry_point, spec, clean_code, buggy_code, bug_origin, bug_type, bug_location, inputs, atol, verification | All fields present. `inputs` (with `expected` and `timeouts`) lives in the test sidecar, read via `collusion.pool.io.load_tests` |
| Each function used at most once per trajectory, never both versions | `collusion.v2.sequences`: 50 distinct pool functions per seed; upstream's loader also rejects repeats and twins |
| About 50% buggy per trajectory | Exactly 25 of 50 per seed |
| 2 functions per episode, 50 per 25-episode trajectory | Yes; 333 ≥ 50 |
| Tests stay hidden | EvalPlus inputs are never exported to `data/tasks/`; agents write their own tests. Caveat: agent tests run unsandboxed from the repo root (`v2_step1_decisions.md` §3a) |
| Branches from one checkpoint share the post-branch sequence | Sequences are fixed per seed; every branch continues the same file |
| Sources balanced within each trajectory; source kept as a covariate | Every episode pairs one HumanEval and one MBPP function (25 + 25 per seed); `source`, `bug_type` and `bug_origin` are in the task manifest for analysis |

## 5. Hand-offs

- **To Step 3 (calibration):** calibration runs the task phase on all **666 tasks** (333 clean + 333 buggy), 3 samples each. That is about 2,000 task phases; the doc's budget assumed about 450 × 3. Drop functions answered correctly 0 of 3 times; if overall accuracy is outside 85–95%, swap mutants from `mutants_all.jsonl` (re-running the spec example and coverage checks for swapped functions). Any change to the pool changes `tasks_sha256`, so it must happen **before** arms 1 and 2.
- **Known pool properties that may affect calibration** (`step1_decisions.md` §12–§13a):
  - spec examples always sit where the bug is invisible;
  - 24 specs have only trivial examples;
  - some specs narrate the implementation.
