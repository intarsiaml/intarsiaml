# Qualification report: <model> on <backend>

_Template of `quick-report.md` and `full-report.md`, as written by `intarsia qualify` (docs/adding-a-model.md).
The quick report has the same sections but only the quick stages; the full report adds the in-depth stages.
Angle brackets are filled in by the tool, the prose in italics is explanation and is not in the output.
`cargo test -p intarsia-cli` keeps the headings below equal to the ones the tool writes._

_Full (quick + in-depth) section, generated unix <ts>, intarsia <version> build `<tree stamp or unstamped>`._

## Headline

_The first thing a reader sees: one verdict, then the numbers people quote._

**Verdict: PASS | FAIL | ERROR | INCOMPLETE** (<n> stages: <p> pass, <f> fail, <e> error, <s> skipped)

| | |
|---|---|
| Model | <name> (`<model dir>`) |
| Backend / device | <hip / AMD Radeon AI PRO R9700 (gfx1201)> |
| Recipe | <builtin:qwen3 covers this model (N/N tensor templates resolve, quant bf16)> |
| Decode | 128 tok: **<tok/s>** tok/s (<n>% of floor); 1024 tok: ...; 4096 tok: ... |
| Prefill | 128 tok: **<tok/s>** tok/s; ... |
| Accuracy (short golden) | **PASS** (prefill top-1 <99.6>% KL <3e-4>; decode top-1 <>% KL <>) |
| Accuracy (long golden) | in-depth only: **PASS** (prefill, decode and prefix reuse) |
| Quick gate | **PASS**: perplexity <7.1>, <5/5> sanity prompts ok, no non-finite rows |
| Wall time | quick <9m12s>, in-depth <31m05s>, total <40m17s> |
| Reference | <url or directory, only with --reference> |

### Failures

_Every FAIL and ERROR stage with its reason (the first line; the log is in `logs/`). "None." when there is none._

- `<stage>` **FAIL**: <reason>

### Skipped

_Every skipped stage with the reason (no golden, tool missing, blocked by a failed stage, cpu has no kernel admission). A skip is not a pass: read this list before trusting a PASS._

- `<stage>`: <reason>

## Stages

_One row per stage in run order: quick stages first, then in-depth. Wall is the stage's own time._

| # | Stage | Section | Verdict | Wall | Result |
|---|---|---|---|---|---|
| 1 | `doctor` doctor: host and device fingerprint, known pitfalls | quick | PASS | 6s | <device, kernel count> |
| 2 | `recipe` recipe resolve | quick | PASS | 1s | <recipe covering the model> |
| 3 | `plan` plan show | quick | PASS | 31s | <ops resolved, exactness, load time> |
| 4 | `kernels` kernels test on the model's shapes | quick | PASS | 40s | <kernels, cases> |
| 5 | `gate` quick gate | quick | PASS | 90s | <perplexity, sanity> |
| 6 | `acc-short` short accuracy against a golden | quick | PASS | 2m | <modes, top-1, KL> |
| 7 | `bench` bench speed 128 / 1024 / 4096 | quick | PASS | 3m | <decode tok/s, % of floor> |
| 8 | `acc-long`, `envelope`, `bench-long`, `ab`, `tp` | full | ... | ... | in-depth stages |

### Notes

_Per-stage observations that are not failures: bench warnings (for example "decode ran eagerly" where the backend has no graphs yet), sizes skipped because they do not fit, warnings from `doctor`._

- `<stage>`: <note>

## Speed

_Bench tables (the median of the repetitions per cell; prefill includes queueing and HTTP, see docs/benchmarking.md). The floor is memory bandwidth divided by the bytes one decoded token reads; the line above each table says where both inputs came from. More than 100% of floor means a wrong input, not a fast engine._

**quick bench** (floor: <16.38> GB/token at <627> GB/s; <where the inputs came from>)

| Prompt tokens | TTFT s | Prefill tok/s | Decode tok/s | stdev | Floor tok/s | % of floor |
|---|---|---|---|---|---|---|
| <126> | <0.060> | <2090> | <36.89> | <0.01> | <38.3> | <96.3> |

_Also: the long-context bench (up to 16k), the tp variant's bench, and the A/B ratios against the reference server (ours / reference, one sequential localeval round)._

## Accuracy

_One table per golden comparison, mode by mode (prefill, decode, reuse). Thresholds and their meaning: docs/accuracy.md._

**acc-short** against `<golden file>` (<found by model name | given | made>; gate class <bf16>)

| Mode | Positions | Top-1 % | Top-5 % | Mean KL | Mean KL (21-way) | Max KL (21-way) | tail diff | Non-finite | Greedy match % | Floor |
|---|---|---|---|---|---|---|---|---|---|---|
| prefill | <434> | <99.54> | <99.1> | <3.1e-4> | <2.0e-4> | <4.0e-3> | <1e-3> | 0 | <80.0> | pass |

_Then the regression-gating artifacts (baseline and noise envelope, with the command that uses them) and the quick-gate line._

## Plan and recipe

_Which recipe the model resolved to, the resolved plan (kernel per op, exactness, whether it passed admission), and on a load error the list of ops without a kernel._

- recipe: <builtin:qwen3 covers this model> (PASS)
- plan: <n ops resolved> (PASS)

| Op | Kernel | Exactness | Admitted | Calls / decode step |
|---|---|---|---|---|
| <matmul> | <kernel name> | <fast> | yes | <n> |

## Provenance

_Where the numbers came from, per stage: build stamp (`scripts/tree-stamp.sh` output, or `unstamped` for a plain cargo build), host, start time, the last command run. Full command lists, environment knobs (`GPU_MAX_HW_QUEUES`, `INTARSIA_*`, `FLEETQ_JOB`) and the model fingerprint are in `state.json` and `full.json`._

| Stage | Build stamp | Host | Started (unix) | Command |
|---|---|---|---|---|
| `bench` | <head=... code-tree=... dirty=no> | <host> | <ts> | `<intarsia bench speed ...>` |

## Artifacts

_Files under the report directory, by stage: logs, `doctor.json`, `plan.json`, bench JSON, baselines and the envelope, made goldens, `state.json`._

- `bench/<label>.json` (bench)
- `state.json` (stage records; `--resume` reads it)
