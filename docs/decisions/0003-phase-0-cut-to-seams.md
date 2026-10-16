# ADR-0003: Phase 0 cut to seams

_Drafted by fleet model `guidance` from the raw note; checked against the note by Claude 2026-10-12_

Status: Accepted

Context: IntarsiaML Phase 0 scope definition following Fable 5.1 review on 2026-10-12.

Decision: Phase 0 is cut to seams. In: Rust workspace; kernel registry with cpu-ref oracle and admission tests; C kernel ABI; cuda backend via the shim (native BF16 kernels + cuBLASLt as Fast); recipe loader + Qwen3 recipe; sequence state with recurrent-state slots (single sequence, contiguous KV); measured-memory allocation; OpenAI chat/completions streaming honouring ignore_eos, min_tokens, usage.cached_tokens; Anthropic /v1/messages; /metrics, /events, /health; intarsia top TUI; golden-logit tool and intarsia accuracy; speed sweep + localeval target. Out (moved to Phase 1): web UI and live architecture map, Responses API, paged KV, containers.

Alternatives considered: Fable recommended moving Anthropic /v1/messages to Phase 1.

Consequences: Anthropic /v1/messages kept in Phase 0 because it is user requirement 6 and is one parallel agent that blocks nothing.

Evidence: Gates: Mac cpu-ref Qwen3-0.6B-architecture model matches HF golden; spark1 Qwen3-8B BF16 on CUDA top-1 >= 99% vs golden, both APIs stream, localeval speed runs. vLLM comparison is a smoke test only. Phase 1 target model: Qwen3.6-35B-A3B NVFP4 (already on spark1 disk), gate >= vLLM 86-90 tok/s single stream.

<details><summary>Raw note</summary>

Decision (accepted by Claude 2026-10-12 after Fable 5.1 review): Phase 0 is cut to seams. In: Rust workspace; kernel registry with cpu-ref oracle and admission tests; C kernel ABI; cuda backend via the shim (native BF16 kernels + cuBLASLt as Fast); recipe loader + Qwen3 recipe; sequence state with recurrent-state slots (single sequence, contiguous KV); measured-memory allocation; OpenAI chat/completions streaming honouring ignore_eos, min_tokens, usage.cached_tokens; Anthropic /v1/messages; /metrics, /events, /health; intarsia top TUI; golden-logit tool and intarsia accuracy; speed sweep + localeval target.
Out (moved to Phase 1): web UI and live architecture map, Responses API, paged KV, containers.
Gates: Mac cpu-ref Qwen3-0.6B-architecture model matches HF golden; spark1 Qwen3-8B BF16 on CUDA top-1 >= 99% vs golden, both APIs stream, localeval speed runs. vLLM comparison is a smoke test only.
Deviation from review: Fable recommended moving Anthropic /v1/messages to Phase 1; kept in Phase 0 because it is user requirement 6 and is one parallel agent that blocks nothing.
Phase 1 target model: Qwen3.6-35B-A3B NVFP4 (already on spark1 disk), gate >= vLLM 86-90 tok/s single stream.

</details>
