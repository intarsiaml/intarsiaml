# Architecture diagram

![IntarsiaML architecture: clients, server door, scheduler, engine step, model recipe, kernel registry, backends, device](assets/architecture.svg)

Reading it top to bottom: clients talk to the axum door (OpenAI, Anthropic, `/health`, `/metrics`,
`/events`, the plan endpoint, and the built-in UI). Requests wait in a bounded queue, the engine runs
chunked prefill and one-token decode steps, and the model recipe walks the layer stack (linear attention,
full attention, MoE). Each block calls kernels through the registry, where a kernel is admitted only if it
agrees with the cpu-ref oracle. The CUDA backend sits behind a torch-free C ABI. See `ARCHITECTURE.md` for
the design and `observability.md` for health, the supervisor and the metrics.

**Legend.** Solid boxes exist in the code on `origin/main`. Dashed boxes in the PLANNED strip are in
[ARCHITECTURE.md](ARCHITECTURE.md) but not built: paged KV with a radix prefix cache (KV is contiguous per slot today),
the autotuner and tuning DB (kernel choice is static priority), a Vulkan / SYCL backend, and MTP / DFlash speculation,
GGUF / AWQ quants and kernel packs. Backends: cpu-ref, CUDA, HIP and Metal are implemented. The picture is still
simplified: the registry boxes are op families, only the CUDA path of the device memory bar is drawn, and continuous
batching (not drawn) exists on Metal only. Per-claim status is in the tables inside `ARCHITECTURE.md`.

The same map, live, is the **Architecture** tab of the built-in UI (served at `/`, turned off with
`intarsia serve --no-ui`). It is drawn from `GET /v1/intarsia/plan` and overlaid with the `/events`
telemetry: edge animation follows token and step rates, kernel nodes are heat-coloured by their share of
measured op time, and the memory bar shows weights, KV fill and free memory.

| Chat | Dashboard | Architecture |
| --- | --- | --- |
| ![chat](assets/ui-chat-dark.png) | ![dashboard](assets/ui-dashboard-dark.png) | ![architecture](assets/ui-arch-dark.png) |

Light theme: `assets/ui-chat-light.png`, `assets/ui-dashboard-light.png`, `assets/ui-arch-light.png`.
