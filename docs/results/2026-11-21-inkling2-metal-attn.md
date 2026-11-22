# Inkling-2-Mini attention on Metal (lane ink-m-attn, 2026-11-21)

## Briefs received

1. 2026-11-21 (coordinator, via SendMessage): **Approved `InkAttnDecode`**, appended as discriminant 75 (reserved for this
   lane; no other lane appends ops now). Conditions: (1) semantics exactly RelSinkAttn(InkKvPrologue(...)) with the
   intermediates rounded to A; the CPU reference is that composition of the existing two refs, plus a test that the CPU op
   equals the pair bitwise; (2) rows 1..16 only, the op refuses larger T in its shape check so the graph can fall back to
   the pair; (3) admission cases at the real shapes plus rows-exact; (4) measure the unfused pair first and put both
   numbers, per visit, in this doc. Build the fused kernel regardless; the coordinator picks at integration. Graph wiring
   (emit 75 on decode when the backend admits it, else the pair) belongs to the integration lane: leave the model code
   alone and write the selection rule into this doc.

(Results follow below as the lane progresses.)
