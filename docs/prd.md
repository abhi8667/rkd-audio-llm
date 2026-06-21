# Engineering Kickoff Document
## Relational Knowledge Distillation for End-to-End Multimodal Audio-LLM Compression

**Prepared for:** Engineering team kickoff
**Source materials:** Project Report (Abhinandan Jaiswal, RV College of Engineering, Bangalore) and Audio-LLM Compression PRD
**Proposed Timeline:** August – September 2026 (8 weeks)
**Document purpose:** Provide enough architectural, mathematical, and engineering specification that implementation can begin immediately without further design discussion.

---

## 1. Executive Summary

### 1.1 What Problem This Project Solves

Every production speech-understanding system today is a **cascade**: a Speech-to-Text (ASR) model converts audio into text, and a separate LLM reasons over that text. This is the architecture behind nearly every voice assistant, transcription-based meeting summarizer, and call-center analytics tool in production.

Cascading throws away information that lives only in the waveform and never survives transcription:

- **Prosody and emotion** — sarcasm, urgency, distress, hesitation. A transcript of "I'm fine" cannot distinguish a calm statement from a clipped, angry one.
- **Ambient acoustic context** — sirens, machinery noise, crowd sounds, room reverberation. These are often the actual signal of interest (e.g., "is this call coming from a factory floor or a quiet office?") and are deleted at the ASR step.
- **Code-switching artifacts** — in mixed-language speech (e.g., Hindi-English "Hinglish"), the acoustic signal itself carries cues (accent shifts, phoneme blending) that a downstream text-only LLM never sees, because the ASR system has already forced a lossy decision about which language's orthography to use.
- **Latency** — running two large models sequentially (ASR, then LLM) compounds inference time, which is unacceptable for edge and real-time use cases.

This project replaces the cascade with a **single continuous pipeline**: an audio encoder produces continuous hidden-state embeddings, a lightweight projection bridge maps those embeddings directly into the input space of a compact language decoder, and the decoder reasons natively over acoustic features — no intermediate text tokenization step exists at all.

The central engineering risk this introduces is **cross-modal alignment collapse**: when you compress (downsample) continuous audio embeddings into a much smaller decoder's input manifold, naive distillation from a large teacher model causes the compact student to lose the fine-grained relational structure of the teacher's representation space, producing degenerate or collapsed embeddings. This project's core technical contribution is solving that problem using **Relational Knowledge Distillation (RKD)** — distilling the *geometry* of the teacher's representation space (pairwise distances and angles between embeddings) rather than just matching final outputs.

### 1.2 Target Users

This is a **research/engineering infrastructure project**, not a consumer-facing product, so "users" here means downstream consumers of the trained artifact and the codebase:

| User type | What they need from this project |
|---|---|
| **ML researchers / the core team** | A reproducible training pipeline, clear ablation hooks (turn RKD on/off, swap encoders), and instrumented metrics to validate the alignment-collapse hypothesis. |
| **Edge application developers** | A compact (<2B parameter), quantizable checkpoint with a documented inference API that can run locally without a network round-trip. |
| **Downstream product teams** (e.g., voice assistants, accessibility tools, call analytics) | A drop-in audio-understanding backbone that outperforms cascaded ASR+LLM pipelines on latency and on emotion/context-sensitive tasks, without requiring them to host a multi-billion-parameter model. |
| **Academic reviewers / thesis evaluators** | Mathematically rigorous, empirically validated evidence that RKD avoids alignment collapse, with reproducible benchmarks against point-wise KD baselines. |

### 1.3 Value Proposition

- **Lower latency** than cascaded pipelines by removing a full forward pass (the ASR model) and the tokenization/detokenization overhead between stages.
- **Higher fidelity on paralinguistic and ambient tasks** because acoustic information is never discarded.
- **Deployability on consumer/edge hardware** via a sub-2B decoder, LoRA fine-tuning, and NF4 quantization — this is explicitly not a cloud-only multi-billion-parameter system.
- **A trained, reusable compression technique** (RKD for cross-modal distillation) that is not specific to this one model and can be reapplied to future audio-LLM compression efforts.

### 1.4 Existing Alternatives

| Approach | Limitation this project addresses |
|---|---|
| **Cascaded ASR → LLM** (e.g., Whisper → GPT-class model) | Loses prosody/emotion/ambient signal; two sequential forward passes; error propagation from ASR mistakes into the LLM. |
| **Large end-to-end multimodal audio-LLMs** (e.g., multi-billion-parameter speech-native models) | Architecturally correct (no tokenization bottleneck) but too large for edge/consumer deployment; often closed-weight or compute-prohibitive to fine-tune. |
| **Standard (point-wise) Knowledge Distillation** for compression | Matches output logits or final-layer activations independently per token; under heavy cross-modal downsampling this causes **alignment collapse** — the student's embedding space loses the relative structure (which tokens are "close" or "far" from each other) that the teacher learned, even if individual point-wise losses look low. |

### 1.5 Why This Solution Is Unique

The novelty is not the encoder-projection-decoder architecture itself (this pattern exists in larger systems) — it is the **application of Relational Knowledge Distillation specifically to solve cross-modal alignment collapse during compression**, validated on a genuinely hard, underexplored task: **code-switched (Hinglish) speech understanding in noisy ambient conditions**. Most distillation literature targets same-modality compression (e.g., distilling a large text LLM into a small one) or single-language, clean-audio ASR. This project's combination of (a) cross-modal distillation, (b) sub-2B target size, (c) code-switched multilingual speech, and (d) ambient noise robustness is the specific gap being filled.

### 1.6 Real-World Applications

- On-device voice assistants for multilingual households/regions (e.g., India) that need to handle natural code-switching without cloud round-trips.
- Emotion- and context-aware call center QA tools that don't strip tone information at the transcription step.
- Accessibility tools that need low-latency, locally-run audio understanding (e.g., live captioning with emotional/contextual annotations).
- Noise-robust voice control for industrial/automotive edge devices where ambient sound is part of the signal, not just noise to be filtered out.

### 1.7 Expected Outcomes

1. A working, end-to-end trained student model (audio encoder + projection bridge + compact decoder) that runs on consumer-grade hardware.
2. Empirical evidence — via the alignment metrics defined in Section 14 — that RKD avoids the alignment collapse exhibited by a point-wise KD baseline.
3. Benchmarked latency and memory footprint improvements over a comparable cascaded ASR+LLM pipeline.
4. A reusable, documented RKD training framework (PyTorch forward hooks, distance/angle loss implementations) that generalizes beyond this one model pairing.

---

## 2. Requirements Analysis

### 2.1 Functional Requirements

#### Core Features (must exist for the MVP / Phase 1–4 deliverable)

**C1. Audio Encoder Integration**
- *Description:* Wrap a pretrained self-supervised audio encoder (WavLM or Whisper encoder) as a frozen feature extractor producing dense hidden-state frames.
- *User interaction:* None directly — this is an internal pipeline component invoked by the training loop and inference API.
- *Input:* Raw waveform, 16kHz mono PCM, arbitrary length (chunked to max 30s segments).
- *Output:* A tensor of shape `(batch, num_frames, encoder_hidden_dim)` — e.g., `(B, T, 1024)` for WavLM-Large.
- *Edge cases:* Silence-only input; sample rates other than 16kHz (must resample); clipped/distorted audio; audio shorter than one encoder frame (~20ms).

**C2. Projection Bridge (Continuous Modality Adapter)**
- *Description:* A trainable linear or convolutional layer mapping encoder hidden states into the decoder's input embedding space, with optional temporal downsampling (frame-rate reduction) to control sequence length fed into the decoder.
- *User interaction:* None directly — trained component, exposed via config (linear vs. conv, downsampling factor).
- *Input:* `(B, T_audio, encoder_dim)`.
- *Output:* `(B, T_proj, decoder_embed_dim)` where `T_proj <= T_audio` depending on downsampling stride.
- *Edge cases:* Mismatched dims if encoder or decoder is swapped without updating config; numerical instability if projection isn't normalized (must apply LayerNorm post-projection).

**C3. Compact Decoder Integration with LoRA**
- *Description:* Load a sub-2B parameter causal LLM, freeze base weights, attach LoRA adapters (r=16) to attention and/or MLP projection matrices, and accept continuous embeddings as a prefix/replacement for token embeddings.
- *User interaction:* None directly — internal; exposed via inference API (Section 2.2 below) for downstream consumers.
- *Input:* Projected audio embeddings `(B, T_proj, decoder_embed_dim)`, optionally concatenated with a text instruction prompt's token embeddings (for instruction-following tasks).
- *Output:* Autoregressive token logits, decoded to text (e.g., "Intent: complaint" or a free-text summary).
- *Edge cases:* Decoder context window overflow if `T_proj` + prompt length exceeds max sequence length; need a defined truncation/sliding-window policy.

**C4. RKD Training Engine**
- *Description:* Forward-hook-based extraction of intermediate activations from both teacher and student models during training, computing the combined loss `L_total = L_CE + α·L_RKD-Distance + β·L_RKD-Angle`.
- *User interaction:* Configured via training config YAML (loss weights α, β; which layers to hook).
- *Input:* A batch of (audio, target text) pairs; teacher and student model references.
- *Output:* Scalar loss values logged per step; gradients backpropagated into the student's projection bridge and LoRA adapters (teacher remains frozen, no gradient).
- *Edge cases:* Teacher and student producing different sequence lengths at the hooked layer (must align via interpolation or the nearest-frame mapping defined in Section 9); hook memory leaks if not properly detached/cleared between batches; teacher unavailable on edge-constrained training hardware (must support teacher-on-CPU-offload or precomputed teacher embeddings cached to disk).

**C5. Dataset Curation & Loading Pipeline**
- *Description:* Loaders for multi-dialect, code-switched (Hinglish), noise-augmented audio datasets, with on-the-fly noise injection (additive background noise at controlled SNR).
- *User interaction:* CLI/config-driven dataset selection and augmentation parameters.
- *Input:* Raw audio files + transcripts/labels (intent labels, summaries) + noise profile library.
- *Output:* Batched, padded tensors ready for the encoder, with attention masks.
- *Edge cases:* Class imbalance in intent labels; missing transcripts for some ambient-noise-only clips; variable-length batching causing excessive padding waste.

**C6. Evaluation Harness**
- *Description:* Runs downstream task evaluation (intent classification accuracy, audio summarization quality) and alignment-collapse diagnostics (Section 14) on held-out data, for both the RKD student and the point-wise-KD baseline student.
- *User interaction:* `python evaluate.py --checkpoint <path> --task intent_classification`.
- *Input:* Trained checkpoint path, eval dataset split.
- *Output:* Metrics report (accuracy, F1, alignment scores, latency, memory) written to a structured log (JSON) and WandB.
- *Edge cases:* Comparing models with different `T_proj` (sequence lengths) fairly; ensuring eval-time noise profiles aren't seen during training (no leakage).

#### Advanced Features (Phase 4+ / stretch goals within Weeks 7–8 if time permits)

**A1. Streaming Inference Mode** — process audio in rolling chunks rather than requiring the full clip up front, enabling lower first-token latency for real-time applications.

**A2. Quantization-Aware Training Pass** — fine-tune with simulated NF4 quantization noise injected during the last training phase, rather than only post-training quantization, to recover accuracy lost from quantization.

**A3. Multi-Teacher Distillation** — support distilling from more than one teacher (e.g., a large audio-LLM plus a separate emotion-recognition model) with weighted RKD losses per teacher.

**A4. Automatic Hook-Layer Search** — a small search procedure that tries RKD distillation at several teacher/student layer pairings and reports which pairing minimizes alignment collapse, rather than requiring a human to hand-pick layers.

#### Future Features (explicitly out of scope for the 8-week window, documented for roadmap continuity)

**F1.** On-device deployment packaging (ONNX/CoreML/TFLite export) for mobile targets.
**F2.** Multi-speaker diarization integration upstream of the encoder.
**F3.** Additional code-switched language pairs beyond Hindi-English (e.g., Spanish-English, Tagalog-English).
**F4.** Continual learning / online adaptation from user-specific audio without full retraining.

### 2.2 Non-Functional Requirements

**Scalability**
- Training pipeline must scale from single-GPU (research/dev iteration) to multi-GPU data-parallel (via Hugging Face `accelerate`) without code changes — config-driven only.
- Dataset loader must handle datasets that don't fit in memory (streaming/iterable datasets via `datasets` library), since ambient noise-augmented corpora can be large.

**Performance**
- Target inference latency: report ms/token for the student model and the cascaded baseline under identical hardware, with a target of **at least 2x speedup** over the cascade (informed target, to be validated empirically in Phase 4 — see Section 18 for risk if not met).
- Training throughput should be profiled and logged (samples/sec) to catch regressions from inefficient hook implementations (forward hooks have non-trivial overhead if not scoped correctly).

**Reliability**
- All training runs must be checkpointed every N steps (configurable) so a crashed run can resume without data loss.
- The RKD loss computation must not silently produce NaN — add assertion-based guards (Section 13) that halt training with a clear error rather than corrupting a checkpoint.

**Security**
- This is a research codebase with no end-user authentication surface. Relevant security concerns are narrower than a typical product:
  - No PII should be logged verbatim in WandB run logs (e.g., raw transcripts containing names) — redact or hash before logging if datasets contain identifiable speech.
  - Dataset licensing/usage rights must be verified before training (do not train on data without a license permitting model training).

**Accessibility**
- N/A in the WCAG sense (no UI). The *downstream applications* of this model (e.g., captioning tools) have accessibility implications, but those are outside this project's scope.

**Maintainability**
- Encoder, projection bridge, and decoder must be swappable via config without touching training loop code (dependency injection pattern, detailed in Section 5).
- RKD loss functions must be unit-testable in isolation from the full training loop (pure functions operating on tensors, not requiring a live model).

**Cost Considerations**
- Teacher model inference during training is the dominant compute cost. Mitigation: precompute and cache teacher activations once per dataset (since the teacher is frozen and doesn't change across epochs), rather than re-running the teacher forward pass every epoch. This is a **required** optimization, not optional, given the 8-week timeline and likely limited GPU budget — detailed in Section 17.
## 3. User Personas

Since this project has no consumer-facing UI, personas are framed around the people who interact with the **codebase, training pipeline, and trained artifact**.

### Persona 1 — Abhinandan, the Research Engineer (Primary Builder)
- **Background:** Final-year engineering student (RV College of Engineering), implementing this as a research project with a defined 8-week window.
- **Goals:** Get a working RKD pipeline running, produce defensible empirical results (alignment metrics, downstream accuracy) for a report/thesis, and avoid wasting GPU budget on dead-end experiments.
- **Pain points:** Limited compute budget; needs the teacher model's forward passes to be cached, not repeated; needs fast iteration on loss-function changes without re-running full training.
- **Technical proficiency:** High — comfortable with PyTorch internals, hooks, and Hugging Face ecosystem, but time-constrained.
- **Usage scenario:** Spends Week 5–6 almost entirely inside the RKD loss implementation, running small-scale overfitting tests on a handful of batches to sanity-check that distance/angle penalties behave correctly before scaling to full training.

### Persona 2 — Priya, the ML Infrastructure Reviewer
- **Background:** A senior engineer or advisor reviewing the project's technical soundness (could be a professor, mentor, or technical reviewer for a report submission).
- **Goals:** Verify the RKD math is correctly implemented and that claimed results (alignment collapse avoidance) are empirically supported, not just asserted.
- **Pain points:** Needs to see ablations — RKD vs. no-RKD vs. point-wise-KD baseline — not just a single trained model's metrics in isolation.
- **Technical proficiency:** High; will read the loss function code directly.
- **Usage scenario:** Runs `evaluate.py` against three checkpoints (baseline cascade, point-wise KD student, RKD student) and compares the alignment-score table from Section 14 to assess whether the central hypothesis holds.

### Persona 3 — Devraj, the Downstream App Developer
- **Background:** Builds a voice-assistant feature for an edge device (e.g., a smart speaker or automotive system) and wants to use this model as a backbone.
- **Goals:** Integrate the trained checkpoint via a simple inference API; needs to know latency/memory numbers up front to decide if it fits his device's constraints.
- **Pain points:** Doesn't want to understand RKD internals — just needs a clean inference interface (Section 8) and clear hardware requirements.
- **Technical proficiency:** Medium-high (competent engineer, not necessarily an ML researcher).
- **Usage scenario:** Loads the quantized checkpoint, calls `AudioLLM.generate(waveform, prompt="What is the customer's intent?")`, and checks the returned latency against his device's real-time budget.

### Persona 4 — Ananya, the Dataset Curator
- **Background:** Responsible for sourcing and validating the code-switched (Hinglish), noise-augmented training data described in Phase 1.
- **Goals:** Produce a clean, well-labeled, license-compliant dataset that covers diverse accents, noise profiles, and intents.
- **Pain points:** Code-switched speech datasets are scarce; needs tooling to validate transcript quality and noise-SNR labeling before it reaches the training pipeline.
- **Technical proficiency:** Medium — comfortable with data tooling and scripting, less so with the model internals.
- **Usage scenario:** Runs the data validation scripts (Section 5.3) against a newly sourced batch of recordings to flag mismatched transcripts or corrupted audio before they enter training.

### Persona 5 — Karthik, the MLOps / Training-Infra Owner
- **Background:** Owns the training cluster/GPU allocation and CI for the project.
- **Goals:** Make sure training jobs are reproducible, checkpointed, resumable, and that WandB logging is consistent across runs so experiments are comparable.
- **Pain points:** Needs config-driven runs (no hardcoded paths/hyperparameters in code) so he can launch sweeps without editing source.
- **Technical proficiency:** High in infra/DevOps, medium in ML modeling specifics.
- **Usage scenario:** Sets up a WandB sweep across `α` and `β` (RKD loss weights) using the YAML config schema in Section 6, and monitors GPU memory/throughput dashboards during Phase 4 training runs.

---

## 4. User Journey Mapping

### Journey: First-Time Contributor (e.g., a new team member joining in Week 3)

| Step | Action | "Screen" / Artifact | Backend interaction | Possible failure points |
|---|---|---|---|---|
| 1 | Clone repo, read `README.md` and this kickoff doc | Terminal / repo root | Git clone | Missing dependency versions not pinned → environment mismatch |
| 2 | Run `setup_env.sh` to create conda/venv environment | Terminal | Installs PyTorch, TorchAudio, transformers, PEFT, TRL | CUDA version mismatch with installed PyTorch wheel |
| 3 | Run a smoke test: `python scripts/smoke_test.py` | Terminal | Loads encoder + tiny decoder, runs one forward pass on a sample clip | OOM if smoke test accidentally loads the full teacher model instead of a tiny stub |
| 4 | Inspect a sample config (`configs/baseline.yaml`) | Config file | None (static read) | Config schema drift if someone changed the dataclass without updating the example |
| 5 | Launch a 10-step debug training run | Terminal / WandB dashboard | Loads cached teacher embeddings, runs training loop for 10 steps | Cache missing → falls back to live teacher forward pass, silently 10x slower; should warn explicitly |
| 6 | Open WandB run, confirm loss curves are logging | Browser / WandB | WandB API write | API key not configured → run logs locally only, contributor doesn't realize |

### Journey: Returning Researcher (e.g., Abhinandan resuming work in Week 6 after Phase 2/3 are done)

| Step | Action | Artifact | Backend interaction | Failure points |
|---|---|---|---|---|
| 1 | Pull latest changes, check which checkpoint is "current best" via `checkpoints/MANIFEST.json` | Repo / manifest file | None | Manifest not updated by previous session → resumes from stale checkpoint |
| 2 | Resume RKD training from last checkpoint | Terminal | Loads optimizer + model state, resumes step counter | Optimizer state mismatch if LoRA config changed between sessions (e.g., rank changed) |
| 3 | Modify `α`/`β` loss weights based on prior week's alignment metrics | Config file | None | No validation that `α, β >= 0`; a negative weight would silently invert the loss |
| 4 | Re-run training for remaining budgeted steps | Terminal / WandB | Forward+backward passes, periodic checkpointing | GPU preemption (shared cluster) without checkpoint-on-interrupt handling → lost progress |
| 5 | Run evaluation harness against the new checkpoint | Terminal | Loads eval split, runs inference, computes alignment + task metrics | Eval set accidentally overlapping with an earlier training data refresh (data leakage) |

### Journey: Power User / Downstream Integrator (Devraj deploying the model)

| Step | Action | Artifact | Backend interaction | Failure points |
|---|---|---|---|---|
| 1 | Pull the published quantized checkpoint + inference package | Package registry / release artifact | Download | Checkpoint not actually NF4-quantized if export step was skipped — must verify checksum/format |
| 2 | Instantiate `AudioLLM` inference class with the checkpoint path | Python script | Loads encoder, projection bridge, quantized decoder + LoRA weights merged | LoRA weights not merged into base weights before quantization → runtime errors or silently using un-adapted base model |
| 3 | Call `.generate(waveform, prompt)` on a sample clip | Python script | Encoder forward → projection → decoder generate loop | Audio not resampled to 16kHz before being passed in → garbage output with no error raised |
| 4 | Measure latency/memory on target hardware | Profiling script | torch.profiler / process memory inspection | Profiling on CPU when target is GPU-edge (e.g., Jetson) gives misleading numbers |
| 5 | Integrate into the larger application, set a timeout/fallback policy | Application code | Calls the inference API as a library | No documented worst-case latency for long audio clips → app hangs if a 5-minute clip is passed in without chunking |

---

## 5. System Architecture

### 5.1 High-Level Architecture Diagram

```
                                  ┌─────────────────────────────────────────┐
                                  │              TRAINING TIME              │
                                  │                                         │
  ┌───────────────┐               │   ┌─────────────────────────────────┐   │
  │  Raw Datasets  │──────────────┼──▶│   Dataset Loader & Augmenter    │   │
  │ (Hinglish, ASR │               │   │  (noise injection, padding,     │   │
  │  noisy corpora)│               │   │   transcript/intent labels)     │   │
  └───────────────┘               │   └────────────────┬────────────────┘   │
                                  │                     │ batched waveforms │
                                  │                     ▼                   │
                                  │   ┌─────────────────────────────────┐   │
                                  │   │     STUDENT PATH                │   │
                                  │   │  ┌───────────────────────────┐  │   │
                                  │   │  │  Audio Encoder (frozen)   │  │   │
                                  │   │  │  WavLM / Whisper-encoder  │  │   │
                                  │   │  └─────────────┬─────────────┘  │   │
                                  │   │                ▼                │   │
                                  │   │  ┌───────────────────────────┐  │   │
                                  │   │  │   Projection Bridge       │◀─┼───┼── trainable
                                  │   │  │ (Linear/Conv + LayerNorm) │  │   │
                                  │   │  └─────────────┬─────────────┘  │   │
                                  │   │                ▼                │   │
                                  │   │  ┌───────────────────────────┐  │   │
                                  │   │  │ Compact Decoder (<2B)     │◀─┼───┼── LoRA r=16, trainable
                                  │   │  │ + LoRA adapters           │  │   │
                                  │   │  └─────────────┬─────────────┘  │   │
                                  │   │                │  hooks ────────┼───┼──┐
                                  │   └────────────────┼────────────────┘   │  │
                                  │                     ▼                   │  │
                                  │            student logits → L_CE        │  │
                                  │                                         │  │
                                  │   ┌─────────────────────────────────┐   │  │
                                  │   │     TEACHER PATH (frozen)       │   │  │
                                  │   │  ┌───────────────────────────┐  │   │  │
                                  │   │  │  Large Teacher Audio-LLM  │  │   │  │
                                  │   │  │  (precomputed/cached)     │  │   │  │
                                  │   │  └─────────────┬─────────────┘  │   │  │
                                  │   │                │  hooks ────────┼───┼──┤
                                  │   └────────────────┼────────────────┘   │  │
                                  │                     ▼                   │  │
                                  │           ┌───────────────────┐         │  │
                                  │           │   RKD ENGINE      │◀────────┼──┘
                                  │           │ Distance penalty  │         │
                                  │           │ Angular penalty   │         │
                                  │           └─────────┬─────────┘         │
                                  │                     ▼                   │
                                  │     L_total = L_CE + αL_RKD-D + βL_RKD-A│
                                  └─────────────────────────────────────────┘

                                  ┌─────────────────────────────────────────┐
                                  │             INFERENCE TIME              │
                                  │  Raw waveform                            │
                                  │       ▼                                  │
                                  │  Audio Encoder (frozen, same as above)   │
                                  │       ▼                                  │
                                  │  Projection Bridge (trained weights)     │
                                  │       ▼                                  │
                                  │  Compact Decoder + merged LoRA weights   │
                                  │  (NF4 quantized for edge deployment)     │
                                  │       ▼                                  │
                                  │  Generated text (intent / summary / etc)│
                                  └─────────────────────────────────────────┘
```

### 5.2 Component Breakdown

**"Frontend" — CLI & Config Layer (no GUI in this project)**
- All interaction is via CLI entry points (`train.py`, `evaluate.py`, `infer.py`) and YAML configs (Section 6 schema).
- Rationale: this is a research/infra project; a GUI would add development cost with no corresponding user need at this stage. A minimal Streamlit/Gradio demo is listed as a Phase 4 stretch item only if time allows (Section 15).

**"Backend" — Training & Inference Engine**
- A Python package (`audiollm_rkd/`) containing model definitions, the RKD loss module, dataset loaders, and training/eval loops, built on PyTorch + Hugging Face `transformers`/`PEFT`/`TRL`.
- Responsible for orchestrating the forward passes of encoder → bridge → decoder, attaching/removing forward hooks, computing losses, and checkpointing.

**Database — Experiment & Artifact Tracking (not a traditional relational DB)**
- No transactional database is required for this project. Persistent state lives in:
  - **Filesystem checkpoints** (`checkpoints/<run_id>/step_<n>.pt`) for model weights/optimizer state.
  - **WandB** for run metadata, metrics, and hyperparameters (acts as the experiment "database").
  - **A flat-file dataset manifest** (JSON/CSV) mapping audio file paths to transcripts/labels/noise metadata — see Section 7 for schema.

**Authentication**
- No end-user authentication exists. The only credential surface is the WandB API key (for logging) and, if applicable, a Hugging Face access token (for downloading gated teacher/encoder checkpoints). Both are managed via environment variables / `.env`, never committed to source control.

**APIs**
- An internal Python inference interface (`AudioLLM` class — fully specified in Section 8) functions as the "API" for downstream consumers, since there is no networked product surface in the 8-week scope. A thin FastAPI wrapper around this class is documented as a Future Feature (F1-adjacent) for teams that need a network-accessible endpoint.

**Caching**
- **Teacher activation caching** is the single most important performance optimization in this project (Section 2.2, Cost Considerations). Since the teacher is frozen, its hidden-layer activations for the entire training set can be computed once and cached to disk (e.g., as memory-mapped `.npy`/`.pt` shards keyed by sample ID), avoiding a teacher forward pass on every training step of every epoch.
- A secondary cache stores preprocessed/augmented audio features when augmentation is deterministic (e.g., fixed noise profile per sample) to avoid redundant resampling/feature extraction.

**Storage**
- Raw datasets and noise profile libraries: local disk or mounted network storage (sized per Section 7).
- Checkpoints and cached teacher activations: local disk, with periodic sync to cloud storage (S3-compatible bucket or equivalent) for durability given multi-week training runs.

**Monitoring**
- WandB dashboards for loss curves (`L_CE`, `L_RKD-Distance`, `L_RKD-Angle`, `L_total`), alignment metrics (Section 14), GPU utilization/memory, and throughput (samples/sec).
- A lightweight `assert`-based sanity-check layer (Section 13) that halts training on NaN/Inf losses or degenerate (zero-variance) embeddings, rather than letting a corrupted run continue silently.

### 5.3 Data Flow (Step-by-Step)

1. **Ingestion:** Raw audio + transcript/label files are validated by a dataset-curation script (run by Persona 4 — Ananya) that checks sample rate, clipping, transcript-audio length consistency, and license metadata. Validated samples are written into the manifest (Section 7).
2. **Augmentation:** At load time, the dataset loader optionally injects background noise from the noise-profile library at a configured SNR range, and may apply code-switch-aware text normalization to transcripts.
3. **Teacher Caching (one-time, pre-training):** The frozen teacher model processes the entire training set once; intermediate-layer activations at the hooked layer(s) are cached to disk keyed by sample ID.
4. **Student Forward Pass (every training step):** Waveform → frozen audio encoder → trainable projection bridge → compact decoder (base frozen + LoRA trainable) → output logits. Forward hooks on the projection bridge output (or an early decoder layer) capture the student's intermediate representation for the same samples.
5. **Loss Computation:** Cross-entropy loss on the decoder's output against target text; RKD distance and angle losses computed between the cached teacher activations and the live student activations for the batch.
6. **Backward Pass & Optimization:** Gradients flow into the projection bridge and LoRA adapters only (audio encoder and decoder base weights remain frozen) via an AdamW optimizer with a configured learning rate schedule.
7. **Checkpointing:** Every N steps, model + optimizer + step counter are serialized; the manifest of "best checkpoint so far" is updated based on validation alignment score.
8. **Evaluation:** Periodically (and at the end of training), the evaluation harness runs the student against held-out data for both downstream task accuracy and alignment diagnostics, logging results to WandB and a structured JSON report.
9. **Inference (post-training):** The trained projection bridge and merged LoRA weights are packaged with the frozen encoder and quantized decoder into a deployable artifact, exposed via the `AudioLLM` inference class.
## 6. Technology Stack Selection

| Layer | Choice | Why chosen |
|---|---|---|
| **Language** | Python 3.10+ | Required by the entire PyTorch/Hugging Face ecosystem; no alternative seriously considered for ML research code. |
| **Deep Learning Framework** | PyTorch + TorchAudio | Native forward-hook support (critical for the RKD engine — hooks are a first-class PyTorch feature), broad pretrained-audio-encoder availability via TorchAudio/Hugging Face, and the team's existing familiarity. **Alternative considered:** JAX/Flax — rejected because the Hugging Face ecosystem (PEFT, TRL) and most pretrained audio encoders are PyTorch-first; porting would cost time the 8-week schedule doesn't have. |
| **Model Ecosystem** | Hugging Face `transformers`, `PEFT`, `TRL`, `Unsloth` | `transformers` provides standardized loading of WavLM/Whisper encoders and candidate compact decoders; `PEFT` provides battle-tested LoRA implementations (no need to hand-roll low-rank adapter injection); `Unsloth` provides memory-efficient training kernels that materially matter given the GPU-budget constraint described in Section 2.2. **Alternative considered:** hand-rolled LoRA — rejected as unnecessary reinvention; PEFT's implementation is correct and well-tested. |
| **Quantization** | `bitsandbytes` NF4 | NF4 (4-bit NormalFloat) is the quantization scheme explicitly specified in the source report (Section 4, Implementation Strategy) for the compact decoder; it's well-integrated with PEFT/QLoRA-style training and has strong empirical support for preserving accuracy at 4-bit. **Alternative considered:** GPTQ/AWQ — viable post-training alternatives, but NF4's tight integration with PEFT's QLoRA training path (fine-tune-then-quantize in one pipeline) fits the project's LoRA-based approach more directly. |
| **Experiment Tracking** | Weights & Biases (WandB) | Explicitly specified in the source PRD; provides loss-curve visualization, hyperparameter sweep tooling, and artifact versioning out of the box. **Alternative considered:** TensorBoard — simpler but lacks built-in sweep orchestration, which matters for the α/β loss-weight tuning described in Section 2.1 (C4). |
| **Audio Encoder (primary)** | WavLM-Large | Self-supervised, strong general acoustic representations including non-phonetic information (speaker, prosody) that pure ASR-objective encoders may suppress — directly relevant since the project's value proposition depends on retaining paralinguistic signal. **Alternative considered:** Whisper encoder — also supported as a swappable config option (Section 2.1, C1) since it benefits from massive multilingual pretraining, which may help with Hinglish code-switching specifically; both should be benchmarked rather than betting on one (see Section 18 risk). |
| **Compact Decoder** | A sub-2B parameter open-weight causal LLM (e.g., a model in the Llama-3.2-1B / Qwen2.5-1.5B / Phi-3.5-mini class) | Must be small enough for edge deployment after NF4 quantization, and must be available with the architecture exposed clearly enough to accept continuous embeddings in place of (or alongside) discrete token embeddings — i.e., the model's `inputs_embeds` interface must be usable, which is standard in Hugging Face `transformers` causal LMs. **Specific model selection is a Phase 1 task** (Section 9), not fixed in this document, because it depends on which checkpoint best supports multilingual (Hindi-English) tokenizer coverage. |
| **Teacher Model** | A large, open or accessible end-to-end audio-LLM with documented intermediate layer structure | Must expose intermediate activations via hooks and ideally already perform reasonably on code-switched/noisy audio so its representation space is worth distilling. **This is a Phase 1 selection task**, not fixed here — candidates should be evaluated for (a) hookability, (b) license terms permitting use as a distillation teacher, and (c) reasonable inference cost for the one-time caching pass in Section 5.3. |
| **Cloud/Compute Infrastructure** | Single-node multi-GPU (or single-GPU for dev iteration), orchestrated via Hugging Face `accelerate` | `accelerate` allows the same training script to run on 1 GPU (dev) or N GPUs (full training) via config only, satisfying the scalability NFR without separate code paths. **Alternative considered:** a full Kubernetes-based distributed training setup — explicitly rejected as disproportionate to an 8-week research project; this is noted as a Future Feature only if the project graduates into a longer-lived effort. |
| **CI** | GitHub Actions running lint + unit tests (loss functions, dataset loader, config validation) on every PR | Lightweight, free for the project's likely scale, integrates with GitHub-hosted repo. Does **not** run full training in CI (too expensive/slow) — only fast unit tests (Section 13). |
| **Monitoring (training-time)** | WandB (metrics) + a custom `assert`-based guard layer (Section 13) for NaN/degenerate-embedding detection | Covered above; the guard layer is necessary because WandB monitoring is passive (a human has to notice a NaN in a chart), whereas the project needs a fail-fast mechanism given limited compute budget — a run that silently produces garbage for 6 hours before a human checks WandB is a real cost risk. |

### 6.1 Alternative Stack Considered (Rejected) — "Fully Custom, No Hugging Face"

A from-scratch implementation (custom transformer decoder, hand-rolled LoRA, no `transformers`/`PEFT` dependency) was considered and rejected. Tradeoff: it would offer maximal control over the decoder's embedding-input interface, but the 8-week timeline cannot absorb reimplementing a correct, performant transformer decoder and LoRA mechanism from scratch when mature, tested implementations exist. The project's actual research contribution is the **RKD engine**, not the decoder architecture — engineering effort should concentrate there.

---

## 7. Dataset & Feature Schema Design

This project has no relational database in the traditional sense. The equivalent design artifact is the **dataset manifest schema** and the **tensor/feature interface contracts** between pipeline components — these are the structures every engineer touching the codebase needs to agree on.

### 7.1 Dataset Manifest — Entity Relationship (ASCII)

```
┌───────────────────────┐        ┌──────────────────────────┐
│       AudioSample      │        │      NoiseProfile         │
├───────────────────────┤        ├──────────────────────────┤
│ sample_id (PK)         │        │ noise_id (PK)              │
│ audio_path             │        │ noise_path                 │
│ duration_sec           │        │ category (siren/machinery/ │
│ sample_rate            │        │   crowd/etc.)               │
│ language_tags[]        │        │ avg_db_level                │
│ transcript              │        └──────────────────────────┘
│ intent_label (nullable)│                    ▲
│ summary_label(nullable)│                    │ used_by (M:N via AugmentationEvent)
│ speaker_id (nullable)  │                    │
│ license_id (FK)        │        ┌──────────────────────────┐
│ split (train/val/test) │        │   AugmentationEvent        │
└───────────┬───────────┘        ├──────────────────────────┤
            │ 1:M                │ event_id (PK)               │
            └───────────────────▶│ sample_id (FK)              │
                                  │ noise_id (FK, nullable)    │
                                  │ snr_db                     │
                                  │ applied_at_train_step       │
                                  └──────────────────────────┘

┌───────────────────────┐
│        License          │
├───────────────────────┤
│ license_id (PK)         │
│ source_dataset_name     │
│ terms_url               │
│ permits_model_training  │ (boolean — must be TRUE to be ingested)
└───────────────────────┘
```

### 7.2 Manifest Fields (Implemented as a flat JSONL or Parquet file, not a live RDBMS)

**AudioSample**
| Field | Type | Constraints |
|---|---|---|
| `sample_id` | string (UUID) | Primary key, unique |
| `audio_path` | string | Must resolve to an existing file at load time |
| `duration_sec` | float | > 0; samples > 30s are chunked at load time, not stored pre-chunked |
| `sample_rate` | int | Must be resampled to 16000 if different; stored value is the *original* rate for provenance |
| `language_tags` | list[string] | e.g., `["hi", "en"]` for Hinglish; non-empty |
| `transcript` | string (nullable) | Required for supervised CE loss samples; nullable for ambient-only/self-supervised samples |
| `intent_label` | string (nullable, enum) | One of a fixed taxonomy defined in `configs/intent_labels.yaml`; nullable if sample isn't part of the intent-classification eval set |
| `summary_label` | string (nullable) | Reference summary for the audio-summarization eval task |
| `speaker_id` | string (nullable) | For diversity tracking/stratified splitting; not used for diarization in this phase |
| `license_id` | string (FK) | Must reference a `License` row where `permits_model_training = TRUE` — **ingestion pipeline must reject any sample failing this check** |
| `split` | enum(`train`,`val`,`test`) | Assigned at ingestion; **must be stratified by language_tags and intent_label** to avoid skewed splits (Hinglish samples are likely a minority class and must appear in all three splits) |

**Indexes:** `sample_id` (primary), composite index on `(split, intent_label)` for fast stratified batch sampling during training.

**Normalization rationale:** `NoiseProfile` is kept as a separate entity (not flattened into `AudioSample`) because noise profiles are reused across many samples via on-the-fly augmentation (the M:N relationship via `AugmentationEvent`) — duplicating noise metadata per sample would be redundant and would make it harder to swap/extend the noise library independently of the speech corpus. `License` is separated similarly because a single source dataset's license applies to many samples; storing license terms per-sample would risk drift if terms are corrected later.

### 7.3 Tensor Interface Contracts (the equivalent of "API request/response shapes" for this project)

Since there are no REST endpoints between pipeline stages in the training loop (it's one Python process), the binding contracts are **tensor shape and dtype agreements** between modules. These must be treated with the same rigor as an API contract — a shape mismatch here is the project's equivalent of a 400 error.

| Interface | Producer | Consumer | Shape | Dtype | Notes |
|---|---|---|---|---|---|
| Raw waveform batch | Dataset loader | Audio encoder | `(B, num_samples)` | `float32` | Padded to max length in batch; accompanied by `attention_mask: (B, num_samples)` |
| Encoder hidden states | Audio encoder | Projection bridge | `(B, T_audio, D_enc)` | `float32` | `D_enc` = 1024 for WavLM-Large; `T_audio` depends on encoder frame rate (~20ms/frame) |
| Projected embeddings | Projection bridge | Decoder (`inputs_embeds`) | `(B, T_proj, D_dec)` | `float32` (or `bfloat16` under mixed precision) | `D_dec` must exactly equal the decoder's embedding dimension; mismatch here is a silent shape-broadcast bug if not explicitly asserted |
| Teacher activations (cached) | Teacher model (precomputed) | RKD loss module | `(B, T_teacher, D_teacher)` | `float32`, stored as `float16` on disk to halve cache size | `T_teacher` and `T_proj` will generally **not** match — Section 9 specifies the alignment strategy |
| Decoder logits | Decoder | CE loss | `(B, T_out, vocab_size)` | `float32` | Standard causal LM output |
| Alignment scores (eval) | Evaluation harness | WandB / JSON report | scalar per metric per checkpoint | `float32` | See Section 14 for the exact metric definitions |

---

## 8. Inference & Module Interface Design

There is no public REST/GraphQL API in this 8-week scope (no networked product surface — see Section 5.2). The binding interface specification instead covers (a) the **Python inference class** downstream consumers use, and (b) the **internal module interfaces** engineers building the training pipeline must implement against. This section is written with the same completeness a REST spec would require, substituting function signatures for endpoints.

### 8.1 Public Inference Interface — `AudioLLM`

```python
class AudioLLM:
    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cuda",
        quantized: bool = True,
    ) -> None:
        """
        Loads the frozen audio encoder, trained projection bridge,
        and the compact decoder with merged LoRA weights (optionally
        NF4-quantized) from a packaged checkpoint directory.

        Raises:
            CheckpointFormatError: if checkpoint_path does not contain
                the expected encoder/bridge/decoder sub-directories.
            DeviceUnavailableError: if device="cuda" but no GPU is visible.
        """

    def generate(
        self,
        waveform: np.ndarray,         # mono float32, any sample rate
        sample_rate: int,
        prompt: str | None = None,    # optional text instruction, e.g. "Summarize this audio."
        max_new_tokens: int = 128,
    ) -> AudioLLMResponse:
        """
        Runs the full encoder -> bridge -> decoder pipeline and returns
        generated text plus timing/metadata.

        Edge cases handled internally:
          - Resamples waveform to 16kHz if sample_rate != 16000.
          - Chunks audio longer than 30s using a sliding window with
            2s overlap; long-clip behavior is documented, not silent.
          - Raises EmptyAudioError if waveform is silence-only (RMS below
            a configured floor) rather than returning a meaningless
            generation.

        Returns:
            AudioLLMResponse(
                text: str,
                latency_ms: float,
                input_duration_sec: float,
                truncated: bool,   # True if input exceeded max context and was chunked/cut
            )
        """

    def batch_generate(
        self,
        waveforms: list[np.ndarray],
        sample_rates: list[int],
        prompts: list[str] | None = None,
        max_new_tokens: int = 128,
    ) -> list[AudioLLMResponse]:
        """Batched version for throughput-sensitive offline use cases."""
```

**Error cases and how they surface (the inference-API equivalent of HTTP status codes):**

| Condition | Behavior |
|---|---|
| Malformed/corrupted audio input | Raises `AudioDecodeError` with the offending file/array description — never silently returns empty output. |
| Audio shorter than one encoder frame (~20ms) | Raises `AudioTooShortError`. |
| Checkpoint missing the projection bridge weights (e.g., only base decoder was packaged) | Raises `CheckpointFormatError` at `__init__` time, not at first `generate()` call — fail fast. |
| Decoder context window exceeded after chunking | Returns a response with `truncated=True` rather than raising, since partial output is often still useful; the caller is responsible for deciding whether `truncated=True` is acceptable for their use case. |
| GPU OOM during generation | Raises `torch.cuda.OutOfMemoryError` (not caught/swallowed) — the caller decides retry/fallback policy; this library does not silently fall back to CPU, since that would produce wildly inconsistent latency without the caller's knowledge. |

### 8.2 Internal Module Interfaces (for engineers implementing Sections 9–14)

```python
class AudioEncoder(Protocol):
    """Wraps WavLM/Whisper-encoder; must be implemented identically
    for both so they're swappable via config (Section 6)."""
    def forward(self, waveform: Tensor, attention_mask: Tensor) -> Tensor:
        """Returns (B, T_audio, D_enc). Must be called under torch.no_grad()
        — this component is always frozen in this project's scope."""

class ProjectionBridge(nn.Module):
    """The only audio-side trainable component besides LoRA adapters."""
    def forward(self, encoder_states: Tensor) -> Tensor:
        """(B, T_audio, D_enc) -> (B, T_proj, D_dec).
        Must apply LayerNorm on output (Section 2.1, C2) to prevent
        scale mismatch with the decoder's native token embedding norms."""

class RKDLoss(nn.Module):
    """Pure-tensor loss module — must be unit-testable without a live model."""
    def forward(
        self,
        student_activations: Tensor,  # (B, T_proj, D_student_hook)
        teacher_activations: Tensor,  # (B, T_teacher, D_teacher_hook)
    ) -> dict[str, Tensor]:
        """Returns {"distance_loss": scalar, "angle_loss": scalar}.
        Internally handles the T_proj != T_teacher alignment problem
        per the strategy specified in Section 9.3 before computing
        pairwise distances/angles."""
```

This Protocol/ABC-based design is what makes the "swap encoder/decoder/teacher via config, not code" maintainability requirement (Section 2.2) actually enforceable — any new encoder or decoder must implement these exact interfaces to be a legal drop-in replacement, and this should be enforced with `isinstance`/type-checking in CI (Section 13), not just a docstring convention.
## 9. RKD Mathematical Specification & Alignment Strategy

This section is the technical core of the project and is specified at the level of detail an engineer needs to implement `RKDLoss` directly, since the source documents describe the loss *conceptually* (Section 3 of the report) but not at implementation precision.

### 9.1 The Sequence-Length Mismatch Problem

The teacher and student will almost never produce the same number of hidden-state frames at the hooked layer (`T_teacher != T_proj`), because:
- The teacher's audio front-end may use a different frame rate than the student's encoder.
- The student's projection bridge may apply temporal downsampling (Section 2.1, C2) to shorten the sequence fed into the compact decoder, which the teacher does not do.

**Required alignment strategy before computing any RKD loss:**

1. Compute a per-batch alignment via **linear interpolation** of the teacher's sequence onto the student's `T_proj` grid (using `torch.nn.functional.interpolate` with `mode="linear"` along the temporal dimension), **not** truncation or naive zero-padding, since truncation would discard information non-uniformly across the clip and naive padding would inject zero-vectors into a distance computation, corrupting it.
2. As a configurable alternative (for ablation, since this is itself a research decision worth testing), support **nearest-frame mapping**: for each student frame `i`, map to the teacher frame at index `round(i * T_teacher / T_proj)`.
3. Log which alignment strategy was used in every training run's config snapshot (WandB), since this is exactly the kind of implementation detail that affects results and must be reproducible.

### 9.2 Projecting Teacher and Student Activations to a Common Dimension

The teacher's hooked hidden dimension (`D_teacher`) and the student's hooked hidden dimension (`D_student_hook`) will generally differ (e.g., a large teacher's hidden size vs. the compact decoder's hidden size). RKD's distance and angle penalties are computed over **relative geometry**, which is dimension-invariant in principle (pairwise distances/angles don't require matching dimensionality directly) — but in practice, a **learned auxiliary linear projection** (separate from the main projection bridge, used *only* inside the loss computation, with its own small set of trainable parameters) is added to map student hook-layer activations into `D_teacher`-dimensional space before computing distances. This auxiliary projection is trained jointly via the RKD loss gradient and is **discarded at inference time** — it exists purely to make the geometry comparison well-posed, and downstream consumers (Section 8) never see it.

*Rationale documented for reviewers (Persona 2):* computing distances directly between mismatched dimensions is mathematically undefined without first establishing a common space; an alternative of simply truncating/padding raw dimensions was considered and rejected because it would not preserve any meaningful geometric correspondence between the two spaces.

### 9.3 Distance Penalty — `L_RKD-Distance`

For a batch of aligned student embeddings `{s_1, ..., s_n}` and aligned, projected teacher embeddings `{t_1, ..., t_n}` (where `n = T_proj` after alignment, flattened across the batch for the pairwise computation):

1. Compute pairwise Euclidean distances within each set:
   - `d_student(i,j) = ||s_i - s_j||_2`
   - `d_teacher(i,j) = ||t_i - t_j||_2`
2. **Normalize** each distance matrix by its own mean pairwise distance (this is the "normalized Euclidean distance" referenced in the source report, Section 3) so the loss is invariant to the absolute scale of each model's embedding space — without this normalization, the loss would just push the student to match the teacher's arbitrary scale, not its *relational structure*:
   - `d_hat_student(i,j) = d_student(i,j) / mean(d_student)`
   - `d_hat_teacher(i,j) = d_teacher(i,j) / mean(d_teacher)`
3. Apply a Huber loss (smooth L1) between the two normalized distance matrices, rather than raw MSE, to reduce sensitivity to outlier pairs (e.g., a single corrupted frame shouldn't dominate the gradient):
   ```
   L_RKD-Distance = HuberLoss(d_hat_student, d_hat_teacher)
   ```

### 9.4 Angular Penalty — `L_RKD-Angle`

For triplets of points `(s_i, s_j, s_k)` in the student space and the corresponding `(t_i, t_j, t_k)` in the teacher space:

1. Compute the angle at vertex `j` formed by vectors to `i` and `k`:
   - `e_student(i,j,k) = cos_angle( (s_i - s_j), (s_k - s_j) )`
   - `e_teacher(i,j,k) = cos_angle( (t_i - t_j), (t_k - t_j) )`
   where `cos_angle(u, v) = (u · v) / (||u|| ||v|| + eps)` — the `eps` (e.g., `1e-8`) is **mandatory**, not optional, to prevent division-by-zero when two embeddings collapse to the same point (which is exactly the failure mode — alignment collapse — this loss is designed to detect and penalize; a NaN here would silently mask the very problem the loss exists to catch).
2. Apply Huber loss between student and teacher angle sets:
   ```
   L_RKD-Angle = HuberLoss(e_student, e_teacher)
   ```
3. **Sampling strategy:** computing all `O(n^3)` triplets is computationally prohibitive for any reasonable sequence length. Sample a fixed number of triplets per batch (configurable, default 1,000) uniformly at random per training step, rather than exhaustively enumerating — this must be implemented with `torch.randint`-based index sampling, vectorized (no Python-level loops over triplets), to keep the RKD engine from becoming the training bottleneck.

### 9.5 Combined Loss

```
L_total = L_CE + α * L_RKD-Distance + β * L_RKD-Angle
```

- `L_CE`: standard next-token cross-entropy on the decoder's output against target text (intent label text, summary text, or transcript depending on the task being trained).
- `α`, `β`: configurable scalar weights. **Starting values for Phase 3 experimentation: `α = 1.0`, `β = 0.5`** — these are *not* claimed to be optimal; Section 15 schedules a sweep across both in Week 6.
- **Required guard:** if `L_RKD-Distance` or `L_RKD-Angle` becomes `NaN`, training must halt immediately with a logged stack trace identifying which batch caused it, rather than `nan_to_num`-ing it away — a silent recovery here would corrupt exactly the metric the entire project is trying to validate (Section 13 specifies this as a CI-tested invariant).

### 9.6 Forward Hook Implementation Notes

```python
# Conceptual sketch — exact layer names depend on the chosen teacher/student
# architectures selected in Phase 1 (Section 6).

activation_cache = {}

def make_hook(name):
    def hook(module, input, output):
        # .detach() is mandatory for the teacher hook — we must never
        # backpropagate into the frozen teacher's graph, both for
        # correctness and to avoid retaining its full computation graph
        # in memory across the entire forward pass.
        activation_cache[name] = output.detach()
    return hook

teacher_layer.register_forward_hook(make_hook("teacher"))
student_layer.register_forward_hook(make_hook("student"))
```

**Two failure modes to explicitly test for in Phase 3 (Section 13):**
1. **Hook accumulation across steps** — if hooks are registered once at model-init time (correct) vs. re-registered every forward pass (incorrect, causes duplicate hooks firing and memory growth over a training run). Unit test must assert exactly one hook is registered per target layer after N training steps.
2. **Hook firing order under `accelerate`/DDP wrapping** — when the model is wrapped for multi-GPU training, the wrapped module's hooked sub-layer must still be reachable; a unit test should register a hook on a DDP-wrapped tiny model and confirm it fires before relying on this in full training.

---

## 10. UI/UX — Minimal Surface for This Project

No end-user-facing UI exists in the 8-week scope (this was scoped out deliberately in Section 5.2). The only "UI" surfaces are:

**Design System for CLI Output:** Training/eval scripts should use structured, consistent console output (a single logging library — `rich` or Python's standard `logging` with a consistent formatter — not ad-hoc `print()` statements scattered across modules), since multiple people (Personas 1, 4, 5) will be reading these logs and inconsistent formatting slows debugging.

**WandB Dashboard Layout (the closest thing to a "screen" in this project):**
- Panel 1: Loss curves (`L_CE`, `L_RKD-Distance`, `L_RKD-Angle`, `L_total`) over training steps.
- Panel 2: Alignment metrics (Section 14) over training steps, with a horizontal reference line marking the point-wise-KD baseline's final alignment score, so collapse-avoidance is visible at a glance rather than requiring a separate comparison step.
- Panel 3: GPU memory and throughput.
- Panel 4: Downstream task accuracy (intent classification, summarization quality) evaluated periodically on the validation split.

**Optional Phase-4 stretch demo (Future Feature, not committed scope):** a minimal Gradio interface (`demo.py`) that lets a reviewer upload an audio clip and see the generated output alongside latency — useful for Persona 2 (the reviewer) to qualitatively sanity-check behavior without writing code, but explicitly not a load-bearing part of the project's deliverables.

---

## 11. Security Architecture

Scoped to what actually applies to a research/ML-infrastructure project with no networked product surface (re-stated and expanded from Section 2.2):

| Concern | Mitigation |
|---|---|
| **Credential handling** | WandB API key and any Hugging Face access tokens are read from environment variables (`.env`, excluded via `.gitignore`) and never hardcoded or logged. CI secrets are stored in GitHub Actions encrypted secrets, not in workflow YAML. |
| **Dataset licensing / provenance** | Enforced at ingestion via the `License.permits_model_training` flag (Section 7.1) — the dataset loader must refuse to load any sample whose license record fails this check, with a hard error, not a warning. |
| **PII in logs** | Transcripts may contain names or other identifying speech content. WandB run logs must log aggregate metrics only, never raw transcript text by default; a separate, access-restricted local debug log (not synced to WandB) may contain raw text for debugging, gated behind an explicit `--debug-log-raw-text` flag that defaults to off. |
| **Model/checkpoint integrity** | Published checkpoints (for downstream consumers like Persona 3) should include a checksum (SHA-256) in the release manifest so integrators can verify they received an uncorrupted artifact — relevant given checkpoints will be transferred across machines/cloud storage repeatedly during the project. |
| **Dependency supply chain** | Pin exact versions in `requirements.txt`/`pyproject.toml` (not loose version ranges) for `torch`, `transformers`, `peft`, `bitsandbytes` — version drift in quantization libraries especially has historically caused silent numerical behavior changes, which would be indistinguishable from a real RKD-collapse finding if it happened mid-project. |

**Threat model summary:** the realistic threats to this project are not "an attacker compromises a running service" (there is no running service) but rather (a) accidental data leakage of PII into shared logs, (b) training on improperly licensed data, and (c) silent numerical/environment drift corrupting results without anyone noticing. The mitigations above are sized to those actual risks rather than importing a generic OWASP Top 10 checklist that mostly doesn't apply to a non-networked research codebase.

---

## 12. DevOps & Deployment

### 12.1 Environments

| Environment | Purpose | Notes |
|---|---|---|
| **Dev** | Local/single-GPU iteration; tiny model stubs and a small data subset (Section 4, "smoke test") for fast feedback loops. | Should run a full train→eval→checkpoint cycle in under 5 minutes on a single consumer GPU to keep iteration fast. |
| **Training (full-scale)** | Multi-GPU node(s) for Phase 4 full training runs, orchestrated via `accelerate`. | This is where teacher-activation caching (Section 5.3) matters most — caching must be verified working *before* committing the full GPU budget to a multi-day run. |
| **Eval/Release** | A fixed environment (pinned dependency versions, Section 11) used to produce the final reported numbers, so results are reproducible by Persona 2 (the reviewer) independent of whatever environment drift happened during iterative development. |

This project does not require a traditional staging/production *web service* environment, since there is no deployed service (Section 5.2). "Production" in this context means **the final packaged checkpoint + inference library**, not a running server.

### 12.2 Containerization

A single `Dockerfile` (CUDA base image matching the pinned PyTorch/CUDA version) is sufficient — there is no need for a multi-service Docker Compose setup, since the project is one Python process at training time and one Python process (or library import) at inference time. Kubernetes orchestration is explicitly out of scope (Section 6.1) given the single-node training/inference model.

### 12.3 CI/CD Pipeline

```
PR opened
   │
   ▼
[GitHub Actions]
   ├─ Lint (ruff/black)
   ├─ Unit tests (Section 13): loss functions, dataset loader, config validation,
   │  hook-registration sanity checks — all run on CPU with tiny stub models, < 5 min total
   ├─ Config schema validation against example configs
   └─ (On merge to main) Tag a release if checkpoint manifest changed; do NOT
      trigger full training from CI — full training is always a manually
      launched job against the training environment, given cost/duration.
```

### 12.4 Backup & Recovery

- Checkpoints synced to durable cloud storage on a schedule (e.g., every saved checkpoint, or at minimum once per epoch) — local-disk-only checkpoints are a real risk given multi-day training runs on shared/preemptible compute.
- Cached teacher activations (Section 5.3) are themselves expensive to regenerate (the whole point of caching is avoiding repeated teacher forward passes), so they must be included in the backup policy, not treated as disposable scratch data.
- **Disaster recovery scope:** if the training environment is lost entirely, recovery means: re-provision environment from the pinned `requirements.txt`/Dockerfile, restore the latest checkpoint and cached teacher activations from cloud storage, resume from the checkpointed step. This should be tested once during Phase 1 (deliberately kill a dev run and practice the resume procedure) rather than assumed to work and discovered broken during Phase 4 under time pressure.

### 12.5 Rollback Strategy

"Rollback" in this project means: if a new training run produces a checkpoint with worse alignment/task metrics than a previous "best" checkpoint, the `checkpoints/MANIFEST.json` "current best" pointer (Section 4, returning-user journey) must **not** be advanced automatically — promotion to "best" requires the evaluation harness (Section 2.1, C6) to confirm an improvement, never just "training completed without crashing."

---

## 13. Testing Strategy

| Test type | Scope | Coverage target | Examples |
|---|---|---|---|
| **Unit tests** | Pure functions: `RKDLoss` (distance/angle math), dataset manifest validation, config schema parsing, alignment-strategy interpolation/nearest-frame mapping. | **90%+** on the loss-function and data-validation modules specifically — these are the modules where a silent bug invalidates the project's core empirical claim, so they warrant higher coverage than typical infra code. | Test that `RKDLoss` returns 0 when student and teacher activations are identical; test that the Huber loss is symmetric; test that the `eps` guard in the angular penalty prevents NaN when two embeddings are made identical on purpose. |
| **Integration tests** | Full forward pass through encoder → bridge → decoder on a tiny stub model (not the real WavLM/teacher, which are too slow for CI); hook registration and firing on a DDP-wrapped tiny model. | **All critical paths covered** (one test per pipeline stage transition), not a percentage target — integration tests here are about catching shape/interface mismatches (Section 8.1's tensor contracts), which is a binary pass/fail concern more than a coverage-percentage concern. | Test that a dummy `(2, 16000)` waveform batch produces a `(2, T_out, vocab_size)` logits tensor without shape errors end-to-end. |
| **End-to-end tests** | `AudioLLM.generate()` against a tiny stub checkpoint, including the documented edge cases from Section 8.1 (silence-only input, oversized input, malformed audio). | One test per documented edge case in Section 8.1's error table — every row in that table must have a corresponding test asserting the documented behavior actually occurs. | Test that `EmptyAudioError` is actually raised for a silence-only waveform, not just documented as intended behavior. |
| **Performance tests** | Throughput (samples/sec) and latency (ms/token) benchmarks, run manually (not in CI, since they require real GPU hardware and the actual-size models) at the end of Phase 4. | N/A (not a coverage metric) — but every reported number in the final results must be reproducible from a checked-in benchmarking script, not a one-off notebook cell. | `benchmark_latency.py --model rkd_student --baseline cascade` producing the comparison table referenced in Section 1.7. |
| **Security/compliance tests** | Automated check that the dataset ingestion script actually rejects a sample with `permits_model_training=False` (Section 7.1, Section 11). | One test, but it is a hard CI gate — should be treated as a blocking test, not a soft warning, given the licensing risk it guards against. | Construct a fixture manifest with one non-permitted sample and assert ingestion raises/excludes it. |

**NaN/degeneracy guard (cross-cutting, tested at the unit level):** a dedicated test must assert that if `L_RKD-Angle` or `L_RKD-Distance` receives `NaN`-containing input tensors, the training loop's guard (Section 9.5) raises a clear, named exception rather than allowing `loss.backward()` to proceed — this is the single most important regression to prevent, since a silent NaN would quietly invalidate exactly the empirical claim (RKD avoids collapse) the entire project exists to demonstrate.
## 14. AI/ML Components — Models, Evaluation, and the Alignment-Collapse Diagnostic

This section receives full depth (not an "if applicable" afterthought) because it is the actual deliverable of this project.

### 14.1 Models

| Model | Role | Trainable? |
|---|---|---|
| Audio Encoder (WavLM-Large or Whisper-encoder, config-selectable per Section 6) | Feature extraction | **Frozen** |
| Projection Bridge | Modality adapter | **Trainable** (this is one of only two trainable components) |
| Compact Decoder base weights | Language reasoning | **Frozen** |
| Compact Decoder LoRA adapters (r=16) | Task/modality adaptation | **Trainable** (the second trainable component) |
| Auxiliary RKD projection (Section 9.2) | Loss-computation-only dimension matching | **Trainable**, discarded at inference |
| Teacher Audio-LLM | Distillation target | **Frozen**, used only to produce cached activations |

This is a deliberately narrow trainable-parameter surface — by design, **only the projection bridge, the LoRA adapters, and the small auxiliary RKD projection receive gradients.** This is what makes an 8-week, limited-compute-budget timeline plausible: the project is not pretraining anything from scratch.

### 14.2 Datasets

Per Phase 1 (Section 15): curated multi-dialect, code-switched (Hinglish) speech with controlled ambient-noise profiles, split per the stratification rule in Section 7.2. Labels required for the two downstream eval tasks named in the source PRD:
- **Intent Classification:** a fixed taxonomy (defined in `configs/intent_labels.yaml`) — exact label set is a Phase 1 deliverable depending on which existing intent-labeled speech corpora can be sourced/licensed (Section 11).
- **Audio Summarization:** reference free-text summaries for longer clips.

### 14.3 Training Pipeline

Specified end-to-end in Section 5.3 (Data Flow) and Section 9 (loss mathematics). No additional pipeline exists beyond what's already specified — repeating it here would only fragment the spec across two sections.

### 14.4 Evaluation Metrics — Task Performance

| Metric | Task | Computation |
|---|---|---|
| Accuracy / Macro-F1 | Intent classification | Standard classification metrics against the fixed taxonomy; macro-F1 specifically because intent classes are expected to be imbalanced (Section 2.1, C5 edge cases). |
| ROUGE-L / BERTScore | Audio summarization | Standard summarization metrics against reference summaries; BERTScore included specifically because ROUGE alone penalizes valid paraphrases, which matters when comparing a model that "hears" continuous audio against a cascaded baseline that necessarily phrases things differently after going through ASR text first. |
| Word/Concept Error Rate on code-switched segments | Robustness check | Measured specifically on the Hinglish-tagged subset of the eval split, reported separately from the aggregate score — an aggregate-only number could hide poor code-switching performance behind good performance on monolingual samples. |

### 14.5 Evaluation Metrics — Alignment-Collapse Diagnostics (the project's core empirical contribution)

This is the metric suite that actually answers the research question. It must be computed for **three models** to be meaningful: (1) the RKD-trained student, (2) a point-wise-KD-trained student (same architecture, trained with standard output-matching distillation instead of RKD — this baseline is a **required deliverable**, not optional, because without it there is no evidence RKD specifically is what avoided collapse), and (3) the frozen teacher itself (as the reference geometry).

**14.5.1 Embedding Rank / Effective Dimensionality**
- Compute the singular value spectrum of the student's hooked-layer activations over a fixed eval batch.
- Report the **effective rank** (e.g., the number of singular values needed to capture 95% of variance).
- *Why this detects collapse:* alignment collapse manifests as the student's embeddings becoming low-rank/degenerate — many distinct audio inputs end up mapping to near-identical embeddings. A healthy student should have effective rank comparable to the teacher's; the point-wise-KD baseline is expected (per the project's central hypothesis) to show a **lower** effective rank than the RKD student.

**14.5.2 Pairwise Distance Correlation**
- For a fixed eval batch, compute the full pairwise distance matrix in both the student's and teacher's embedding spaces (after the same normalization as Section 9.3).
- Compute the **Pearson correlation** between the flattened upper-triangular entries of the two distance matrices.
- *Why this detects collapse:* this is a direct, model-agnostic measurement of whether the *relational structure* RKD is explicitly trained to preserve actually transfers — a high correlation here is the most direct possible evidence supporting the project's central claim, independent of whether downstream task accuracy happens to look fine for other reasons.

**14.5.3 Neighbor Preservation (k-NN agreement)**
- For each point in the eval batch, compute its k=5 nearest neighbors (by embedding distance) in the student space and separately in the teacher space.
- Report the **average overlap** (Jaccard similarity) between the two neighbor sets across all points.
- *Why this detects collapse:* distance correlation (14.5.2) can be misleadingly high if global scale is preserved but local neighborhoods are scrambled; k-NN agreement specifically checks whether "which audio samples the model considers similar to each other" survives compression — directly relevant to the project's claim about preserving semantic structure.

**14.5.4 Collapse Stress Test**
- Construct a small synthetic eval set of near-duplicate audio pairs that differ only in one paralinguistic dimension (e.g., same sentence spoken calmly vs. urgently; same content with vs. without background siren noise).
- Measure the embedding distance between each pair in the student's space.
- *Why this detects collapse:* this is the most direct test of the project's actual motivating claim from Section 1.1 — that paralinguistic/ambient information should be **preserved**, not just that geometry is preserved in the abstract. A collapsed model would map these pairs to near-identical embeddings (correctly capturing the shared lexical content, but failing the project's stated reason for existing); a model preserving the desired information should show measurably non-zero, consistent separation between such pairs.

**14.5.5 Reporting Format**

All four diagnostics must be reported in a single comparison table for every checkpoint evaluated, of this form:

| Model | Effective Rank | Distance Correlation | k-NN Agreement | Stress-Test Separation |
|---|---|---|---|---|
| Teacher (reference) | — (baseline) | 1.00 (self) | 1.00 (self) | (reference value) |
| Point-wise KD student | *(expected lower)* | *(expected lower)* | *(expected lower)* | *(expected near-zero — the collapse failure mode)* |
| RKD student | *(expected closer to teacher)* | *(expected high)* | *(expected high)* | *(expected non-trivial, teacher-comparable)* |

This table, populated with real numbers at the end of Phase 4, **is the central result of the project.** Everything else in this document exists to produce this table correctly and reproducibly.

### 14.6 Inference Architecture

Fully specified in Section 8.1 (`AudioLLM` class).

### 14.7 Monitoring (Post-Training / Drift)

Out of scope for an 8-week research project with no live deployment — there is no production traffic to monitor for drift. Documented here as a Future Feature: if this model is adopted into a long-lived product (per Persona 3's use case), a monitoring plan would need to track input-distribution drift (e.g., new accents, new noise environments not represented in training) against the alignment metrics in Section 14.5, since those metrics are reusable as a deployed-model health check, not just a one-time research diagnostic.

### 14.8 Retraining Strategy

Not committed scope for the 8-week window. Documented for roadmap continuity: if new code-switched dialects, noise environments, or downstream tasks are added later, the recommended approach is **not** full retraining from scratch but re-running Phase 3/4 (RKD distillation + LoRA fine-tuning) against the existing cached teacher activations extended with new samples — this is precisely why the teacher-caching infrastructure (Section 5.3) is designed as a reusable, incrementally extensible artifact rather than a one-off training-time optimization.

---

## 15. Development Roadmap

The roadmap below maps the source PRD's 8-week phase structure onto concrete sprint deliverables, since the original phases were stated as headlines ("Phase 1: Environment & Baseline") without sprint-level granularity.

### Phase 1 — MVP Foundation (Weeks 1–2)
**Features:** Environment setup; encoder/decoder/teacher candidate selection and benchmarking (Section 6); dataset curation pipeline and manifest schema (Section 7) implemented and validated against a first batch of real data; cascaded ASR+LLM baseline pipeline running end-to-end to establish comparison numbers (per the source PRD's explicit Phase 1 goal).
**Deliverables:** `setup_env.sh`; `configs/baseline.yaml`; dataset manifest populated with a first validated batch; cascaded-baseline latency/accuracy numbers logged to WandB as the reference point everything else is compared against.
**Timeline:** Weeks 1–2.

### Phase 2 — Core Engineering (Weeks 3–4)
**Features:** Projection Bridge implemented (C2); LoRA (r=16) attached to the chosen compact decoder via PEFT (C3); end-to-end forward pass (encoder → bridge → decoder) running without RKD yet — i.e., trainable on cross-entropy loss alone, to validate the architecture wiring before adding distillation complexity.
**Deliverables:** A checkpoint trained with `L_CE` only (no RKD), purely to confirm the pipeline is mechanically correct; integration tests from Section 13 passing.
**Timeline:** Weeks 3–4.

### Phase 3 — The RKD Engine (Weeks 5–6)
**Features:** Forward hooks (Section 9.6); distance penalty (9.3) and angular penalty (9.4) implemented and unit-tested in isolation (Section 13) before being wired into the full training loop; teacher-activation caching pipeline (Section 5.3) built and verified; **point-wise KD baseline implemented in parallel** — this is listed explicitly here, not deferred to Phase 4, because the baseline must exist before final comparisons can be made, and because point-wise KD is the simpler of the two to implement (good order-of-operations for an 8-week schedule).
**Deliverables:** `RKDLoss` module passing all unit tests in Section 13; a small-scale (few-batch overfitting) sanity check confirming the loss decreases and doesn't NaN, before committing to full-scale training; α/β sweep launched via WandB (Section 6) to select working loss weights ahead of Phase 4's full run.
**Timeline:** Weeks 5–6.

### Phase 4 — Training, Evaluation & Reporting (Weeks 7–8)
**Features:** Full-scale training run of the RKD student (using the α/β values selected at the end of Phase 3); full-scale training of the point-wise-KD baseline student under matched compute budget (same number of steps/epochs, for a fair comparison); evaluation harness run against all three models (teacher, KD baseline, RKD student) producing the Section 14.5.5 comparison table; latency/memory benchmarking against the Phase 1 cascaded baseline (Section 1.7's "at least 2x speedup" target evaluated here).
**Deliverables:** Final checkpoints (packaged per Section 8.1's `AudioLLM` format); the populated alignment-diagnostics table (Section 14.5.5); the final report/results writeup.
**Timeline:** Weeks 7–8.

*(There is no "Scale" phase within this document's committed scope — scaling considerations are addressed only as Future Features in Section 2.1, since an 8-week research project's roadmap properly ends at "validated result + packaged artifact," not at production scale-out.)*

---

## 16. Team Structure

Sized for a small research team (the source documents indicate a single named author — Abhinandan Jaiswal — so this section describes role *functions* that map onto either one person wearing multiple hats, or a small team if more people are available, rather than assuming a large dedicated team that doesn't match the project's actual scale.)

| Role | Responsibilities | Maps to persona |
|---|---|---|
| **Research Engineer / Lead** | Owns the RKD engine (Section 9), architecture decisions (Section 6), and the final results/report. | Persona 1 (Abhinandan) |
| **Dataset Curator** (can be a part-time/shared responsibility if team is small) | Owns dataset sourcing, licensing verification (Section 11), manifest validation (Section 7). | Persona 4 (Ananya) |
| **MLOps / Infra** (can be the same person as the lead, if no dedicated infra person exists) | Owns environment setup, CI (Section 12.3), checkpoint/backup policy (Section 12.4), WandB sweep configuration. | Persona 5 (Karthik) |
| **Reviewer / Advisor** (external to day-to-day work) | Periodically reviews the alignment-diagnostics table (Section 14.5.5) and the ablation completeness (RKD vs. KD baseline vs. teacher) for scientific rigor. | Persona 2 (Priya) |

No dedicated frontend/backend/UI engineers or a Product Manager role are recommended for this project's actual scope — those roles would be appropriate if this graduated into the productized version referenced in Section 14.7/14.8's Future Features, but allocating them now would mismatch effort against an 8-week research deliverable with no UI surface.

---

## 17. Cost Estimation

Costs are dominated by **GPU compute**, not headcount or third-party SaaS spend, given the project's scope.

| Scenario | GPU compute (training + caching) | Third-party services | Notes |
|---|---|---|---|
| **Low budget** | Single mid-tier GPU (e.g., one 24GB-class card), rented hourly; teacher-activation caching run once, aggressively reused; smaller compact decoder variant chosen to fit memory. | WandB free tier; no paid dataset licenses (rely on permissively licensed/public corpora only). | Highest risk of missing the 8-week timeline if the chosen decoder/teacher pairing turns out to need re-selection mid-project (Section 18 risk) — little budget slack for redoing Phase 1 selection work. |
| **Medium budget** | Multi-GPU node (2–4 GPUs) for Phase 4 full training; teacher caching parallelized across the same node. | WandB team tier (sweep features); modest budget for one or two licensed code-switched speech corpora if public data proves insufficient (a real risk given Hinglish data scarcity, flagged in Section 18). | This is the scenario the timeline in Section 15 is implicitly sized against. |
| **High budget** | Larger multi-GPU allocation, enabling a broader Phase 1 encoder/decoder/teacher benchmarking sweep (testing more candidate combinations than the medium scenario can afford) and a larger α/β hyperparameter sweep in Phase 3. | Paid licensed datasets covering more accents/noise conditions; potential paid access to a stronger teacher model if a suitable open one underperforms. | Reduces the risk items in Section 18 related to data scarcity and teacher-model selection, at added cost. |

**Cost-saving requirement restated from Section 2.2:** regardless of budget tier, teacher-activation caching (computing the frozen teacher's forward pass once per sample, not once per epoch) is mandatory engineering practice here, not a nice-to-have — without it, every budget scenario above would need to multiply its compute estimate by the number of training epochs, which would likely make even the "high" scenario infeasible within 8 weeks.

---

## 18. Risks & Challenges

| Risk | Category | Mitigation |
|---|---|---|
| **Hinglish/code-switched labeled data is scarce.** Public datasets with high-quality code-switched transcripts plus ambient noise plus intent/summary labels, all together, may simply not exist at sufficient scale. | Technical / Data | Phase 1 must include an explicit data-availability audit before committing to a final task/eval design; if labeled data is insufficient, fall back to a smaller eval set with synthetic noise augmentation (Section 2.1, C5) layered onto otherwise-clean existing Hinglish ASR corpora, and report this limitation explicitly rather than overclaiming generality. |
| **The chosen teacher model may not actually be good at code-switched/noisy audio itself**, in which case distilling from it (even perfectly, with zero alignment collapse) would just transfer mediocre capability, confounding the evaluation. | Technical | Phase 1 selection (Section 6) must include a baseline check of the teacher's *own* raw performance on a small held-out Hinglish/noisy sample before committing to it as the distillation target — this should gate teacher selection, not be discovered after Phase 3 is already built around it. |
| **RKD hyperparameters (α, β) may require a wider sweep than the 1-week Phase 3 budget allows**, risking suboptimal weights being locked in before Phase 4's full run. | Technical | The small-scale overfitting sanity check (Section 15, Phase 3 deliverables) is specifically designed to catch gross misconfiguration cheaply before the expensive full-scale sweep, narrowing the search space the real sweep needs to cover. |
| **Alignment-collapse metrics (Section 14.5) might not show a clean, unambiguous difference between the RKD and point-wise-KD students** — i.e., the central hypothesis might simply not hold, or hold only weakly. | Business / Research-validity | This is a legitimate possible research outcome, not purely a "risk to mitigate" — the project's reporting (Phase 4 deliverable) must be structured to honestly report a null or mixed result if that's what the data shows, per the evaluation harness's design (Section 14.5.5 reports all four diagnostics independently rather than collapsing them into one number that could be cherry-picked). |
| **Latency target ("at least 2x speedup over cascade," Section 2.2) may not be met**, e.g., if the projection bridge or LoRA-adapted decoder turns out slower than expected, or if NF4 quantization introduces unexpected runtime overhead on the target hardware. | Technical / Scaling | Benchmark early (end of Phase 2, once the basic pipeline runs even without RKD) rather than only at the very end of Phase 4 — a latency problem discovered in Week 4 is fixable; the same problem discovered in Week 8 is not. |
| **GPU preemption / shared cluster contention** during the multi-day Phase 4 training run. | Scaling / Infra | Checkpoint-on-interrupt handling and the resume procedure (Section 12.4) must be tested in Phase 1, not assumed — this is explicitly called out there for exactly this reason. |
| **Licensing risk** — accidentally training on data without a license permitting model training, discovered late. | Security / Legal | Enforced as a hard CI-gated check (Section 13) at ingestion time, not a manual review step that can be skipped under time pressure. |
| **Single-point-of-failure team risk** — if the project is effectively run by one person (Persona 1), any illness/unavailability directly threatens the 8-week timeline with no redundancy. | Business | Keep the codebase and WandB run history reproducible enough (Section 12, pinned environments, config-driven runs) that a second person could pick up mid-project from documentation alone if needed — this is a direct, practical reason the interface contracts in Section 8 and the config schema in Section 6 are specified as rigorously as they are in this document. |

---

## 19. Repository Structure

```
audiollm-rkd/
├── README.md                      # Quickstart, links to this kickoff doc
├── pyproject.toml                 # Pinned dependencies (Section 11, supply-chain mitigation)
├── setup_env.sh                   # Environment bootstrap (conda/venv + CUDA-matched torch)
├── .github/
│   └── workflows/
│       └── ci.yml                 # Lint + unit/integration tests (Section 12.3)
├── configs/
│   ├── baseline.yaml              # Cascaded-baseline config (Phase 1)
│   ├── rkd_student.yaml           # Main RKD training config (encoder/decoder/teacher choice, α/β)
│   ├── pointwise_kd_baseline.yaml # Required comparison baseline (Section 14.5)
│   └── intent_labels.yaml         # Fixed intent taxonomy (Section 14.2)
├── audiollm_rkd/                  # Main package
│   ├── encoders/
│   │   ├── base.py                # AudioEncoder Protocol (Section 8.2)
│   │   ├── wavlm.py
│   │   └── whisper_encoder.py
│   ├── bridge/
│   │   └── projection_bridge.py   # ProjectionBridge (Section 8.2 / 2.1 C2)
│   ├── decoder/
│   │   └── compact_decoder.py     # LoRA-wrapped decoder loading (Section 6, C3)
│   ├── distillation/
│   │   ├── rkd_loss.py            # RKDLoss: distance + angle penalties (Section 9)
│   │   ├── pointwise_kd_loss.py   # Baseline loss for comparison (Section 14.5)
│   │   ├── hooks.py               # Forward-hook registration utilities (Section 9.6)
│   │   └── alignment.py           # Sequence-length alignment strategies (Section 9.1)
│   ├── data/
│   │   ├── manifest_schema.py     # AudioSample/NoiseProfile/License dataclasses (Section 7)
│   │   ├── loaders.py             # Dataset loading + batching
│   │   └── augmentation.py        # Noise injection (Section 2.1, C5)
│   ├── eval/
│   │   ├── task_metrics.py        # Intent accuracy/F1, ROUGE/BERTScore (Section 14.4)
│   │   └── alignment_diagnostics.py # Effective rank, distance correlation, k-NN agreement,
│   │                                 # stress test (Section 14.5) — the project's core output
│   └── inference/
│       └── audio_llm.py           # Public AudioLLM class (Section 8.1)
├── scripts/
│   ├── smoke_test.py              # Fast tiny-model sanity check (Section 4, first-time journey)
│   ├── cache_teacher_activations.py # One-time teacher caching pass (Section 5.3)
│   ├── train.py                   # Main training entry point
│   ├── evaluate.py                # Evaluation harness entry point (Section 2.1, C6)
│   ├── benchmark_latency.py       # Latency/memory comparison (Section 13, performance tests)
│   └── validate_dataset.py        # Manifest/licensing validation (Section 7, Section 11)
├── tests/
│   ├── unit/                      # Section 13 unit tests
│   ├── integration/                # Section 13 integration tests
│   └── e2e/                        # Section 13 end-to-end tests
├── checkpoints/                   # .gitignored; MANIFEST.json tracks "current best" (Section 4)
└── docs/
    └── kickoff_doc.md             # This document
```

**Folder-purpose rationale:** `distillation/` is kept as its own top-level module (not buried inside a generic `models/` folder) specifically because the RKD engine is the project's actual research contribution (Section 14.1) and deserves to be the most discoverable, most heavily tested (Section 13) part of the codebase — a new contributor (Section 4's first-time journey) should be able to find it immediately.

---

## 20. Development Kickoff Package

### 20.1 Architecture Summary
See Section 5.1 (diagram) and Section 5.3 (data flow) for the complete picture: frozen audio encoder → trainable projection bridge → LoRA-adapted compact decoder, trained against combined cross-entropy + RKD distance/angle losses computed via forward hooks against a frozen, activation-cached teacher.

### 20.2 MVP Scope
The MVP (end of Phase 2, per Section 15) is: **a mechanically correct encoder→bridge→decoder pipeline trainable on cross-entropy loss alone**, with the cascaded baseline (Phase 1) already benchmarked for comparison. RKD is explicitly *not* part of the MVP — it is the Phase 3 addition layered onto a verified-working base, which is the right order of operations for de-risking the project (verify plumbing before adding the novel/risky research component).

### 20.3 Sprint 1 Tasks (Week 1)
1. `setup_env.sh` + pinned `pyproject.toml`; smoke test passing on CI.
2. Benchmark candidate audio encoders (WavLM-Large vs. Whisper-encoder) on a small Hinglish sample for qualitative sanity (Section 6).
3. Begin dataset availability audit (Section 18, first risk) — survey existing Hinglish/code-switched corpora and noise-profile libraries; document findings against the manifest schema (Section 7) feasibility.
4. Stand up the cascaded ASR+LLM baseline pipeline end-to-end (even crudely) to start collecting comparison numbers early.

### 20.4 Sprint 2 Tasks (Week 2)
1. Finalize dataset manifest population for an initial training-sized batch; validation script (`validate_dataset.py`) enforcing the licensing gate (Section 11) and split stratification (Section 7.2).
2. Finalize cascaded-baseline latency/memory/accuracy numbers, logged to WandB as the permanent reference point.
3. Select final compact decoder candidate (sub-2B) based on Hindi-English tokenizer coverage check (Section 6).
4. Select and license-check the teacher model candidate; run the Phase-1-mandated raw-performance gate (Section 18, second risk) before committing to it.

### 20.5 Sprint 3 Tasks (Weeks 3–4, start of Phase 2)
1. Implement `ProjectionBridge` (Section 8.2) with LayerNorm output; unit test against synthetic encoder-shaped tensors.
2. Wire up PEFT LoRA (r=16) on the selected compact decoder (Section 6, C3); confirm `inputs_embeds`-based forward pass accepts the bridge's output shape without error (Section 7.3 tensor contract).
3. Run a CE-only training loop on a small batch; confirm loss decreases (this is the MVP-completion gate from Section 20.2).
4. Begin `cache_teacher_activations.py` implementation ahead of Phase 3's need for it (Section 5.3) — this can start in parallel since it only depends on the teacher selection from Sprint 2, not on the student pipeline being finished.

### 20.6 Initial Feature/Tensor Schema
Already fully specified — see Section 7 (dataset manifest) and Section 7.3 (tensor interface contracts). No separate schema needs to be invented here; Sprint 1–3 work should implement directly against those specifications.

### 20.7 Initial "Endpoints" (Module Interfaces)
Already fully specified — see Section 8.1 (`AudioLLM` public interface) and Section 8.2 (internal `AudioEncoder`/`ProjectionBridge`/`RKDLoss` interfaces). These should be stubbed out (correct signatures, `NotImplementedError` bodies) in Sprint 1 even before they're fully implemented, so that interface-contract tests (Section 13) can be written against them immediately and catch drift as real implementations land in later sprints.

### 20.8 Recommended Milestones

| Milestone | Target | Gate condition |
|---|---|---|
| M1: Baseline established | End of Week 2 | Cascaded baseline metrics logged; dataset manifest populated and passing validation. |
| M2: MVP pipeline verified | End of Week 4 | CE-only training loop runs and loss decreases; integration tests (Section 13) passing. |
| M3: RKD engine validated in isolation | End of Week 6 | `RKDLoss` unit tests passing; small-scale overfitting sanity check shows decreasing, non-NaN RKD losses; point-wise-KD baseline implemented. |
| M4: Final results produced | End of Week 8 | Section 14.5.5 comparison table fully populated for teacher, KD-baseline student, and RKD student; latency/memory benchmark vs. cascade complete. |

### 20.9 Definition of Done

A task/feature in this project is **Done** only when all of the following hold:
1. Code implements the exact interface specified in Section 8 (or the relevant module contract) — no ad-hoc shape/signature deviations.
2. Unit and/or integration tests exist per the Section 13 coverage targets for that module, and pass in CI.
3. Relevant config options (if any) are documented in the corresponding `configs/*.yaml` example file, not left as undocumented code-only parameters.
4. If the task touches the RKD loss or alignment diagnostics specifically (Sections 9, 14.5), the NaN/degeneracy guard (Section 9.5) has an explicit passing test — this is a non-negotiable gate given how central these metrics are to the project's validity, not a generic "nice to have" testing courtesy.
5. Any new metric, checkpoint, or result that will appear in the final report is logged to WandB with a reproducible run config (Section 6) — no figure or number in the final write-up should originate from an unlogged, ad-hoc script run.