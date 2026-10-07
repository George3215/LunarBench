# Models and hardware feasibility, 2026-10-04

Measured host: NVIDIA RTX 2080 Ti 11 GiB (Turing, compute capability 7.5), about 64 GiB system RAM, Xeon W-2223 / 8 logical CPUs. Existing Ollama 0.32.15 and PyTorch 2.11.0+cu128. Hardware and exact model digests are recorded in runs/20261004_qwen35/hardware_manifest.json.

| Candidate | Status / next action | Evidence boundary |
|---|---|---|
| Qwen3.5-4B, Q4_K_M (GGUF reports 4.7B total) | Completed 115 original-task selections and 20 online episodes; image intervention check passed | A current-generation compact multimodal baseline, not a claim to be the world's best VLN policy |
| Qwen3.5-9B, Q4_K_M (about 6.6 GB file) | Completed real image inference, 46 original-task selections and 4 online episodes on this GPU | Runtime verified: initial post-inference whole-GPU sample 7839 MiB, later concurrent rendering sample 8007 MiB; not a measured peak. See runtime_qwen3.5_9b.json |
| Qwen3-VL-4B/8B | Official Ollama family verified; alternative if Qwen3.5 runtime unsuitable | Not downloaded for this first run because Qwen3.5 is already running |
| NaVILA-8B | Most directly relevant legged VLA candidate; defer full deployment while testing local VLMs | Native 8B FP16 language weights alone are about 16 GB, above 11 GB VRAM; CPU offload or quantization may permit inference but have not been validated. Original low-level visual locomotion uses LiDAR/heightmap and separate policy; adapting to current Go2 controller is not a faithful original-system reproduction |
| WoMAP | Closest world-model planning reference; record for a later trained/adapted comparison | Requires its own representation, model and task-specific training/interface; no zero-shot drop-in claim. Not demonstrated to be blocked solely by VRAM |
| Large 27B+ multimodal models | Not attempted on this GPU in this turn | Typical 4-bit weight storage alone is around 13.5 GB for 27B before overhead; CPU offload remains possible but slower and unmeasured |

Official model listings checked: https://ollama.com/library/qwen3.5:4b and https://ollama.com/library/qwen3.5:9b . Exact card text saved with the run. Relevant paper sources and qualifications are in RELATED_WORK.md. This is a feasibility shortlist, not an exhaustive 2026 VLN/VLA leaderboard review.

All inference is local. No paid remote model API and no real robot are involved. Model and Ollama state needed for this experiment are stored under D:/LunarBench/.local/vlm/.
