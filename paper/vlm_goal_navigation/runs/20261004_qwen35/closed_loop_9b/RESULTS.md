# Local VLM benchmark results

Only completed episodes are included. Read protocol and limitations before comparison.

| Model | Input condition | Episodes | Reacquired | Stop near marker | Mean decision s |
|---|---|---:|---:|---:|---:|
| qwen3.5:9b | current_direct | 2 | 1 (0.500) | 0 | 3.90 |
| qwen3.5:9b | images_actions_reasoned | 2 | 2 (1.000) | 0 | 16.73 |

Reacquisition uses actual camera frames during executed motion. Arrival is supplemental (STOP + visible target + distance/depth thresholds), not validated full navigation success. These are simulation runs, one model seed, a designed scene suite, and quantized local models; they do not establish global SOTA or universal task difficulty.
