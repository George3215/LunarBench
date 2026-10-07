# Local VLM benchmark results

Only completed episodes are included. Read protocol and limitations before comparison.

| Model | Input condition | Episodes | Reacquired | Stop near marker | Mean decision s |
|---|---|---:|---:|---:|---:|
| qwen3.5:4b | actions_reasoned | 1 | 0 (0.000) | 0 | 6.76 |
| qwen3.5:4b | current_direct | 2 | 0 (0.000) | 0 | 0.89 |
| qwen3.5:4b | current_reasoned | 1 | 0 (0.000) | 0 | 6.30 |
| qwen3.5:4b | images_actions_reasoned | 1 | 0 (0.000) | 0 | 6.55 |
| qwen3.5:4b | images_reasoned | 1 | 0 (0.000) | 0 | 6.83 |

Reacquisition uses actual camera frames during executed motion. Arrival is supplemental (STOP + visible target + distance/depth thresholds), not validated full navigation success. These are simulation runs, one model seed, a designed scene suite, and quantized local models; they do not establish global SOTA or universal task difficulty.
