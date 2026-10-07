# Local VLM benchmark results

Only completed episodes are included. Read protocol and limitations before comparison.

| Model | Input condition | Episodes | Reacquired | Stop near marker | Mean decision s |
|---|---|---:|---:|---:|---:|
| qwen3.5:4b | actions_reasoned | 4 | 2 (0.500) | 0 | 9.43 |
| qwen3.5:4b | current_direct | 4 | 3 (0.750) | 0 | 1.87 |
| qwen3.5:4b | current_reasoned | 4 | 4 (1.000) | 0 | 10.88 |
| qwen3.5:4b | images_actions_reasoned | 4 | 1 (0.250) | 0 | 11.88 |
| qwen3.5:4b | images_reasoned | 4 | 2 (0.500) | 0 | 11.93 |

Reacquisition uses actual camera frames during executed motion. Arrival is supplemental (STOP + visible target + distance/depth thresholds), not validated full navigation success. These are simulation runs, one model seed, a designed scene suite, and quantized local models; they do not establish global SOTA or universal task difficulty.
