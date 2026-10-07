# Exact original six-action VLM test

Offline selection from previously physically executed branches. One nominal model seed. Descriptive counts only; worlds sharing construction groups are not independent. No new online navigation claim.

| Model | Mode | Success | Base worlds | Window worlds | Invalid | Median decision s |
|---|---|---:|---:|---:|---:|---:|
| qwen3.5:4b | actions_reasoned | 13/23 | 1/7 | 12/16 | 0 | 6.85 |
| qwen3.5:4b | current_direct | 17/23 | 1/7 | 16/16 | 0 | 0.77 |
| qwen3.5:4b | current_reasoned | 15/23 | 1/7 | 14/16 | 0 | 6.68 |
| qwen3.5:4b | images_actions_reasoned | 5/23 | 2/7 | 3/16 | 0 | 6.94 |
| qwen3.5:4b | images_reasoned | 12/23 | 1/7 | 11/16 | 0 | 6.91 |
| qwen3.5:9b | current_direct | 5/23 | 0/7 | 5/16 | 0 | 2.32 |
| qwen3.5:9b | images_actions_reasoned | 3/23 | 2/7 | 1/16 | 0 | 12.56 |

Fixed forward: 17/23. Uniform random expectation: 0.3623.

See summary.json for action distributions, paired outcomes, raw record paths and integrity verification.
