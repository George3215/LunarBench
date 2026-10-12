# Task3 全地形六条件实验

完成 186/186；记录审计通过：True。

|传感器|上下文|完成|成功|资源终止|
|---|---|---:|---:|---:|
|rgb|current|31|1|0|
|rgb|first_current|31|0|0|
|rgb|all_history|31|0|0|
|rgbd|current|31|0|0|
|rgbd|first_current|31|18|0|
|rgbd|all_history|31|0|0|

上述比例是实际完成任务的描述性比例；资源终止不能解释为模型导航能力失败。

- One deterministic trial per scene and condition; descriptive paired comparison, no statistical reliability claim.
- 31 scenarios cover 15 terrain families plus 16 window parameter variants; variants are not independent terrain types. Adjusted window targets are on the near side, so this is not the original H2 benchmark.
- Policy evidence citations are public self-reports, not causal explanations of internal computation.
- MuJoCo Go2 physical state; no physical hardware dog was connected.
- All history means every saved action-boundary observation; evaluator 10Hz frames are not model inputs.
- Explicit runtime context/GPU limits are reported separately and cannot establish VLM navigation inability.