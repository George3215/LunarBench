# Completed run, 2026-10-04

- Exact original six-action task: 161/161 selections complete (4B: 23 worlds x 5 modes; 9B: 23 x 2 modes), zero invalid action outputs.
- Actual online closed loop: 24/24 episodes complete (4B: four worlds x five modes; 9B: two worlds x two modes). STOP-near-marker criterion: 0/24; reacquisition and exploratory proximity are reported separately.
- Both models ran locally on RTX 2080 Ti. Model/runtime records and exact digests retained.
- Perception check: 4B 46/46; 9B 45/46 (missed barrier seen frame). Both passed three simple labeled-sheet identity cases.
- All 24 prefix eligibility labels and 23 eligible loss times reproduce original data. All 161 selected branch/image hashes verified. 114 online model-input hashes and 19 physically executed first steps verified; trajectory coordinate differences <5e-5 m versus rounded old logs.

Canonical paths: original_six_action/, closed_loop_v2/, closed_loop_9b/. single_step_v1/v2, root episodes/, and historical-image closed_loop_v1 are diagnostic archives, not the original benchmark table. See EXECUTION_PROTOCOL.md and ../../VLM_BASELINE_RESULTS.md.

No new algorithm, training, real hardware, global SOTA result or full traversability-certified arrival claim. All experiment processes completed; the reproducible local Ollama server can remain available, with model residency released after reporting.
