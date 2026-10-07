# Local VLM evaluation protocol, 2026-10-04

## Canonical results

- `original_six_action/`: exact original task. A VLM selects one of forward, back, left, right, turn_left, turn_right from a reproduced loss frame and enabled history. Success is looked up only after selection in the SHA256-pinned, previously physically executed 3-second branch table. This is **offline one-step selection**, not a newly executed closed-loop rollout. 23 eligible worlds / 15 construction groups; all original eligibility labels and loss times reproduced. 4B: five conditions, 115 decisions. 9B planned capacity check: current_direct and images_actions_reasoned, 46 decisions.
- `closed_loop_v2/`: 4B actual online MuJoCo pilot on stairs_up, trench, pit, hist001_left, five conditions, up to ten 3-second motions (20 episodes). Model decisions control physics and the next camera observation. Goal prompt includes reacquisition, approach and STOP. These four selected worlds are not a representative population estimate.
- `closed_loop_9b/`: corresponding actual online 9B capacity pilot on stairs_up and hist001_left, current_direct and images_actions_reasoned (four episodes), if runtime succeeds. Completed result files and final report establish actual completion, not this plan.

One-step and online tasks have different goals and action spaces. Never mix their success tables. Online STOP ends the episode; no search fallback is injected. Low-level motion commands never steer using target coordinates. Camera is runtime-injected rendered RGB on the simulated robot. Evaluator state, scene names, target coordinates, detector output and future branch labels are excluded from model requests.

## Input and inference controls

Five modes: current image/direct answer; current image/reasoning; current plus commands/reasoning; current plus image history/reasoning; current plus both histories/reasoning. Up to three uniformly sampled chronological history frames. Commands describe requested actions and duration, not measured pose.

Native multi-image requests failed simple image-identity/order checks. Final image-history conditions therefore use a labeled contact sheet containing the same source images, with HISTORY and CURRENT labels. Three synthetic paired checks passed. This does not guarantee perfect temporal understanding on real scenes. Larger composed input and model resizing remain a confound to diagnose before claiming history is harmful.

Reasoning uses a fixed 512-token bounded deliberation call, followed by a non-thinking 180-token structured-action call using that analysis and the same image input. Direct uses one 180-token call. Temperature 0, nominal seed 7, context 8192, Q4_K_M models. This tests this bounded reasoning protocol, not unlimited model reasoning. There is no new training, proposed memory algorithm or world model. Input files and hashes, prompts, response text, model timing and raw traces are retained.

## Evaluation

Original camera criterion: at least 3 red target pixels in 3 consecutive 10 Hz samples before falling/out-of-bounds. Qualifying loss: zero pixels in 3 consecutive samples after seen. The 24th world perlin_rough never qualifies and is excluded identically across methods. Original fixed-forward is 17/23; all 16 paired window worlds are solved by forward.

Online reports separately: (1) ever reacquired before failure, (2) ever within planar 1 m, exploratory, and (3) STOP while target visible, planar range <=1 m, target visual depth <=1.25 m and robot upright. The third is supplemental stop-near-marker evidence; neither distance metric certifies traversability or arrival on the same support surface. Inference pauses simulation. Wall-clock request latency can include queueing; raw model service timing is logged. No real-time or hardware claim.

## Diagnostic archives and provenance

Root `episodes/` contains an early thinking-format pilot. `single_step_v1/` and `single_step_v2/` are mixed approach/STOP one-step pilots, excluded from the exact original-task table. `closed_loop_v1/` uses native multi-image inputs; history-image episodes are diagnostic only. Ten completed non-image-history online episodes were copied byte-for-byte to v2 because their requests and physics were unchanged; see `closed_loop_v2/REUSED_EPISODES.json`.

`closed_loop_v2/PROVENANCE_CORRECTION.json` annotates one episode whose client hash was read from disk after a source edit while the process retained the earlier imported class. Process-start source snapshots establish the loaded version. Raw outcomes were not edited. New runner captures hashes at run start.

No population p-values, independent-frame sample-size claims, global SOTA label or new-method superiority follow from these designed, deterministic, single-seed pilot tests.
