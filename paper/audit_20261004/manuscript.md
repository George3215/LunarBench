# Camera-Defined Target Reacquisition on Legged Robots

## Observation, Memory, and Motion

Anonymous research draft · 4 October 2026
## Abstract

A robot that loses a previously detected target must choose motions that restore useful visual evidence. Evaluating this capability requires separating geometric visibility, image detection, and the consequences of physical execution. We audit an existing quadruped simulation benchmark and find that its principal experiments used geometric success labels and ground-truth target memory while being described as non-privileged visual reacquisition. A reported comparison also mixed an eight-world random baseline with a 24-world learned planner. On the common eight-world subset, both historical success rates are 0.625. Grouping paired constructions together changes the legacy action-conditioned prediction AUC from 0.934 to 0.930, preserving predictive information in the proxy task without establishing a visual control result. We then collect a camera-defined supplement with an injected body-mounted RGB-D camera and six target-independent locomotion primitives. Of 24 candidate worlds, 23 satisfy a physically executed seen–lost precondition, yielding 138 executed recovery branches. Geometric and image-detection labels agree on 91.67% of sampled branch frames. Exact uniform-random one-step success is 0.362, fixed forward motion achieves 0.739, and a group-held-out image-summary scorer averages 0.725 across three fitting seeds. Its paired difference from forward motion has a descriptive 95% group-bootstrap interval of [-0.050, 0.000]. Adding last-detection memory achieves 0.725; the paired memory-added difference is 0.000 with interval [0.000, 0.000]. These exploratory results support studying sensor-grounded reacquisition, while limiting claims about memory necessity, learned planning superiority, and generalization.

**Keywords:** visual reacquisition; active perception; legged locomotion; simulation; benchmark validity; partial observability.

## Introduction

Target loss is a practical interruption of visual feedback. A mobile robot can know that it previously saw an object yet lack an immediately useful observation for tracking, inspection, or manipulation. Terrain, camera orientation, self-occlusion, target scale, and the detector can all contribute to that interruption. The next useful action need not move directly toward the last image bearing. However, neither that observation nor the use of history is by itself a new robotics problem.

The research question examined here is narrower: *when a legged robot loses a target according to its camera, which evidence supports choosing a locomotion primitive that restores a usable detection?* Three quantities matter: a geometric line-of-sight label, an image-derived detection event, and the executed motion that changes the camera pose. Substituting one quantity for another changes the scientific task even when software interfaces use similar names such as “visible” or “memory.”

We use an existing Go2+Piper MuJoCo benchmark as a case study. Its archived experiments offer substantial implementation and diagnostic material, including action rollouts, paired-world constructions, and consequence predictors. A source-to-result audit nevertheless reveals that the principal planner consumes exact simulated target coordinates, while the evaluation labels come from an analytic visibility function. A camera exists in the model but is disabled in the default environment. These facts invalidate the description of those results as non-privileged visual recovery. The audit also found unreset internal history in the exported locomotion controller, which can couple otherwise separate counterfactual branches.

This paper contributes an evidence audit, a corrected camera-defined measurement protocol, and a small controlled supplement. The supplement uses physically generated loss states and evaluates every recovery action from an identical saved simulator state. It intentionally includes strong simple behavior, exact random-action expectations, memory ablations, and excluded worlds. We do not propose a general world model or claim a new theory of active perception. The study is exploratory and simulation-only; its purpose is to establish what the available evidence can support before scaling methods or making deployment claims.

## Related Work and Scope

Active perception treats sensing and action as coupled processes [1]. Planning under partial observability provides a classical basis for retaining information from history [2]. The existence of visually ambiguous states is therefore motivation for a controlled task, rather than evidence of conceptual novelty.

Lost-target search is directly relevant. Radmard and Croft explicitly address loss of a target from camera feedback, use information available before disappearance, and plan online subject to system constraints [3]. Occlusion resolution in active visual search is also an established topic [4]. An implementation that always moves toward the remembered bearing is a useful heuristic, but it is not a reproduction of these complete methods. Poor performance of that heuristic cannot establish the failure of the literature it is named after.

Next-best-view planning likewise cannot be reduced to unconstrained viewpoint scoring. OA-NBV integrates target-centric visibility, target scale and completeness, and traversable pose selection under motion constraints [5]. We checked its full HTML text and therefore reject a blanket distinction in which earlier NBV methods consider only where to see and our task uniquely considers whether a robot can get there. The practical issue studied here is whether a particular geometry proxy matches the camera and whether the chosen locomotion primitive produces the required visual evidence.

WoMAP already combines learned action-conditioned prediction and active object localization [6]. Our small random-forest action scorer is not a reproduction of WoMAP and is not presented as a world model. The literature supports the importance of restoring target observations, while the present evidence is limited to a particular morphology, marker detector, terrain suite, and primitive set. A claim of algorithmic novelty or superiority to those systems would require direct, faithful comparisons that this study does not supply.

## Problem and Measurement Protocol

### Camera-defined events

Let $I_t$ and $D_t$ denote a rendered RGB image and depth image. For this controlled experiment, the target is a unique red sphere, and the detector counts pixels satisfying

$$

R>50,\qquad R>1.5G,\qquad R>1.5B.

$$

The primary detection event is $d_t=\mathbb{1}[n_t\geq3]$, where $n_t$ is the count at $320\times240$ resolution. Detection is sampled at 10 Hz. A world becomes eligible only after at least three consecutive detections followed by three consecutive frames with zero target-color pixels during a dynamically executed approach. Thus “lost” means absence under the stated color detector; it does not prove complete semantic or geometric occlusion.

The approach starts at the scene's historical or approach location. A fixed body-frame command points toward the nominal loss anchor, with total commanded translational speed capped at $0.6$ and a yaw-hold controller, for at most 12 simulated seconds. This is a scenario-initialization program shared by all recovery methods, not a learned recovery policy. Ground-truth scenario anchors are allowed in initialization and are not provided to the recovery scorer. The first eligible loss snapshot is frozen. Every candidate world is retained in the eligibility ledger, including those with no qualifying loss.

### Actions and success

The recovery action set contains forward, backward, left strafe, right strafe, left turn, and right turn. Each action has a three-second budget and uses the existing pretrained locomotion controller without retraining. Forward and backward command magnitudes are $1.0$ and $0.8$; lateral magnitude is $0.6$; turn commands are $+0.75$ and $-0.65$. These are controller commands, not guarantees of achieved velocity or rotation. In particular, the asymmetric turn commands reflect the existing controller configuration, not symmetry of physical outcomes.

We exclude the legacy target-steering primitive: a primitive that directly steers toward the simulator's target position embeds privileged information in the action interface. Heading stabilization itself need not be privileged, but the legacy multi-step implementation must first be corrected and revalidated before being used as an independent baseline.

For action $a$ in world $i$, $Y_i(a)=1$ if its executed branch contains three consecutive camera detections while the upright component is at least 0.25 and the robot remains inside the terrain boundary with a 0.3 m margin. A physical termination before a valid streak is a failure. The internal observation history of the exported locomotion policy and its outer last-action vector are both reset before each primitive. This common controller initialization prevents one candidate branch from affecting another; it is an explicit primitive convention, not uninterrupted hidden-state continuation from the approach. This is a first-reacquisition criterion; it does not require that visibility persist through the remainder of the primitive. We record the full branch and its motion cost, but do not compare post-success full-branch costs as if they were costs to first detection.

For a policy choosing one action at the loss state,

$$

\widehat{\mathrm{RSR}}_1(\pi)=\frac{1}{N}\sum_{i=1}^{N}Y_i(\pi(o_i,m_i)).

$$

This is a single-step measure. It must not be substituted for the archived ten-step episode success rate. The exhaustive branch table gives an exact uniform-random expectation

$$

\widehat{\mathrm{RSR}}_1(\mathrm{random})=\frac{1}{6N}\sum_i\sum_aY_i(a),

$$

and a privileged finite-action upper bound $N^{-1}\sum_i\max_aY_i(a)$. Computing those values requires evaluator rollouts; a deployable scorer does not receive the branch outcomes at decision time.

### Information boundary

The camera-summary input contains a $4\times4$ depth grid, clipped to 20 m, and three RGB channel means. The memory input contains the last detected image centroid, median target-pixel depth, and elapsed time since that detection. This is a small image-derived memory, not a reconstructed world-coordinate target location. Each candidate has the same six-dimensional action one-hot encoding. No recovery model receives absolute robot pose, terrain labels, true target coordinates, geometric visibility, or executed future outcomes as an input.

The image representation is deliberately simple. It does not establish performance of an image transformer or a general target detector. The target is static and uniquely colored, the camera is rendered without real sensor artifacts, and segmentation is used only for evaluator spot checks. These limits are part of the task specification.

## Audit of Archived Evidence

### Task and information mismatch

The archived environment exposes optional RGB-D fields but defaults to disabled rendering. Its episode success variable is computed by a privileged analytic visibility function. The target-memory constructor writes the simulator target coordinates directly, and the historical approach is reconstructed by analytic interpolation. As a result, target-memory availability does not prove that an earlier camera image contained a detection.

The analytic visibility model also differs from the rendered sensor: its default horizontal and vertical fields of view are $87^\circ$ and $58^\circ$, whereas the rendered camera has a vertical field of view of $87^\circ$. The analytic eye omits the camera's 0.31 m forward mount offset and does not model the complete dynamic camera orientation. Therefore, even the geometric label is an approximate proxy for the rendered camera, rather than an interchangeable sensor ground truth.

The exported Go2RL TorchScript model has its own observation-history tensor and reset method, whereas the wrapper previously cleared only the last action. Restoring physical state therefore did not reset the complete policy state, so candidate order could influence action outcomes. We now reset both states. A direct same-snapshot test with and without camera sampling produces identical final generalized coordinates and path cost after this correction. Camera transforms are computed in a private simulator-data copy, so rendering cannot recompute dynamics in the live state.

A further implementation defect concerns quaternion indexing. The archived API computes its yaw numerator using $wx+yz$ rather than $wz+xy$. This affects reported poses and frame transformations. We corrected the formula and checked known rotations against the rollout convention. Historical JSON artifacts remain unchanged. Claims based on their pose errors require regenerated data; averaging translational errors in metres with yaw errors in radians also cannot produce a meaningful error in metres.

### Denominators and sampling units

The historical random-baseline artifact contains eight base worlds and three random seeds. The learned-planner artifact contains 24 worlds: those eight plus 16 members of eight paired constructions. Comparing 0.625 against 0.792 across those sets is invalid. On the common eight worlds, the learned planner succeeds in five, while the historical random mean is also 0.625. The random artifact lacks the paired per-world outcomes needed for a proper paired inference, so we report equality of the descriptive means without inventing a significance test.

The consequence dataset contains 2240 state–action rows but only 224 states and 16 construction groups. Splitting by scene name holds out one member of a pair while retaining its companion. To diagnose the magnitude of that effect, we retain the archived data and classifier configuration and repeat classification with both pair members held out together. This is explicitly a reanalysis of the old proxy task, including its state-feature defects, possible branch-history carryover, and privileged memory, rather than a corrected visual experiment.

**Table 1. Archived proxy labels; not camera-control results.**

| Split | Inputs | AUC | AP |
|---|---|---:|---:|
| Scene | O+M | 0.854 | 0.600 |
| Scene | O+M+a | 0.934 | 0.748 |
| Construction group | O+M | 0.850 | 0.596 |
| Construction group | O+M+a | 0.930 | 0.743 |
| Construction group | a | 0.448 | 0.139 |

The modest change in Table 1 does not support attributing the entire predictive gain to paired-scene overlap. Action conditioning remains useful for the archived label. It also does not repair the privileged input or turn AUC into evidence of camera-based control. Among 224 archived low-resolution frames, 89 already meet the three-pixel detector threshold, further showing that the classification dataset is not exclusively a collection of currently lost targets.

### What constructive examples establish

Identical current observations with disjoint successful one-step action sets establish a limitation of that particular current-observation decision rule under the stipulated action set. They do not prove that all sequential memoryless search procedures are impossible. Likewise, different optimal actions under identical target memories do not imply that the successful sets are disjoint. These distinctions prevent the paired-world constructions from being promoted to universal necessity claims.

## Supplementary Experiment Design

### Simulation and eligibility

We use the vendored Go2+Piper robot model, the existing Go2RL locomotion weights, and the repository's terrain construction. The camera is injected at runtime on the body; it is not an upstream native sensor or a physical camera. All movement described in this paper is simulated rigid-body dynamics. Gravity is $9.81$ m/s$^2$ and the simulation time step is 0.002 s. The repository name does not imply validation under lunar gravity or on a lunar surface.

A two-world low-resolution pilot found that distant targets occupied only one or two red pixels at $160\times120$. We retained that pilot and used $320\times240$ for the complete supplement. Visual inspection of a second pilot showed dark target fragments below its brightness threshold. The final protocol therefore lowers the red-channel threshold from 120 to 50 and requires three zero-pixel loss frames. Earlier pilot data and a stopped run with the old controller reset are retained separately and excluded from final results. The higher-resolution protocol was saved before its full acquisition, but this is local protocol fixation after exploration, not independent preregistration. We attempted all 24 candidate worlds without tuning scene-specific thresholds or selecting worlds for method performance. The exclusions are `perlin_rough`. They do not count as recovery failures because they did not instantiate the declared seen–lost task within the approach budget, but the eligibility rate is reported alongside conditional recovery.

### Baselines and learned scorers

We evaluate all six fixed actions, exact uniform random, and a simple last-image-bearing strafe heuristic. The latter chooses left or right from the last horizontal image centroid. It is labeled as a heuristic, not as a reproduction of a published lost-target search algorithm. Fixed forward is a particularly important reference because camera-triggered loss may occur before a revealing motion is completed.

Four fitted conditions use current RGB-D summaries plus action, last-detection memory plus action, their combination, and a training-memory permutation control. Each is a random forest with 200 trees, depth at most four, and minimum leaf size two. Fitting seeds are 7, 19, and 37. We perform no hyperparameter search. In the permutation control, memory vectors are permuted by world within the training fold; test memory is unmodified. All six action rows from a state remain together, and paired constructions remain together across train and test.

The analysis evaluates held-out action scores against the exhaustive executed branch table. It is not an online ten-step deployment of the fitted policy. Model seeds measure fitting variability on the same physical trajectories, not independent physical replications. Per-world model success is averaged over seeds before group bootstrap.

### Uncertainty and sensitivity

We resample the 15 eligible construction groups with replacement 10000 times, retaining all worlds in a sampled group. Intervals are descriptive percentile intervals for this designed suite; the worlds are not a random sample of real environments. Degenerate bootstrap intervals at an empirical zero or one do not imply population certainty. We do not interpret thousands of correlated frames as thousands of independent trials.

Two action-space comparisons reuse the same recorded branches. The restricted set contains forward, left, and right; the full set adds backward and both turns. We also recompute detection success at one, three, and ten red pixels, conditional on the primary zero-pixel loss snapshots. This sensitivity analysis does not regenerate the loss state under each alternative threshold and is labeled accordingly.

## Results

### Camera and geometry diverge

The complete supplement contains 23 eligible worlds in 15 construction groups and 138 executed recovery branches. Among 4140 sampled recovery frames, geometry and camera agree in 91.67% of cases. The confusion counts are shown in Table 2. At 0 of the 23 camera-triggered loss states, the geometry proxy still reports visibility. The discrepancy can involve scale, image threshold, camera orientation, self-occlusion, and the proxy mismatch; it is not evidence that every loss is caused by terrain occlusion.

**Table 2. Paired geometry/camera frame labels.**

| | Camera absent | Camera detected |
|---|---:|---:|
| Geometry hidden | 3131 | 318 |
| Geometry visible | 27 | 664 |

### Action selection and memory

Table 3 reports conditional one-step recovery. Fixed forward already achieves 0.739, compared with 0.362 for exact uniform random. The current-image scorer averages 0.725, but its paired difference from fixed forward is -0.014 with interval [-0.050, 0.000]. The descriptive interval does not establish an advantage over this fixed action on the small designed suite. Reporting only improvement over random would conceal that stronger reference.

**Table 3. Camera-defined one-step recovery.**

| Method | RSR1 | Group bootstrap 95% interval |
|---|---:|---|
| Uniform random | 0.362 | [0.300, 0.413] |
| Fixed forward | 0.739 | [0.500, 0.920] |
| Last-bearing strafe | 0.000 | [0.000, 0.000] |
| Image + action | 0.725 | [0.491, 0.889] |
| Memory + action | 0.739 | [0.500, 0.920] |
| Image + memory + action | 0.725 | [0.491, 0.889] |
| Shuffled training memory | 0.710 | [0.456, 0.889] |
| Branch upper bound | 1.000 | [1.000, 1.000] |

Adding last-detection memory gives 0.725, versus 0.725 for current image summaries. The memory-added difference is 0.000, with interval [0.000, 0.000]. The descriptive interval does not establish a positive benefit of this memory representation. The result does not establish necessity or sufficiency of all possible history representations. The permutation control reaches 0.710. These results are more informative than imposing a narrative in which every additional information source must help.

The physically generated camera-loss states differ from the old manually specified loss anchors. Their action outcomes must therefore be measured again rather than inherited from the old paired-world tables. This does not refute a mathematical ambiguity at a deliberately fixed common observation. It shows that the manner of constructing the decision state is part of the evaluated task, and that those constructions should not be treated as interchangeable populations.

### Action set and threshold

The finite-action upper bound is 1.000 with six primitives and 0.783 with forward/left/right only. Thus adding backward and rotation makes some recorded loss states recoverable within the three-second horizon. The restricted-set random expectation is 0.493; a smaller action set can have a higher random expectation even while it has a lower best-action bound. This separates action availability from the quality of selecting an action.

With primary loss states fixed, the uniform-random expectation is 0.406 at one pixel, 0.362 at three pixels, and 0.261 at ten pixels. The corresponding upper bound at ten pixels is 0.826. The sizeable sensitivity confirms that a camera-defined success rate needs an explicit image resolution and detection rule. A future detector study should regenerate loss events for each threshold and evaluate real object identity, rather than using these conditional numbers as sensor-robustness certification.

## Discussion and Limitations

The audit establishes that the original research intent and the implemented proxy had diverged. The mismatch is not resolved by adding a rendered image to a figure or by training a vision encoder against the same privileged labels. Correcting the evidence requires a sensor-derived event definition, an explicit information boundary, and method comparisons on identical states and actions.

The new experiment supports the feasibility of measuring visual reacquisition during legged dynamics, while exposing the dependence of the measured result on scenario initialization, sensing thresholds, controller-state reset, and available actions. Improving average predictive metrics is therefore insufficient to establish a need for sophisticated planning. The proper next test is a locked collection of independently generated scenes with specified occlusion mechanisms and authentic approach histories, rather than selecting additional examples because one method wins on them.

Several limits remain. First, this is one morphology and one third-party controller, with deterministic scene setups and no independent physical disturbance replications. Second, only one loss state per eligible world is evaluated. Third, the detector recognizes a unique colored marker and lacks distractors, open-world appearance, real illumination, depth noise, and motion blur. Fourth, the one-step branch evaluator does not measure accumulated state-estimation errors or exploration in a multi-step camera feedback loop. Fifth, exclusion changes the conditional population and must remain visible to the reader. Sixth, compact depth and RGB summaries may underuse image information; a negative memory result concerns these features and this model, not all possible representations.

Finally, comparisons to full lost-target search, reachable-view planning, or world-model systems require faithful implementations, matched sensing, and comparable compute budgets. Their absence limits this paper to a controlled exploratory study and evaluation audit. No real robot was connected or moved, and no simulation result is described as hardware perception.

## Conclusion

A scientifically reasonable target-reacquisition question requires evidence that the target was observed, lost, and restored in the camera used by the task. The audited proxy experiments did not provide that chain and also contained a mismatched baseline denominator. The corrected supplement provides camera-derived events and matched physical action branches. It reports fixed actions, exact random expectations, and group-held-out scorers together, rather than treating the strongest fitted score as proof of a universal need for learning or memory. The defensible outcome is a reproducible measurement protocol and a bounded empirical result, with further algorithmic and deployment claims reserved for independent experiments.

## Reproducibility and Disclosure

Scripts, complete per-frame trajectories, saved camera images, input and asset SHA-256 hashes, fitting seeds, software versions, exclusions, and group-held-out predictions accompany this draft in the local `paper/audit_20261004` directory. The acquisition and analysis outputs are separate from archived artifacts. Tests cover yaw conventions, camera threshold logic, event continuity, catalogue size, and rendered target checks. The manuscript tables are generated from the analysis JSON. Author names, affiliations, funding, and venue-specific declarations remain for the actual authors; no institutional approval or public release is implied.

## References

1.  R. Bajcsy, “Active perception,” *Proceedings of the IEEE*, vol. 76, pp. 966–1005, 1988. doi: [10.1109/5.5968](https://doi.org/10.1109/5.5968).

2.  L. P. Kaelbling, M. L. Littman, and A. R. Cassandra, “Planning and acting in partially observable stochastic domains,” *Artificial Intelligence*, vol. 101, pp. 99–134, 1998. doi: [10.1016/S0004-3702(98)00023-X](https://doi.org/10.1016/S0004-3702(98)00023-X).

3.  S. Radmard and E. A. Croft, “Active target search for high dimensional robotic systems,” *Autonomous Robots*, vol. 41, pp. 163–180, 2017 (online 2015). doi: [10.1007/s10514-015-9539-8](https://doi.org/10.1007/s10514-015-9539-8).

4.  S. Radmard, D. Meger, J. J. Little, and E. A. Croft, “Resolving occlusion in active visual target search of high-dimensional robotic systems,” *IEEE Transactions on Robotics*, vol. 34, pp. 616–629, 2018. doi: [10.1109/TRO.2018.2796577](https://doi.org/10.1109/TRO.2018.2796577).

5.  B. Hu et al., “OA-NBV: Occlusion-Aware Next-Best-View Planning for Human-Centered Active Perception on Mobile Robots,” arXiv:2603.11072v2, 2026. [https://arxiv.org/abs/2603.11072](https://arxiv.org/abs/2603.11072).

6.  T. Yin et al., “WoMAP: World Models For Embodied Open-Vocabulary Object Localization,” *Proceedings of the 9th Conference on Robot Learning*, PMLR, vol. 305, pp. 3605–3630, 2025. [https://proceedings.mlr.press/v305/yin25b.html](https://proceedings.mlr.press/v305/yin25b.html).
