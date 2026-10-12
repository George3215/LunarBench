"""TASK3: terrain-induced target re-observability.

Target observability is a non-monotonic function of robot motion: walking toward
a target can hide it behind terrain.  The research order is H1 non-monotonicity,
then H2 history necessity, then H3 counterfactual planning.  Entry points, all
runnable as scripts from the repository root:

``benchmark.py``
    Experiment 2: V(x) along each approach, the physical visibility curve
    V(a, t) of every candidate action, RSR / T_reobs / motion cost, and the
    A / B / D / C decision rules.
``history.py``
    Experiment 1: 8 paired worlds whose lost-state camera images are pixel
    identical but whose optimal re-observation action is opposite.
``history_pairs.py``
    Searches for and geometrically validates those paired worlds against seven
    pre-registered criteria before any physics is run.
``visibility_model.py``
    Experiment 3: trains a predictor of action-conditioned re-observability on
    the recorded rollouts and compares feature sets, including the control whose
    memory features are permuted.
``experiment.py``
    Experiment 0: geometric counterfactual oracle plus the injected RGB-D camera
    and a contact-surface consistency check.
``locomotion.py``
    Single-scene motion test with a vendored policy selected by ``--controller``.
``slabs.py``
    Exact box slab terrain kit (MuJoCo's height-field collider stalls the
    vendored locomotion policies).
``api.py``
    The **frozen benchmark API**: ``Environment`` -> ``Observation`` ->
    ``ReObservationPolicy`` -> ``action_id``.  Terrain code never sees a policy.
``policies.py``
    The baselines (Reactive, HeadingLock, HistoryOnly, LostTargetSearch,
    ObjectPermanence, NBV, Learned, Oracle) behind that one interface.
``episode.py``
    The **multi-step episode protocol**: protocol A (open loop) and B (closed
    loop), one shared budget, declared termination conditions, phase tracking and
    the failure taxonomy.
``benchmark_api.py``
    Runs every policy on every terrain through the frozen API and the episode
    protocol.
``test_api.py``
    Contract tests that keep the API frozen and the terrain policy-agnostic.
``terrain_viewer.py``
    ``--list`` prints every terrain, ``--check`` self-tests the terrain and the
    robot on it headlessly, and the default opens an interactive MuJoCo window
    where the robot can be driven with the keyboard.
``figure*.py``
    The six figures referenced from the TASK3 README.
``realsense.py``
    Standalone real RealSense probe; never moves the robot.
"""
