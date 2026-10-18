"""V-A / V-B: does vision carry information the state does not, and does it need
history and odometry?

The capability ladder that motivated this should **not** compare an MLP on state
against a ViT on images: those differ in architecture *and* in input, so any gap is
unattributable. This experiment therefore controls the input and the model:

  * one downstream classifier, one split, one target;
  * only the **feature block** changes.

Three questions, in order:

  V-A  does adding RGB-D to ``O+M+a`` raise re-observation AUC above the
       **non-vision ceiling already measured** (0.934 AUC, 0.792 RSR@10)?
  V-B  given the camera, is history needed, and is odometry needed?  H1 and H2
       already prove that the decision-point image is *ambiguous* - pixel-identical
       observations can require opposite actions - so a vision model that cannot use
       history or odometry is predicted to fail for a reason that has nothing to do
       with its visual encoder.

**One frame per state, not per (state, action).** The 2240 rows are 24 terrains x 10
states x 10 actions; the camera at a state is the same for all ten actions, so only
240 frames are needed. Rendering 2240 would be eight times the cost for no
information.

**The encoder is frozen.** A trained CNN would reintroduce the architecture confound
this design exists to remove. A frozen encoder is the conservative direction: if a
fixed representation already carries information beyond the state, vision helps; if
it does not, the result is suggestive rather than conclusive, and that limitation is
recorded rather than hidden.

    python tasks/task3_reobservability/vision_ablation.py --render     # once, ~3 min
    python tasks/task3_reobservability/vision_ablation.py --analyse
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    from . import api
    from . import episode as ep
except ImportError:
    import api
    import episode as ep

ROOT = Path(__file__).resolve().parents[2]
FRAMES = Path(__file__).resolve().parent / "generated" / "vision_frames.npz"
DATA = Path(__file__).resolve().parent / "generated" / "consequence_dataset.json"
OUT = Path(__file__).resolve().parent / "generated" / "vision_ablation.json"

RENDER_SIZE = (160, 120)   # (width, height), the API default
ENCODER = "resnet18"
HISTORY = 8          # history steps in the base condition; V-B ablates this to 0
VISION_PCS = 16      # vision projected to this many components inside each fold


# ---------------------------------------------------------------------------
# part 1: render one frame per (terrain, state)
# ---------------------------------------------------------------------------

def render_frames() -> None:
    """Re-walk each terrain exactly as the consequence dataset did, and render."""
    from benchmark import build_scene_terrain
    from consequence_dataset import wander_actions

    payload = json.loads(DATA.read_text(encoding="utf-8"))
    wanted = {(r["terrain"], r["state_id"]) for r in payload["rows"]}
    rgb, depth, keys = [], [], []

    for name, scene in api.scene_catalogue().items():
        if not any(t == name for t, _ in wanted):
            continue
        environment = api.MuJoCoEnvironment(scene, action_table=ep.protocol_action_table(),
                                            render=True, render_size=RENDER_SIZE)
        try:
            observation = environment.reset()
            snapshots = [ep.snapshot(environment)]
            observations = [observation]
            for action_id in wander_actions()[:9]:
                environment.step(action_id)
                now = environment.observe()
                if float(environment.data.xmat[1].reshape(3, 3)[2, 2]) < 0.25:
                    break
                observations.append(now)
                snapshots.append(ep.snapshot(environment))

            for state_id, (snap, obs) in enumerate(zip(snapshots, observations)):
                if (name, state_id) not in wanted:
                    continue
                ep.restore(environment, snap)
                fresh = environment.observe()
                rgb.append(np.asarray(fresh.current_rgb, dtype=np.uint8))
                depth.append(np.asarray(fresh.current_depth, dtype=np.float32))
                keys.append((name, state_id))
        finally:
            environment.close()
        print(f"  {name:18s} frames={len(keys)}", flush=True)

    np.savez_compressed(FRAMES, rgb=np.stack(rgb), depth=np.stack(depth),
                        keys=np.array([f"{t}|{s}" for t, s in keys]))
    print(f"\nwrote {FRAMES}  rgb={np.stack(rgb).shape} depth={np.stack(depth).shape}")


# ---------------------------------------------------------------------------
# part 2: encode frames with a frozen backbone
# ---------------------------------------------------------------------------

def encode(frames) -> np.ndarray:
    import torch
    import timm

    model = timm.create_model(ENCODER, pretrained=True, num_classes=0)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    out = []
    with torch.no_grad():
        for start in range(0, len(frames), 64):
            batch = torch.from_numpy(frames[start:start + 64]).float().permute(0, 3, 1, 2)
            batch = (batch / 255.0 - mean) / std
            out.append(model(batch.to(device)).cpu().numpy())
    return np.concatenate(out, axis=0)


def depth_features(depth: np.ndarray) -> np.ndarray:
    """Compact depth summary: quantiles plus a coarse spatial grid.

    A depth image cannot go through an RGB backbone, so it gets an explicit
    geometric summary rather than a learned encoder. That keeps the depth branch
    interpretable and cheap, and the ablation compares *presence* of depth, not the
    quality of a depth network.
    """
    d = np.asarray(depth, dtype=np.float32)
    d = np.where(np.isfinite(d), d, 0.0)
    d = np.clip(d, 0.0, 20.0)
    q = np.quantile(d.reshape(len(d), -1), [0.05, 0.25, 0.5, 0.75, 0.95], axis=1).T
    h, w = d.shape[1], d.shape[2]
    grid = d[:, : h // 4 * 4, : w // 4 * 4].reshape(len(d), h // 4, 4, w // 4, 4)
    coarse = grid.mean(axis=(2, 4)).reshape(len(d), -1)
    return np.hstack([q, coarse]).astype(np.float32)


# ---------------------------------------------------------------------------
# part 3: ablations
# ---------------------------------------------------------------------------

def analyse() -> dict:
    from consequence_model import build_training_matrix, load, pr_auc, roc_auc

    payload = load(DATA)
    rows = payload["rows"]
    action_ids = payload["action_ids"]
    blob = np.load(FRAMES, allow_pickle=True)
    key_to_index = {str(k): i for i, k in enumerate(blob["keys"])}

    order = [(r["terrain"], r["state_id"]) for r in rows]
    missing = [k for k in order if f"{k[0]}|{k[1]}" not in key_to_index]
    if missing:
        raise SystemExit(f"{len(missing)} states have no rendered frame, e.g. {missing[:3]}")
    idx = np.array([key_to_index[f"{t}|{s}"] for t, s in order])

    print(f"encoding {len(blob['rgb'])} frames with frozen {ENCODER} ...", flush=True)
    vis = encode(blob["rgb"])
    dep = depth_features(blob["depth"])[idx]
    vis = vis[idx]
    print(f"  rgb embedding {vis.shape}, depth features {dep.shape}")

    base = build_training_matrix(rows, "O+M+a", action_ids, history=HISTORY)
    base_nohist = build_training_matrix(rows, "O+M+a", action_ids, history=0)
    print(f"  history block: {HISTORY} steps in the base condition, 0 in the ablation "
          f"({base.shape[1]} vs {base_nohist.shape[1]} features)")

    y = np.array([1.0 if r["reobserved"] else 0.0 for r in rows])
    terrain = np.array([r["terrain"] for r in rows])

    vision_block = np.hstack([vis, dep])
    conditions = {
        f"state only (O+M+a, H={HISTORY}, no camera)": ("raw", base, None),
        "vision only (RGBD, PCA)": ("pca", None, vision_block),
        f"state + RGBD (PCA)": ("raw+pca", base, np.hstack([vis, dep])),
        f"state + RGBD (H=0)": ("raw+pca", base_nohist, vision_block),
    }

    from sklearn.decomposition import PCA
    from sklearn.ensemble import HistGradientBoostingClassifier
    results = {}
    for name, (mode, raw, pca_block) in conditions.items():
        probs = np.zeros(len(rows))
        for t in np.unique(terrain):
            test, train = terrain == t, terrain != t
            if len(np.unique(y[train])) < 2:
                probs[test] = y[train].mean()
                continue
            if mode == "raw":
                Xtr, Xte = raw[train], raw[test]
            else:
                # PCA inside the fold. The vision block has ~1700 columns against 30
                # for the state, so comparing them raw would confound the input with
                # the capacity. Fitting the projection on the training fold only
                # keeps the test terrain out of the reduction.
                p = PCA(n_components=min(VISION_PCS, pca_block.shape[1],
                                         int(train.sum()) - 1), random_state=0)
                tr_p = p.fit_transform(pca_block[train])
                te_p = p.transform(pca_block[test])
                if mode == "pca":
                    Xtr, Xte = tr_p, te_p
                else:
                    Xtr = np.hstack([raw[train], tr_p])
                    Xte = np.hstack([raw[test], te_p])
            m = HistGradientBoostingClassifier(max_depth=3, max_iter=250,
                                               learning_rate=0.08, random_state=7)
            m.fit(Xtr, y[train])
            probs[test] = m.predict_proba(Xte)[:, 1]
        results[name] = {"roc_auc": roc_auc(y, probs), "pr_auc": pr_auc(y, probs),
                         "n_features": int(Xtr.shape[1])}
        print(f"  {name:38s} features={Xtr.shape[1]:>4d}  "
              f"AUC={results[name]['roc_auc']:.4f}  PR-AUC={results[name]['pr_auc']:.4f}")

    ceiling_key = f"state only (O+M+a, H={HISTORY}, no camera)"
    ceiling = results[ceiling_key]["roc_auc"]
    vision_keys = [k for k in results if k != ceiling_key]
    best_vision = max(results[k]["roc_auc"] for k in vision_keys)
    verdict = {
        "non_vision_ceiling_auc": ceiling,
        "best_vision_auc": best_vision,
        "vision_adds_information": bool(best_vision > ceiling + 0.005),
        "vision_only_auc": results["vision only (RGBD, PCA)"]["roc_auc"],
        "history_ablation_gain": round(
            results[f"state + RGBD (PCA)"]["roc_auc"]
            - results["state + RGBD (H=0)"]["roc_auc"], 4),
        "note": "encoder is frozen and the vision block is PCA-reduced inside each "
                "fold, so the comparison is input-controlled in capacity as well as "
                "in content. A trained CNN could do better; that limitation is "
                "recorded rather than hidden.",
    }
    return {"conditions": results, "verdict": verdict}


def main() -> None:
    p = argparse.ArgumentParser(description="V-A/V-B vision ablation")
    p.add_argument("--render", action="store_true")
    p.add_argument("--analyse", action="store_true")
    p.add_argument("--output", type=Path, default=OUT)
    args = p.parse_args()

    if args.render:
        render_frames()
    if args.analyse:
        report = analyse()
        print("\n=== V-A / V-B ===")
        v = report["verdict"]
        print(f"non-vision ceiling (O+M+a, no camera): AUC {v['non_vision_ceiling_auc']:.4f}")
        print(f"best vision condition                 : AUC {v['best_vision_auc']:.4f}")
        print(f"vision adds information               : {v['vision_adds_information']}")
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
