"""Audit and analyse paired TASK3 checkpoint transfer; partial runs stay partial."""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np

MODELS = ("qwen3vl8b", "spatial_interactor8b")
FORBIDDEN = ("target_centre_xyz_evaluator", "camera_xyz_evaluator", "blocking_geom",
             "target_segment_pixels", "base_xy_target_distance", "task3_target_geom",
             '"terrain_id"', '"terrain_spec"', '"oracle"')


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    tmp.replace(path)


def lines(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def unpadded(net):
    return [v for v, m in zip(net["input_ids"], net["attention_mask"]) if m]


def episode(path):
    result = read(path / "result.json")
    prefix = result["controlled_approach_actions"]
    samples = [s for s in lines(path / "evaluator.jsonl") if s["step"] > prefix]
    streak = maximum = 0
    first_reobserved = None
    for sample in samples:
        ok = sample["target_visible_rgb_pixels"] >= 20 and sample["upright"] >= .25 and sample["in_bounds"]
        streak = streak + 1 if ok else 0
        maximum = max(maximum, streak)
        if streak >= 3 and first_reobserved is None:
            first_reobserved = sample["sim_time_s"]
    actions = result["policy_action_sequence"]
    transitions = [t for t in lines(path / "trace.jsonl") if t.get("phase") == "vlm_autonomous"]
    turns = ("TURN_LEFT", "TURN_RIGHT")
    alternating = sum(x in turns and y in turns and x != y for x, y in zip(actions, actions[1:]))
    loss_types = collections.Counter()
    grammar_interceptions = 0
    for output in path.glob('steps/*/inference_*/output.json'):
        grammar_interceptions += read(output).get('action_grammar', {}).get('forbidden_stop_path_interceptions', 0)
    visibility_correct = near_claims = false_near_claims = 0
    valid_decisions = 0
    for t in transitions:
        decision = t.get("decision")
        if decision:
            valid_decisions += 1
            observed = t["before_evaluator"]
            visibility_correct += decision["target_visible"] == (observed["target_visible_rgb_pixels"] >= 20)
            if decision["proximity"] == "near":
                near_claims += 1
                distance = observed["median_target_surface_range_m"]
                false_near_claims += not (observed["target_visible_rgb_pixels"] >= 20 and distance is not None and 0 < distance < 1)
        if t["after_evaluator"]["target_visible_rgb_pixels"] >= 20:
            continue
        ray = t["evaluator_ray"]
        loss_types["fov_exit" if not ray["target_centre_in_fov"] else
                   "terrain_occlusion" if ray["entity_occlusion"] else "other"] += 1
    row = {"scene": result["scene"], "terrain_id": result["terrain_id"],
           "arrived": result["visible_near_arrival"], "legal_vlm_stop": result["legal_vlm_stop"],
           "reobserved": first_reobserved is not None,
           "first_reobserved_sim_s": first_reobserved, "max_reobserved_streak": maximum,
           "termination": result["termination"], "autonomous_actions": result["autonomous_actions"],
           "path_m_including_prefix": result["path_m"],
           "policy_forward_only": bool(actions) and all(a in ("FORWARD", "STOP") for a in actions),
           "low_translation_actions": result["low_translation_actions"],
           "turn_actions": sum(a in turns for a in actions),
           "alternating_turn_pairs": alternating,
           "model_plan_updates": sum(isinstance((t.get("decision") or {}).get("plan_update"), str) and bool(t["decision"]["plan_update"].strip()) for t in transitions),
           "fov_exit_boundaries": loss_types["fov_exit"],
           "terrain_occlusion_boundaries": loss_types["terrain_occlusion"],
           "valid_decisions": valid_decisions,
           "current_visibility_correct_decisions": visibility_correct,
           "model_near_claims": near_claims,
           "model_near_claims_outside_1m_gate": false_near_claims,
           "rejected_stops": result["rejected_stop_count"], "initial_plan_valid": result["initial_plan_valid"],
           "stop_paths_masked_during_generation": grammar_interceptions,
           "inference_count": result["inference_count"],
           "amortized_actor_inference_s": result["amortized_inference_s"],
           "resource_limit": result["resource_limit"]}
    return row


def audit_run(run):
    requests = 0
    violations = []
    unsupported = 0
    for path in run.rglob("messages.json"):
        saved = read(path)
        messages = saved["messages"]
        texts = []
        if [m["role"] for m in messages] != ["system", "user"]:
            violations.append({"file": str(path), "reason": "unexpected role history"})
        for message in messages:
            blocks = message["content"]
            if isinstance(blocks, str):
                texts.append(blocks)
                continue
            for block in blocks:
                if block["type"] == "text":
                    texts.append(block["text"])
                elif block["type"] != "image":
                    violations.append({"file": str(path), "reason": "unexpected input type"})
        text = "\n".join(texts)
        for key in FORBIDDEN:
            if key in text:
                violations.append({"file": str(path), "reason": key})
        requests += 1
        out = path.with_name("output.json")
        if out.exists():
            record = read(out)
            unsupported += bool(record.get("unsupported_evidence_frame_ids") or record.get("unsupported_depth_claim") or record.get("unsupported_past_claim"))
    return {"requests_checked": requests, "forbidden_field_or_role_violations": violations,
            "unsupported_model_evidence_claims": unsupported,
            "scope": "actual text blocks and role allowlist; image paths are provenance, not model text; not a proof against all possible semantic leakage"}


def matched_initial_inputs(a, b):
    pairs = []
    for folder in sorted(a.glob("rgbd/*/trials/*")):
        other = b / folder.relative_to(a)
        first, second = folder / "initial_plan/network_input.json", other / "initial_plan/network_input.json"
        if not first.exists() or not second.exists():
            continue
        x, y = read(first), read(second)
        g, h = read(folder / "approach_gate.json"), read(other / "approach_gate.json")
        pairs.append({"scene": folder.name, "same_initial_prompt_tokens": unpadded(x) == unpadded(y),
                      "same_initial_processed_images": [i["pixel_values_float32_sha256"] for i in x["images"]] == [i["pixel_values_float32_sha256"] for i in y["images"]],
                      "same_initial_rgb": g["initial_rgb_sha256"] == h["initial_rgb_sha256"],
                      "same_lost_rgb": g["lost_rgb_sha256"] == h["lost_rgb_sha256"],
                      "same_loss_qpos": g["loss_qpos"] == h["loss_qpos"]})
    return pairs


def summary(rows):
    return {"completed": len(rows), "arrivals": sum(r["arrived"] for r in rows),
            "reobserved": sum(r["reobserved"] for r in rows),
            "legal_vlm_stop": sum(r["legal_vlm_stop"] for r in rows),
            "forward_only_arrivals": sum(r["arrived"] and r["policy_forward_only"] for r in rows),
            "rejected_stops": sum(r["rejected_stops"] for r in rows),
            "stop_paths_masked_during_generation": sum(r["stop_paths_masked_during_generation"] for r in rows),
            "valid_initial_plans": sum(r["initial_plan_valid"] for r in rows),
            "valid_decisions": sum(r["valid_decisions"] for r in rows),
            "current_visibility_correct_decisions": sum(r["current_visibility_correct_decisions"] for r in rows),
            "model_near_claims": sum(r["model_near_claims"] for r in rows),
            "model_near_claims_outside_1m_gate": sum(r["model_near_claims_outside_1m_gate"] for r in rows),
            "mean_autonomous_actions": float(np.mean([r["autonomous_actions"] for r in rows])) if rows else None,
            "terminations": dict(collections.Counter(r["termination"] for r in rows))}


def preserve_case(root, key, folder):
    """Compact auditable decisions, not the full multi-GB request/image archive."""
    steps = []
    for t in lines(folder / "trace.jsonl"):
        if t.get("phase") != "vlm_autonomous":
            continue
        decision = t.get("decision")
        value = {"step": t["step"], "action": t["action"], "decision": decision,
                 "current_frame_before": t["frame_before"],
                 "actual_displacement_xy_m": t["actual_displacement_xy_m"],
                 "self_odometry_after": t["self_odometry_after"],
                 "current_evaluator_after": {k: t["after_evaluator"][k] for k in
                     ("target_visible_rgb_pixels", "median_target_surface_range_m", "upright", "in_bounds")},
                 "evaluator_only_loss_diagnosis": t["evaluator_ray"],
                 "rejected_stop_count": t["rejected_stop_count"], "termination": t["termination"]}
        inputs = folder / "steps" / f"{t['step']:04d}" / "inference_1/network_input.json"
        if inputs.exists():
            net = read(inputs)
            ids = unpadded(net)
            value["prompt_token_ids_sha256"] = hashlib.sha256(json.dumps(ids).encode()).hexdigest()
            value["input_images"] = [{k: image[k] for k in ("label", "image_sha256", "pixel_values_float32_sha256")} for image in net["images"]]
        if not decision:
            outputs = sorted((folder / "steps" / f"{t['step']:04d}").glob("inference_*/output.json"))
            value["rejected_or_invalid_outputs"] = [{"raw_output": read(p)["raw_output"],
                 "rejected_stop": read(p).get("rejected_stop", False)} for p in outputs]
        steps.append(value)
    plan = folder / "initial_plan/output.json"
    write(root / "case_records" / key / f"{folder.name}.json", {
          "scope": "derived compact evidence; evaluator fields never sent to model; raw images and full inputs remain local",
          "initial_plan": read(plan).get("plan_parsed") if plan.exists() else None,
          "result": read(folder / "result.json"), "steps": steps})


def trajectories(root, by_model):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    scenes = sorted(set(r["scene"] for rows in by_model.values() for r in rows))
    if not scenes:
        return
    fig, axes = plt.subplots(math.ceil(len(scenes)/4), 4, figsize=(16, 3.7*math.ceil(len(scenes)/4)), squeeze=False)
    for ax, scene in zip(axes.flat, scenes):
        for key, color in zip(MODELS, ("#1769aa", "#ed8311")):
            folder = root / "runs" / key / "rgbd/event4_odom_plan/trials" / scene
            if not (folder / "result.json").exists():
                continue
            result = read(folder / "result.json")
            states = lines(folder / "state.jsonl")
            if not states:
                continue
            xy = np.array([s["qpos"][:2] for s in states])
            origin = np.array(read(folder / "initialization.json")["qpos"][:2])
            xy -= origin
            ax.plot(xy[:, 0], xy[:, 1], color=color, label=key, linewidth=1.4)
            ax.scatter(*xy[-1], color=color, marker="*" if result["success"] else "x", s=85)
            target = np.array(states[-1]["evaluation"]["target_centre_xyz_evaluator"][:2])-origin
            ax.scatter(*target, color="red", marker="+", s=65)
            gate = read(folder / "approach_gate.json")
            loss = np.array(gate["loss_qpos"][:2])-origin
            ax.scatter(*loss, color="black", marker="D", s=18)
        ax.scatter(0, 0, color="black", marker="o", s=18)
        ax.set_title(scene, fontsize=10)
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(alpha=.25)
        ax.set_xlabel("world x offset (m)")
        ax.set_ylabel("world y offset (m)")
    for ax in list(axes.flat)[len(scenes):]:
        ax.axis("off")
    fig.suptitle("V4 checkpoint comparison: blue=Qwen; orange=Spatial; *=arrived; x=failed; red+=target; black diamond=loss\nEvaluator trajectories only; target coordinates never supplied to policy", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, .96))
    fig.savefig(root / "trajectories.svg")
    svg = root / "trajectories.svg"
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    fig.savefig(root / "trajectories.png", dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("--plots", action="store_true")
    a = p.parse_args()
    root = a.root.resolve()
    by_model = {}
    audit = {}
    for key in MODELS:
        run = root / "runs" / key
        by_model[key] = [episode(f.parent) for f in sorted(run.glob("rgbd/*/trials/*/result.json"))]
        for path in sorted(run.glob("rgbd/*/trials/*/result.json")):
            preserve_case(root, key, path.parent)
        audit[key] = audit_run(run)
    metrics = {key: summary(rows) for key, rows in by_model.items()}
    pairs = matched_initial_inputs(root / "runs" / MODELS[0], root / "runs" / MODELS[1])
    index = {key: {r["scene"]: r for r in rows} for key, rows in by_model.items()}
    common = sorted(set(index[MODELS[0]]) & set(index[MODELS[1]]))
    difference = np.array([int(index[MODELS[1]][s]["arrived"])-int(index[MODELS[0]][s]["arrived"]) for s in common])
    paired = {"completed_pairs": len(common)}
    if len(difference):
        ci = np.quantile(np.random.default_rng(7).choice(difference, size=(20000, len(difference))).mean(axis=1), [.025, .975])
        better, worse = int((difference > 0).sum()), int((difference < 0).sum())
        n = better+worse
        prob = min(1., 2*sum(math.comb(n, k) for k in range(min(better, worse)+1))/2**n) if n else 1.
        paired.update(arrival_difference=float(difference.mean()), bootstrap_95_ci=ci.tolist(),
                      spatial_only_arrived=better, qwen_only_arrived=worse, mcnemar_exact_two_sided_p=prob)
        clusters = collections.defaultdict(list)
        for scene, delta in zip(common, difference):
            clusters[scene.rsplit("_a", 1)[0]].append(delta)
        groups = [np.asarray(v) for v in clusters.values()]
        rng = np.random.default_rng(17)
        draws = [float(np.concatenate([groups[i] for i in rng.integers(len(groups), size=len(groups))]).mean())
                 for _ in range(20000)]
        paired.update(terrain_families=len(groups), family_cluster_bootstrap_95_ci=np.quantile(draws, [.025, .975]).tolist())
    complete = all(v["completed"] == 22 for v in metrics.values())
    common_settings = ("version", "scenes", "admission_manifest_sha256", "system_prompt", "plan_system_prompt",
                       "action_budget", "max_new_tokens", "plan_max_new_tokens", "batch_size", "frame_area_cap",
                       "event_selection", "sensors", "stop_gate", "action_constraint", "shared_output_adapter", "arrival_endpoint", "seed", "decoding")
    protocols = [root / "runs" / key / "protocol.json" for key in MODELS]
    mismatches = []
    if all(p.exists() for p in protocols):
        x, y = map(read, protocols)
        mismatches = [k for k in common_settings if x[k] != y[k]]
    input_pairs_ok = len(pairs) == 22 and all(all(v for k, v in pair.items() if k != "scene") for pair in pairs)
    audit_ok = complete and input_pairs_ok and not mismatches and all(not r["forbidden_field_or_role_violations"] for r in audit.values())
    write(root / "metrics.json", {"status": "complete" if complete else "partial", "models": metrics, "paired": paired,
                                   "uncertainty_scope": "scene bootstrap, not repeated-seed or checkpoint-training uncertainty"})
    write(root / "INPUT_AUDIT.json", {"models": audit, "paired_initial_inputs": pairs,
          "shared_protocol_setting_mismatches": mismatches, "complete_comparison_inputs_pass": audit_ok,
          "identical_dynamic_requests_claimed": False,
          "reason": "self-generated plans and observations diverge after checkpoint decisions"})
    flat = [{"model": key, **row} for key, rows in by_model.items() for row in rows]
    if flat:
        with (root / "episodes.csv").open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(flat[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(flat)
    text = ["# V4 空间检查点迁移实验", "", "状态：" + ("44/44 回合完成。" if complete else "进行中，以下为阶段记录，不能作完整模型比较。"), "",
            "| 模型 | 完成 | 确认到达 | 遮挡后恢复可见 | 合法 VLM STOP | 到达中仅前进 |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for key in MODELS:
        r = metrics[key]
        text.append(f"| {key} | {r['completed']}/22 | {r['arrivals']} | {r['reobserved']} | {r['legal_vlm_stop']} | {r['forward_only_arrivals']} |")
    text += ["", "到达由连续三次当前可见且 <1m 的 evaluator 证据自动结束；不等同于模型主动 STOP。自动结束可能先于下一次 STOP 决策，因此合法 STOP 数量不是独立的终止能力评分。参考动作准入不计入模型结果。", "",
             "| 场景 | Qwen 到达 / 终止 | Spatial 到达 / 终止 |", "| --- | --- | --- |"]
    for scene in sorted(set(index[MODELS[0]]) | set(index[MODELS[1]])):
        cells = []
        for key in MODELS:
            r = index[key].get(scene)
            cells.append(("成功" if r["arrived"] else "失败") + " / " + r["termination"] if r else "待完成")
        text.append(f"| {scene} | {' | '.join(cells)} |")
    if len(common):
        text += ["", f"已完成配对数 {len(common)}；Spatial−Qwen 到达率差 {paired['arrival_difference']*100:.1f} 个百分点。",
                 f"配对场景 bootstrap 95% 区间 [{paired['bootstrap_95_ci'][0]*100:.1f}, {paired['bootstrap_95_ci'][1]*100:.1f}] 个百分点；精确 McNemar p={paired['mcnemar_exact_two_sided_p']:.4f}。",
                 f"考虑同一地形族 a0/a1/a2 的相关性：{paired['terrain_families']} 个族的 cluster bootstrap 区间 [{paired['family_cluster_bootstrap_95_ci'][0]*100:.1f}, {paired['family_cluster_bootstrap_95_ci'][1]*100:.1f}] 个百分点。McNemar 的独立场景假设在此不完全成立，仅作描述。",
                 "单回合/固定种子的场景级小样本区间，不覆盖多次训练或生成随机性的方差。"]
    if complete:
        text += ["", "## 本轮结论", ""]
        if not audit_ok:
            text += ["回合已完成，但配对输入或协议审计未全部通过。暂不据此得出模型优劣结论，先检查 INPUT_AUDIT.json。"]
        else:
            delta = paired["arrival_difference"]
            low, high = paired["family_cluster_bootstrap_95_ci"]
            if delta > 0 and low > 0:
                statement = "本轮相同适配协议下，Spatial 检查点的到达率较高，地形族 bootstrap 区间也保持为正；这是本批场景的迁移证据，仍需独立场景和重复实验验证。"
            elif delta > 0:
                statement = "Spatial 检查点的到达率有数值提升，但地形族 bootstrap 区间未排除零；本轮不足以确认可靠的端到端提升。"
            elif delta == 0:
                statement = "两检查点到达数相同，本轮没有观察到端到端到达率提升；不能由此证明所有空间能力均无用。"
            else:
                statement = "Spatial 检查点在本轮到达率较低，当前适配下未获得正向迁移；不能据此概括其原生空间 QA 能力。"
            text.append(statement)
            q, s = (metrics[k] for k in MODELS)
            text += [f"重观测分别为 Qwen {q['reobserved']}/22、Spatial {s['reobserved']}/22；到达分别为 {q['arrivals']}/22、{s['arrivals']}/22。重观测提升和最终到达提升须分别判断。",
                     "", "| 诊断 | Qwen | Spatial |", "| --- | ---: | ---: |",
                     f"| 有效初始计划 | {q['valid_initial_plans']} | {s['valid_initial_plans']} |",
                     f"| 被拒绝的提前 STOP 请求 | {q['rejected_stops']} | {s['rejected_stops']} |",
                     f"| 当前可见性判断正确 / 有效动作输出 | {q['current_visibility_correct_decisions']}/{q['valid_decisions']} | {s['current_visibility_correct_decisions']}/{s['valid_decisions']} |",
                     f"| near 判断未满足 1m 门槛 / near 判断数 | {q['model_near_claims_outside_1m_gate']}/{q['model_near_claims']} | {s['model_near_claims_outside_1m_gate']}/{s['model_near_claims']} |", "",
                     "near 的诊断采用任务明确的 1m 到达门槛；可见性诊断采用原始图像目标像素≥20。视觉输入被共同缩放，细小目标识别亦可能影响该指标。",
                     "后续只针对已有证据选择改进：若主要是提前 STOP，先验证距离与动作终止校准；若恢复可见失败，使用动作条件的重现概率/距离预测；若已经重观测但无法接近，分别检查路径可达性、运动执行和接近策略。不能以单一到达数决定所有训练方向。"]
    text += ["", "## 解释边界", "",
             "本轮检验相同适配协议下两个完整检查点的零样本迁移，不单独隔离空间训练数据、训练目标或其他检查点变化的因果作用。",
             "初见—遮挡物理状态重新冻结；目标真值只用于 evaluator 和轨迹图。里程计为理想仿真自身运动，尚未验证真实 RGB-D/视觉里程计噪声。",
             "V4 原准入认证重观测，不代表所有场景均已验证完整 <1m 导航参考路线；连续控制参考检查的未验证场景保留在 continuous_reference_scope.json，未据模型结果删除场景。到达失败不能全部归于模型能力。",
             "未进行 RSI、微调、新任务头或世界模型训练。本轮不能作为新方法收益或论文创新性的证据。",
             "完整逐回合动作数、失败类型、重观测、低位移、STOP 拒绝及资源错误见 episodes.csv 和 metrics.json；输入检查见 INPUT_AUDIT.json。",
             "模型指令相同；模型自己产生的初始计划和后续画面可以不同。初始计划输入 token、图像及丢失位姿需逐场景一致。",
             "GitHub 保存协议、结果与分析；原始请求/轨迹/RGB-D 在本机，没有据此声称原始数据异地备份。"]
    (root / "REPORT.md").write_text("\n".join(text)+"\n")
    write(root / "STATUS.json", {"status": "complete_awaiting_review" if complete else "model_evaluation_running",
          "completed_model_episodes": sum(v["completed"] for v in metrics.values()), "expected_model_episodes": 44,
          "models": metrics, "updated_unix": time.time()})
    if a.plots:
        trajectories(root, by_model)
    print(json.dumps({"models": metrics, "paired": paired}, ensure_ascii=False))


if __name__ == "__main__":
    main()
