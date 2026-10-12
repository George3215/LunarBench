"""Summarise a terrain_design_v4 Qwen3-VL trial directory.

Reads every `*/*/trials/*/result.json` and reports success and termination by
sensor condition, history condition, terrain family and ablation level.
Writes `summary.json` and `REPORT.md` next to the run.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def family_level(scene: str) -> tuple[str, str]:
    for level, suffix in (("simple", "_a0"), ("medium", "_a1"), ("complex", "_a2")):
        if scene.endswith(suffix):
            return scene[: -len(suffix)], level
    return scene, "unknown"


def rate(success: int, total: int) -> float:
    return round(100.0 * success / total, 2) if total else 0.0


def table(rows, key):
    counter = defaultdict(lambda: [0, 0])
    for row in rows:
        counter[row[key]][1] += 1
        counter[row[key]][0] += int(row["success"])
    lines = [f"| {key} | success | total | rate |", "|---|---:|---:|---:|"]
    for name in sorted(counter):
        success, total = counter[name]
        lines.append(f"| {name} | {success} | {total} | {rate(success, total)}% |")
    return lines, {k: {"success": v[0], "total": v[1], "rate": rate(v[0], v[1])} for k, v in counter.items()}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("--out", type=Path)
    a = p.parse_args()
    root = a.root.resolve()
    rows = []
    for path in sorted(root.glob("**/trials/*/result.json")):
        r = json.loads(path.read_text())
        fam, level = family_level(r["scene"])
        rel = path.relative_to(root)
        protocol = rel.parts[0] if len(rel.parts) > 3 else ("matrix" if "native8" not in r["horizon"] else "native")
        rows.append(
            {
                "scene": r["scene"],
                "family": fam,
                "level": level,
                "protocol": protocol,
                "mode": r["mode"],
                "horizon": r["horizon"],
                "success": bool(r["success"]),
                "termination": r["termination"],
                "action_steps": r["action_steps"],
                "autonomous_actions": r["autonomous_actions"],
                "approach_actions": r["controlled_approach_actions"],
                "inference_count": r["inference_count"],
                "total_inference_s": r["total_inference_s"],
                "final_distance_m": (r.get("final_evaluator") or {}).get("median_target_surface_range_m"),
                "path_result": str(path.relative_to(root)),
            }
        )
    total = len(rows)
    success = sum(r["success"] for r in rows)
    summary = {
        "root": str(root),
        "trials": total,
        "success": success,
        "success_rate": rate(success, total),
        "terminations": dict(Counter(r["termination"] for r in rows)),
        "by_mode": {},
        "by_horizon": {},
        "by_condition": {},
        "by_family": {},
        "by_level": {},
        "by_protocol": {},
    }
    _, summary["by_protocol"] = table(rows, "protocol")
    _, summary["by_mode"] = table(rows, "mode")
    _, summary["by_horizon"] = table(rows, "horizon")
    _, summary["by_family"] = table(rows, "family")
    _, summary["by_level"] = table(rows, "level")
    cond = defaultdict(lambda: [0, 0])
    for r in rows:
        cond[f'{r["mode"]}/{r["horizon"]}'][1] += 1
        cond[f'{r["mode"]}/{r["horizon"]}'][0] += int(r["success"])
    summary["by_condition"] = {
        k: {"success": v[0], "total": v[1], "rate": rate(v[0], v[1])} for k, v in cond.items()
    }

    lines = [
        "# terrain_design_v4 Qwen3-VL-8B 试验结果",
        "",
        f"- 试验数：{total}；成功：{success}；成功率：{rate(success, total)}%",
        f"- 条件：sensor ∈ {{rgb, rgbd}} × history ∈ {{current, first_current, all_history}}",
        f"- 终止分布：{summary['terminations']}",
        "",
    ]
    for key, title in (("protocol", "按协议"), ("mode", "按传感器"), ("horizon", "按历史"), ("condition", "按条件"), ("family", "按地形族"), ("level", "按档位")):
        data = summary[f"by_{key}"]
        lines += [f"## {title}", "", "| 条件 | 成功 | 总数 | 成功率 |", "|---|---:|---:|---:|"]
        for name in sorted(data):
            v = data[name]
            lines.append(f"| {name} | {v['success']} | {v['total']} | {v['rate']}% |")
        lines.append("")
    lines += ["## 逐回合", "", "| 地形 | mode | horizon | 结果 | 终止 | 自主动作 | 总动作 |", "|---|---|---|---|---|---:|---:|"]
    for r in sorted(rows, key=lambda x: (x["family"], x["level"], x["mode"], x["horizon"])):
        lines.append(
            f"| {r['scene']} | {r['mode']} | {r['horizon']} | {'success' if r['success'] else 'fail'} | "
            f"{r['termination']} | {r['autonomous_actions']} | {r['action_steps']} |"
        )
    out = a.out or root
    (out / "summary.json").write_text(json.dumps({"summary": summary, "trials": rows}, ensure_ascii=False, indent=2) + "\n")
    (out / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({k: summary[k] for k in ("trials", "success", "success_rate", "terminations", "by_condition")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
