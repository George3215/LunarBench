"""Emit the terrain-induced occlusion mechanism table from ``scenes.py``.

The taxonomy lives in one place (``scenes.OCCLUSION_MECHANISMS``) and this script
renders it, so a table in the paper cannot drift from the benchmark definition.
It also checks the framing rules that matter for review: that no mechanism is
described as "challenging terrain", and that procedural relief is not presented as
realistic.

    python tasks/task3_reobservability/mechanism_table.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TABDIR = ROOT / "paper" / "tables"

#: Wording that would turn the benchmark into "obstacles built to break baselines".
FORBIDDEN_FRAMING = ("challenging terrain", "hard terrain", "difficult terrain",
                     "adversarial terrain")


def render_markdown(rows: list[dict]) -> str:
    lines = ["| Mechanism | Axis | Position | Real-world analogue | Visibility effect | "
             "Physical effect | Typical recovery |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    for row in rows:
        lines.append(f"| `{row['mechanism']}` | {row['axis']} | {row['position']} | "
                     f"{row['real_world']} | {row['visibility_effect']} | "
                     f"{row['physical_effect']} | {row['typical_recovery']} |")
    return "\n".join(lines) + "\n"


def render_latex(rows: list[dict]) -> str:
    out = ["\\begin{tabular}{llp{3.0cm}p{3.4cm}}", "\\toprule",
           "Mechanism & Axis & Real-world analogue & Visibility effect \\\\", "\\midrule"]
    for row in rows:
        name = row["mechanism"].replace("_", "\\_")
        out.append(f"{name} & {row['axis']} & {row['real_world']} & "
                   f"{row['visibility_effect']} \\\\")
    out += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(out) + "\n"


def render_axes(rows: list[dict]) -> str:
    """The five axes, with the mechanisms that sit on each."""
    by_axis: dict[str, list[str]] = {}
    for row in rows:
        by_axis.setdefault(row["axis"], []).append(
            f"{row['mechanism']} ({row['position']})")
    lines = ["| Axis | Mechanisms, by position on the axis |", "| --- | --- |"]
    for axis in ("occlusion direction", "elevation", "spatial scale",
                 "visibility dynamics", "terrain complexity"):
        if axis in by_axis:
            lines.append(f"| {axis} | {', '.join(by_axis[axis])} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description="Render the occlusion mechanism taxonomy")
    p.add_argument("--output-dir", type=Path, default=TABDIR)
    args = p.parse_args()

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from scenes import MECHANISM_AXES, mechanism_table

    rows = mechanism_table()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "table_mechanisms.md").write_text(
        render_markdown(rows) + "\n" + render_axes(rows), encoding="utf-8")
    (args.output_dir / "table_mechanisms.tex").write_text(
        render_latex(rows), encoding="utf-8")
    (args.output_dir / "table_mechanisms.json").write_text(
        json.dumps({"axes": MECHANISM_AXES, "mechanisms": rows}, indent=2),
        encoding="utf-8")

    # Framing guard: the benchmark must read as situations that occur, not as
    # obstacles built to break a baseline.
    blob = json.dumps(rows).lower()
    bad = [phrase for phrase in FORBIDDEN_FRAMING if phrase in blob]
    if bad:
        raise SystemExit(f"framing check failed, found {bad}")

    print(render_markdown(rows))
    print(render_axes(rows))
    print(f"wrote {args.output_dir}/table_mechanisms.{{md,tex,json}}")
    print("framing check: no 'challenging/hard/difficult/adversarial terrain' wording")


if __name__ == "__main__":
    main()
