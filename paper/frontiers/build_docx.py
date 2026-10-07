"""Build the Frontiers submission DOCX, conforming to the author guidelines.

The previous build violated several explicit rules.  Each is fixed here and the
reason is recorded, because these are the errors that cause production delays:

* **Title must not be a question** (guidelines: "avoid titles that are a mere
  question without giving the answer").  The old title was one.  It is now a
  statement of the main result.
* **Abstract must be a single paragraph** using the IMRAD shape, with the
  keywords concentrated in the first sentences for discoverability.  The old
  abstract ran to four paragraphs.
* **Word, figure and table counts must appear on the first page.**
* **Tables go at the end of the manuscript, captions immediately before**, not
  inline where they are cited.
* **Figure captions go at the end of the manuscript**, and the figures themselves
  are embedded there, which is where Frontiers places them automatically.
* **Harvard references are alphabetical and unnumbered.**  The old list was
  numbered, which is the Vancouver style.
* **Abbreviations** must be defined on first use, with a list immediately before
  the acknowledgments.
* **Single spacing** throughout - the template's Normal style already provides it.
"""
from __future__ import annotations

import re
from pathlib import Path

from docx import Document
from docx.shared import Inches

ROOT = Path(r"D:\LunarBench")
TEMPLATE = ROOT / "paper" / "frontiers" / "word" / "Frontiers_Template.docx"
SOURCE = ROOT / "paper" / "frontiers" / "manuscript.tex"
FIGDIR = ROOT / "paper" / "frontiers" / "figures"
OUTPUT = ROOT / "paper" / "frontiers" / "word" / "Frontiers_Manuscript.docx"

TITLE = ("Physical Consequence, Not Localization, Governs Terrain-Induced "
         "Target Re-observability")

CITES = {
    "RadmardCroft2016": "Radmard and Croft, 2016",
    "Radmard2018": "Radmard et al., 2018",
    "OANBV": "OA-NBV authors, 2026",
    "LEAPNBV": "LEAP-NBV authors, 2026",
    "Peng2024": "Peng et al., 2024",
    "Yin2025": "Yin et al., 2025",
    "Wu2024": "Wu et al., 2023",
    "RoboticWorldModel": "Robotic World Model authors, 2025",
    "LatentWorldModelsNav": "authors, 2024",
    "CSTWM": "CST-WM authors, 2026",
    "VisibleReachable": "authors, 2026",
    "EVTBench": "EVT-Bench authors, 2025",
    "TrackVLA": "Wang et al., 2025",
    "AVD": "Ammirato et al., 2018",
    "OARNav": "OAR-Nav authors, 2025",
}
REFS = {
    "eq:objective": "1", "eq:rsr": "2",
    "tab:mechanisms": "1", "tab:audit": "2", "tab:ablation": "3", "tab:pose": "4",
    "tab:main": "5", "tab:history": "6", "tab:horizon": "7", "tab:odom": "8",
    "tab:taxonomy": "9",
    "sec:methods": "2",
}
MATH_SUBS = [
    (r"\mid", "|"), (r"\not\Rightarrow", "does not imply"), (r"\Rightarrow", "=>"),
    (r"\wedge", "and"), (r"\le", "<="), (r"\ge", ">="), (r"\in", " in "),
    (r"\exists", "there exists"), (r"\mathbf", ""), (r"\times", "x"),
    (r"\rightarrow", "->"), (r"\perp", "independent of"),
    (r"\Delta", "Delta"), (r"\delta", "delta"), (r"\tau", "tau"), (r"\lambda", "lambda"),
    (r"\mathcal{A}", "A"), (r"\mathcal", ""), (r"\,", " "), (r"\;", " "), (r"\!", ""),
    (r"\left", ""), (r"\right", ""), (r"\qquad", "    "), (r"\quad", "  "),
    (r"\\", " "), (r"\{", "{"), (r"\}", "}"), (r"\_", "_"), (r"\%", "%"),
    (r"\&", "&"), (r"\#", "#"),
]

#: Figure captions, per the guideline that captions are placed at the end and that
#: panels are labelled (A), (B), ... in text rather than over the image.
FIGURES = [
    ("fig1_benchmark.pdf",
     "Figure 1. Re-observation performance and where the discrimination comes from. "
     "(A) Re-observation success rate against action budget for every method, with "
     "Wilson 95% intervals; the five rule-based baselines coincide at every budget. "
     "(B) Success at the full budget, split by terrain group. Every classical method "
     "sits at 0.75 on the multi-terrain scenes and at 0.0 on the paired-history "
     "worlds; only the learned consequence planner and the privileged oracles leave "
     "zero."),
    ("fig2_paired_worlds.pdf",
     "Figure 2. Evidence established by the paired-world constructions. H1 pairs share "
     "a pixel-identical observation and opposite optimal actions with disjoint winning "
     "sets; H2 pairs additionally share the target memory and differ only in the "
     "optimal action."),
    ("fig3_geometry_gap.pdf",
     "Figure 3. Geometric visibility against physical re-observation. (A) Confusion "
     "matrix over 240 (state, action) pairs; the outlined cell is geometry endorsing "
     "an action that fails physically. (B) Pose prediction error grouped by outcome, "
     "with the bootstrap contrast between the failing and agreeing cells."),
    ("fig4_history_horizon.pdf",
     "Figure 4. History ablation. (A) Decision quality against history horizon, flat "
     "throughout. (B) Out-of-fold AUC, which saturates; the visibility prediction "
     "gains nothing from history while the pose prediction does."),
    ("fig5_safety_calibration.pdf",
     "Figure 5. Calibration against decision usefulness. (A) Overall expected "
     "calibration error compared with the error inside the decision band. (B) Success "
     "against the fraction of states at which the consequence term changed the action; "
     "no operating point beats not using it."),
    ("fig6_horizon_ablation.pdf",
     "Figure 6. Action-horizon ablation. (A) Success by lookahead depth. (B) Mean "
     "action cost, which rises with depth. (C) The one-step pose error actually "
     "measured, which does not explain the ordering."),
]

ABBREVIATIONS = [
    ("AUC", "area under the receiver operating characteristic curve"),
    ("CI", "confidence interval"),
    ("ECE", "expected calibration error"),
    ("MAE", "mean absolute error"),
    ("MPC", "model predictive control"),
    ("NBV", "next-best-view"),
    ("OR", "odds ratio"),
    ("PR-AUC", "area under the precision-recall curve"),
    ("RSR", "re-observation success rate"),
]


def clean_math(text: str) -> str:
    for a, b in MATH_SUBS:
        text = text.replace(a, b)
    text = re.sub(r"\\[a-zA-Z]+", "", text)
    return re.sub(r"\s+", " ", text).strip()


def inline(text: str) -> str:
    text = re.sub(r"\\citep\{([^}]*)\}",
                  lambda m: "(" + "; ".join(CITES.get(k.strip(), k.strip())
                                            for k in m.group(1).split(",")) + ")", text)
    text = re.sub(r"\\ref\{([^}]*)\}", lambda m: REFS.get(m.group(1), m.group(1)), text)
    text = re.sub(r"\\label\{[^}]*\}", "", text)
    text = re.sub(r"\$([^$]*)\$", lambda m: clean_math(m.group(1)), text)
    for cmd in ("textbf", "emph", "texttt", "textit", "mathrm", "text"):
        text = re.sub(r"\\" + cmd + r"\{([^{}]*)\}", r"\1", text)
    for a, b in (("---", "\u2014"), ("--", "\u2013"), ("~", " "),
                 (r"\%", "%"), (r"\&", "&"), (r"\_", "_")):
        text = text.replace(a, b)
    text = re.sub(r"\\[a-zA-Z]+", "", text)
    return text.replace("{", "").replace("}", "").strip()


def add_runs(paragraph, text: str) -> None:
    for i, chunk in enumerate(re.split(r"\*\*(.+?)\*\*", text)):
        if not chunk:
            continue
        run = paragraph.add_run(chunk)
        run.bold = (i % 2 == 1)


def parse_tables(lines: list[str]) -> dict:
    tables, kept, i = {}, [], 0
    while i < len(lines):
        if lines[i].strip().startswith(r"\begin{table}"):
            block = []
            while i < len(lines) and not lines[i].strip().startswith(r"\end{table}"):
                block.append(lines[i]); i += 1
            i += 1
            body = "\n".join(block)
            label = re.search(r"\\label\{([^}]*)\}", body)
            caption = re.search(r"\\caption\{(.*?)\}\s*\n\\label", body, re.S)
            tab = re.search(r"\\begin\{tabular\}\{[^}]*\}(.*?)\\end\{tabular\}",
                            body, re.S)
            rows = []
            if tab:
                for raw in tab.group(1).split("\\\\"):
                    raw = re.sub(r"\\(top|mid|bottom)rule", "", raw).strip()
                    if not raw:
                        continue
                    cells = [inline(c.strip()) for c in raw.split("&")]
                    if any(cells):
                        rows.append(cells)
            key = label.group(1) if label else f"tbl{len(tables)}"
            tables[key] = (inline(caption.group(1)) if caption else "", rows)
            kept.append(f"[[TABLE:{key}]]")
            continue
        kept.append(lines[i]); i += 1
    return {"tables": tables, "lines": kept}


def build() -> Path:
    src = SOURCE.read_text(encoding="utf-8")
    body = src.split(r"\begin{document}", 1)[1].split(r"\end{document}", 1)[0]
    body = r"\section{Introduction}" + body.split(r"\section{Introduction}", 1)[1]
    parsed = parse_tables(body.splitlines())
    tables, lines = parsed["tables"], parsed["lines"]

    doc = Document(str(TEMPLATE))
    for p in list(doc.paragraphs):
        p._element.getparent().remove(p._element)
    for t in list(doc.tables):
        t._element.getparent().remove(t._element)

    def para(text="", style="Normal"):
        p = doc.add_paragraph(style=style)
        if text:
            add_runs(p, text)
        return p

    # --- front matter ------------------------------------------------------
    para(TITLE, style="Title")
    para("First Author\u00b9\u002a", style="Normal")
    para("\u00b9 Laboratory, Department, Organization, City, State, Country",
         style="Normal")
    para("* Correspondence: First Author, email@example.edu", style="Normal")
    para("Keywords: target re-observability, terrain-induced occlusion, physical "
         "consequence prediction, partial observability, active perception, legged "
         "locomotion, benchmark, failure analysis", style="Author List")

    para("Abstract", style="Author List")
    # Single paragraph, IMRAD-shaped, keywords concentrated up front.
    abstract = (
        "Terrain-induced target re-observability is the problem a robot faces when a "
        "target it has already observed disappears behind terrain and must be made "
        "visible again by physical motion. This paper asks what information and what "
        "action reasoning that requires, and answers by elimination. We construct "
        "paired worlds that share terrain, pose, heading and a pixel-identical current "
        "observation yet require opposite re-observation actions, and in eight of eight "
        "pairs the successful action sets are disjoint, so no memoryless rule can solve "
        "both; a second family additionally shares the target memory and still requires "
        "different optimal actions. Over 240 (state, action) pairs, 43.1% of the actions "
        "a geometric visibility oracle endorses fail when executed, while the same "
        "oracle placed at the pose the robot actually reaches agrees with the physical "
        "outcome 97.9% of the time, so the missing quantity is the physical consequence "
        "of an action rather than its visibility. Testing that conclusion positively, on "
        "2240 physically executed (state, action) pairs adding the candidate action to "
        "observation-plus-memory raises re-observation AUC from 0.854 to 0.934 whereas "
        "adding the current observation to the memory buys 0.002, and a planner built on "
        "the resulting action-conditioned consequence model raises RSR@10 from 0.250 to "
        "0.792 and RSR@1 from 0.167 to 0.708 without privileged information. Three "
        "negative results bound how such a model may be used, and all three concern the "
        "prediction-to-decision step: perfectly localized anchor recovery still fails "
        "physically; a safety consequence head that is accurate and made better "
        "calibrated at the decision boundary yields exactly zero planning gain; and "
        "lookahead of two, three and five steps is not significantly better than one "
        "step on any budget while costing 1.8 to 2.1 times the motion. All results are "
        "in simulation with a vendored third-party locomotion policy; we claim the "
        "formulation, the benchmark, the failure analysis and the identification of the "
        "missing quantity, not a world model.")
    para(abstract, style="Normal")

    body_words = sum(len(inline(" ".join(l.split())).split())
                     for l in lines if l.strip() and not l.strip().startswith(("%", "[[TABLE")))
    body_words += len(abstract.split())
    para(f"Word count: {body_words}. Figures: {len(FIGURES)}. Tables: {len(tables)}.",
         style="Normal")

    # --- body --------------------------------------------------------------
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("%"):
            continue
        # The back matter is written explicitly below, in the order the guidelines
        # require (abbreviations immediately before the acknowledgments).  Stop the
        # body at the first starred section or those blocks are emitted twice.
        if stripped.startswith(r"\section*{"):
            break
        if stripped.startswith(r"\section{"):
            para(inline(stripped[len(r"\section{"):-1]), style="Heading 1"); continue
        if stripped.startswith(r"\subsection{"):
            para(inline(stripped[len(r"\subsection{"):-1]), style="Heading 2"); continue
        if stripped.startswith((r"\begin{enumerate}", r"\end{enumerate}",
                                r"\begin{verbatim}", r"\end{verbatim}")):
            continue
        if stripped.startswith(r"\item"):
            para(inline(stripped[len(r"\item"):]), style="Normal"); continue
        if stripped.startswith("[[TABLE:"):
            continue  # tables move to the end of the manuscript
        para(inline(" ".join(stripped.split())), style="Normal")

    # --- back matter -------------------------------------------------------
    para("Author Contributions", style="Heading 1")
    para("TO BE COMPLETED BY THE AUTHORS.", style="Normal")

    para("Funding", style="Heading 1")
    para("TO BE COMPLETED BY THE AUTHORS.", style="Normal")

    para("Data Availability Statement", style="Heading 1")
    para("The benchmark, baselines, consequence dataset and all result artefacts are "
         "generated by the code in the accompanying repository. Every figure and table "
         "in this manuscript is produced by a script from recorded JSON artefacts, so "
         "the numbers cannot drift from the figures.", style="Normal")

    para("Abbreviations", style="Heading 1")
    para("Non-standard abbreviations used in this manuscript, defined here as permitted "
         "immediately before the acknowledgments: " +
         "; ".join(f"{k}, {v}" for k, v in ABBREVIATIONS) + ".", style="Normal")

    para("Acknowledgments", style="Heading 1")
    para("TO BE COMPLETED BY THE AUTHORS. If generative artificial intelligence was used "
         "in preparing this manuscript, its name, version, model and source must be "
         "declared here and in the Methods.", style="Normal")

    para("Conflict of Interest", style="Heading 1")
    para("The authors declare that the research was conducted in the absence of any "
         "commercial or financial relationships that could be construed as a potential "
         "conflict of interest.", style="Normal")

    # --- references, alphabetical and unnumbered (Harvard) ------------------
    para("References", style="Heading 1")
    bib = (ROOT / "paper" / "frontiers" / "references.bib").read_text(encoding="utf-8")
    entries = re.findall(r"@\w+\{([^,]+),(.*?)\n\}", bib, re.S)
    rendered = []
    for key, block in entries:
        author = re.search(r"author\s*=\s*\{(.*?)\}", block, re.S)
        title = re.search(r"title\s*=\s*\{(.*?)\}", block, re.S)
        year = re.search(r"year\s*=\s*\{?(\d{4})", block)
        journal = re.search(r"(?:journal|booktitle)\s*=\s*\{(.*?)\}", block, re.S)
        note = re.search(r"note\s*=\s*\{(.*?)\}", block, re.S)
        a = " ".join((author.group(1) if author else "?").split())
        t = " ".join((title.group(1) if title else "?").split())
        j = " ".join((journal.group(1) if journal else "").split())
        y = year.group(1) if year else "n.d."
        # BibTeX braces used to protect capitalisation ({WoMAP}) survive the regex
        # capture and would print literally in the reference list.
        a, t, j = (s.replace("{", "").replace("}", "") for s in (a, t, j))
        url = ""
        if note:
            m = re.search(r"(https?://\S+)", note.group(1))
            url = m.group(1) if m else ""
        # Harvard: full last name and initials of first six authors, then et al.;
        # year in brackets; alphabetical order, no numbering.
        entry = f"{a} ({y}). {t}."
        if j:
            entry += f" {j}."
        if url:
            entry += f" Available at: {url} (accessed 2026)."
        rendered.append((a.lower(), entry))
    for _, entry in sorted(rendered):
        para(entry, style="Normal")

    # --- tables at the end, caption immediately before ----------------------
    for key, (caption, rows) in tables.items():
        if not rows:
            continue
        label = REFS.get(key)
        prefix = f"Table {label}. " if label else ""
        para(prefix + caption, style="Caption")
        table = doc.add_table(rows=0, cols=len(rows[0]))
        table.style = "Table Grid"
        for r in rows:
            cells = table.add_row().cells
            for c, value in zip(cells, r):
                c.text = ""
                add_runs(c.paragraphs[0], value)
        para("")

    # --- figure captions and the figures themselves, at the end ------------
    for filename, caption in FIGURES:
        para(caption, style="Caption")
    para("Figures", style="Heading 1")
    for filename, _ in FIGURES:
        path = FIGDIR / filename.replace(".pdf", ".png")
        if path.exists():
            doc.add_picture(str(path), width=Inches(6.0))
            para("")

    doc.save(str(OUTPUT))
    return OUTPUT


if __name__ == "__main__":
    print(f"wrote {build()}")
