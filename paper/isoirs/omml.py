"""Minimal OMML (Office Math) construction for the ISoIRS manuscript.

python-docx cannot write equations, and the official template's own equation is in
the Word 97-2003 Symbol-font encoding, which is not editable maths in a modern
Word.  The conference requires equations "in editable format from the equation
editor", so the display equations are built here as real OMML: Word opens them in
the equation editor, they re-flow with the column, and they are not images.

Only the primitives this manuscript needs are implemented - runs, sub/superscripts,
fractions, n-ary operators with limits, delimiters, and a lower limit for argmax.
"""
from __future__ import annotations

from docx.oxml.ns import qn
from lxml import etree

M = "http://schemas.openxmlformats.org/officeDocument/2006/math"
W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
NSMAP = {"m": M, "w": W}


def _m(tag: str):
    return etree.SubElement(etree.Element("dummy"), qn(f"m:{tag}"))


def el(tag: str, parent=None):
    node = etree.Element(qn(f"m:{tag}"), nsmap=NSMAP)
    if parent is not None:
        parent.append(node)
    return node


def run(text: str, italic: bool = True, parent=None):
    """A math run. Latin variables are italic; operators and digits are not."""
    node = el("r", parent)
    rpr = el("rPr", node)
    if italic:
        sty = etree.SubElement(rpr, qn("m:sty"))
        sty.set(qn("m:val"), "i")
    t = el("t", node)
    t.set(qn("xml:space"), "preserve")
    t.text = text
    return node


def sub(base, subscript, parent=None):
    node = el("sSub", parent)
    el("e", node).append(base)
    el("sub", node).append(subscript)
    return node


def sup(base, superscript, parent=None):
    node = el("sSup", parent)
    el("e", node).append(base)
    el("sup", node).append(superscript)
    return node


def subsup(base, subscript, superscript, parent=None):
    node = el("sSubSup", parent)
    el("e", node).append(base)
    el("sub", node).append(subscript)
    el("sup", node).append(superscript)
    return node


def frac(numerator, denominator, parent=None):
    node = el("f", parent)
    el("num", node).append(numerator)
    el("den", node).append(denominator)
    return node


def nary(chr_: str, body, sub_=None, sup_=None, parent=None, hide_limits: bool = False):
    node = el("nary", parent)
    pr = el("naryPr", node)
    c = etree.SubElement(pr, qn("m:chr")); c.set(qn("m:val"), chr_)
    lim = etree.SubElement(pr, qn("m:limLoc"))
    lim.set(qn("m:val"), "undOvr" if not hide_limits else "subSup")
    el("e", node).append(body)
    if sub_ is not None:
        el("sub", node).append(sub_)
    if sup_ is not None:
        el("sup", node).append(sup_)
    return node


def delim(inner, beg: str = "(", end: str = ")", parent=None):
    node = el("d", parent)
    pr = el("dPr", node)
    b = etree.SubElement(pr, qn("m:begChr")); b.set(qn("m:val"), beg)
    e = etree.SubElement(pr, qn("m:endChr")); e.set(qn("m:val"), end)
    el("e", node).append(inner)
    return node


def limlow(base, lower, parent=None):
    node = el("limLow", parent)
    el("e", node).append(base)
    el("lim", node).append(lower)
    return node


def seq(parts, parent=None):
    """Concatenate several elements into an ``m:e``-like container."""
    node = el("e", parent) if parent is not None else etree.Element(qn("m:oMath"),
                                                                   nsmap=NSMAP)
    for part in parts:
        node.append(part)
    return node


def omath(parts):
    node = etree.Element(qn("m:oMath"), nsmap=NSMAP)
    for part in parts:
        node.append(part)
    return node


def omath_para(parts):
    node = etree.Element(qn("m:oMathPara"), nsmap=NSMAP)
    node.append(omath(parts))
    return node


def r_(text: str) -> "etree._Element":
    return run(text, italic=True)


def op(text: str) -> "etree._Element":
    return run(text, italic=False)


# ---------------------------------------------------------------------------
# The two display equations of the manuscript
# ---------------------------------------------------------------------------

def equation_objective() -> list:
    """a*_{t:t+H} = argmax_{a_{t:t+H}} P(V_{t+H}=1 | O_t, M_t, a_{t:t+H})"""
    lhs = subsup(r_("a"), op("t:t+H"), op("*"))
    argmax = limlow(op("arg max"), r_("a") )
    inner = seq([
        r_("P"), delim(seq([
            sub(r_("V"), op("t+H")), op(" = 1 "), op("|"), op(" "),
            sub(r_("O"), op("t")), op(", "), sub(r_("M"), op("t")), op(", "),
            sub(r_("a"), op("t:t+H")),
        ])),
    ])
    rhs = seq([argmax, inner])
    return [lhs, op(" = "), rhs]


def equation_rsr() -> list:
    """r_i(K) = 1[ exists h <= K : V_{i,t+h} = 1 ],   RSR@K = (1/N) sum_i r_i(K)"""
    lhs = seq([sub(r_("r"), op("i")), delim(op("K"))])
    ind = seq([op("1"), delim(seq([
        op("\u2203"), op(" h \u2264 K : "),
        sub(r_("V"), op("i,t+h")), op(" = 1"),
    ]), beg="[", end="]")])
    rsr = seq([
        op("RSR@"), r_("K"), op(" = "),
        frac(op("1"), r_("N")),
        nary("\u2211", sub(r_("r"), op("i")), sub_=op("i=1"), sup_=r_("N")),
        delim(op("K")),
    ])
    return [lhs, op(" = "), ind, op(",      "), rsr]
