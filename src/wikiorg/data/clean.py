"""Cleaning of MediaWiki ``prop=extracts&explaintext=1`` output.

Steps (in this order):
1. cut the text at the first level-2 heading that starts the reference apparatus
   (See also, References, Notes, External links, Further reading, Bibliography, Sources, ...);
2. strip TextExtracts math residue ``{\\displaystyle ...}`` / ``{\\textstyle ...}`` with a brace
   matcher (nested braces are common: ``{\\displaystyle x^{2}}``);
3. convert ``== Heading ==`` to Markdown ``## Heading`` (level = number of ``=``);
4. normalise whitespace (no trailing spaces, at most one blank line).
Documents shorter than ``MIN_CHARS`` after cleaning are dropped by the caller.
"""
from __future__ import annotations

import re

MIN_CHARS = 1200
CUT_HEADINGS = {
    "see also", "references", "notes", "external links", "further reading", "bibliography",
    "sources", "citations", "footnotes", "notes and references", "references and notes",
    "notes and citations", "external links and references", "literature", "works cited",
    "reference", "footnotes and references", "references and further reading",
}
_HEADING_RE = re.compile(r"^(={2,6})\s*(.*?)\s*\1\s*$", re.M)
_MATH_START_RE = re.compile(r"\{\\(?:displaystyle|textstyle|scriptstyle|scriptscriptstyle)\b")


_DUMP_LINE_RE = re.compile(r"^[ \t]*$|^[ \t]+\S{1,24}$")   # one MathML token per indented line
_INVISIBLE_RE = re.compile("[⁡⁢⁣⁤]")
MAX_INLINE_MATH = 120


def _match_braces(text: str, start: int) -> tuple[int, bool]:
    """Index just past the brace group opened at ``start``; ``balanced`` False -> cut at line end."""
    depth = 0
    i = start
    while i < len(text):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i + 1, True
        elif ch == "\n":
            break
        i += 1
    nl = text.find("\n", start)
    return (len(text) if nl < 0 else nl), False


def strip_displaystyle(text: str) -> str:
    """Remove TextExtracts math residue.

    A formula appears as a block of indented one-token-per-line MathML text followed by the
    annotation ``{\\displaystyle <latex>}`` (brace-matched; an unbalanced one is cut at the end of
    its line).  The whole block is replaced by the annotation's inner LaTeX when it is short
    (<= MAX_INLINE_MATH chars), so "FFTs with O(n\\log n) complexity" stays readable; longer
    formulas are dropped.  Invisible operator characters (U+2061..U+2064) are removed."""
    out = []
    pos = 0
    while True:
        m = _MATH_START_RE.search(text, pos)
        if not m:
            out.append(text[pos:])
            break
        end, balanced = _match_braces(text, m.start())
        inner = text[m.end():end - 1].strip() if balanced else ""
        # backward over the MathML dump lines that precede the annotation line
        line_start = max(pos, text.rfind("\n", pos, m.start()) + 1)
        bstart = line_start
        while bstart > pos:
            prev_start = max(pos, text.rfind("\n", pos, bstart - 1) + 1)
            if prev_start >= bstart:
                break
            if _DUMP_LINE_RE.match(text[prev_start:bstart - 1]):
                bstart = prev_start
            else:
                break
        prefix = text[line_start:m.start()]
        if bstart < line_start:  # dump lines were consumed: the formula continues the previous line
            head = text[pos:bstart]
            if head.endswith("\n") and not head.endswith("\n\n"):
                head = head[:-1] + " "
            if prefix.strip():
                head += prefix.strip() + " "
        else:
            head = text[pos:m.start()]
        # forward over whitespace-only lines that follow the annotation
        fend = end
        newlines = 0
        while True:
            nl = text.find("\n", fend)
            seg = text[fend:] if nl < 0 else text[fend:nl]
            if seg.strip() == "" and nl >= 0:
                fend = nl + 1
                newlines += 1
                continue
            break
        repl = f" {inner} " if inner and len(inner) <= MAX_INLINE_MATH else " "
        if newlines >= 3:  # a real paragraph break followed the formula
            repl = repl.rstrip() + "\n\n"
        out.append(head.rstrip(" \t") + repl)
        pos = fend
    s = _INVISIBLE_RE.sub("", "".join(out))
    s = re.sub(r"[ \t]{2,}", " ", s)
    s = re.sub(r" +([,.;:)])", r"\1", s)
    return s


def cut_trailing_sections(text: str) -> str:
    for m in _HEADING_RE.finditer(text):
        if len(m.group(1)) == 2 and m.group(2).strip().casefold() in CUT_HEADINGS:
            return text[: m.start()]
    return text


def convert_headings(text: str) -> str:
    def repl(m: re.Match) -> str:
        level = len(m.group(1))
        return "#" * level + " " + m.group(2).strip()
    return _HEADING_RE.sub(repl, text)


def normalise_whitespace(text: str) -> str:
    lines = [ln.rstrip() for ln in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    s = "\n".join(lines)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def clean_extract(text: str) -> str:
    s = cut_trailing_sections(text)
    s = strip_displaystyle(s)
    s = convert_headings(s)
    s = normalise_whitespace(s)
    # drop empty trailing headings (a section whose body was entirely math/cut)
    s = re.sub(r"(?:\n#{2,6} [^\n]*)+\s*$", "", s).strip()
    return s
