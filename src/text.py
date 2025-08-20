"""
Lightweight text utilities for server-side processing.

Currently includes a real-time friendly sentence splitter used to
classify metrics as per-sentence vs block.
"""

from typing import List

def split_sentences(text: str) -> List[str]:
    """Split text into sentences with simple real-time friendly rules.

    - Ends on any run of '?' or '!' (e.g., ?, !, ?!, !!!)
    - Ends on a single '.' that is not part of an ellipsis '...'
    - Includes any trailing quotes/brackets immediately after the end mark
    - Skips whitespace after a sentence end
    """
    if not isinstance(text, str):
        return []
    s = text.strip()
    if not s:
        return []
    out: List[str] = []
    i, n, last = 0, len(s), 0
    trailing = ")]}\"'”"
    while i < n:
        ch = s[i]
        if ch in ("?", "!"):
            j = i
            while j < n and s[j] in ("?", "!"):
                j += 1
            end = j
            while end < n and s[end] in trailing:
                end += 1
            seg = s[last:end].strip()
            if seg:
                out.append(seg)
            i = end
            while i < n and s[i].isspace():
                i += 1
            last = i
            continue
        if ch == ".":
            if s[i : i + 3] == "...":
                i += 3
                continue
            end = i + 1
            while end < n and s[end] in trailing:
                end += 1
            seg = s[last:end].strip()
            if seg:
                out.append(seg)
            i = end
            while i < n and s[i].isspace():
                i += 1
            last = i
            continue
        i += 1
    tail = s[last:].strip()
    if tail:
        out.append(tail)
    return out


