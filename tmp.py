# -*- coding: utf-8 -*-


import re
import unicodedata
from typing import List, Tuple, Dict, Any


def norm(s: str) -> str:
    """Normalize string: NFKC, casefold, hyphen/underscore -> space, strip non-alnum except spaces."""
    if s is None:
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = s.casefold()
    s = re.sub(r"[-_]", " ", s)
    s = re.sub(r"[^0-9a-z ]+", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def camel_to_spaces(x: str) -> str:
    """Split CamelCase into space-separated tokens: 'MotionDevice' -> 'Motion Device'."""
    return re.sub(r"(?<!^)(?=[A-Z])", " ", x)


def levenshtein(a: str, b: str) -> int:
    """Levenshtein distance, iterative DP, O(min(m,n)) memory."""
    m, n = len(a), len(b)
    if m < n:
        a, b = b, a
        m, n = n, m
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        curr = [i] + [0] * n
        ca = a[i - 1]
        for j in range(1, n + 1):
            cb = b[j - 1]
            cost = 0 if ca == cb else 1
            curr[j] = min(prev[j] + 1,          # deletion
                          curr[j - 1] + 1,      # insertion
                          prev[j - 1] + cost)   # substitution
        prev = curr
    return prev[n]


def normalized_edit_distance(a: str, b: str) -> float:
    """Levenshtein(a,b) / max(len(a), len(b)); returns 0..1."""
    if not a and not b:
        return 0.0
    d = levenshtein(a, b)
    return d / max(len(a), len(b))


def build_lexicon(keywords: List[str]) -> List[Dict[str, Any]]:
    """
    Build a lexicon with original labels and variants.
    Returns list of dicts: {label_original, label_norm, canonical, variant}
    """
    lex = []
    for k in keywords:
        # canonical/original
        lex.append({
            "label_original": k,
            "label_norm": norm(k),
            "canonical": k,
            "variant": "raw"
        })
        # CamelCase spaced variant (if different)
        spaced = camel_to_spaces(k)
        if spaced != k:
            lex.append({
                "label_original": spaced,
                "label_norm": norm(spaced),
                "canonical": k,
                "variant": "camel_spaced"
            })
    return lex


def sentence_tokens(sentence: str) -> List[str]:
    """Simple whitespace tokenization preserving order; drops empties."""
    return [t for t in re.split(r"\s+", sentence.strip()) if t]


def generate_ngrams(tokens: List[str], max_n: int) -> List[Tuple[str, int, int]]:
    """
    Generate raw n-gram strings and their (start_idx, end_idx) token spans.
    end_idx is exclusive.
    """
    ngrams = []
    N = len(tokens)
    for i in range(N):
        for n in range(1, max_n + 1):
            j = i + n
            if j <= N:
                ngrams.append((" ".join(tokens[i:j]), i, j))
    return ngrams


def match_type(raw_ngram: str, label_original: str) -> str:
    """EXACT if raw equals label; PUNCT_NORM if norm equals; else APPROX."""
    if raw_ngram == label_original:
        return "EXACT"
    if norm(raw_ngram) == norm(label_original):
        return "PUNCT_NORM"
    return "APPROX"


def match_im_entities(
    sentence: str,
    lexicon: List[Dict[str, Any]],
    tau: float = 0.20,
    max_ngram: int = None
) -> List[Dict[str, Any]]:
    """
    Match IM-like entities from lexicon in a sentence with tolerance tau.
    Returns list of resolved matches per n-gram span.
    """
    raw_tokens = sentence_tokens(sentence)
    print("raw_tokens:",raw_tokens)
    # choose max_ngram based on longest lexicon label (in tokens)
    if max_ngram is None:
        max_len = 1
        for L in lexicon:
            max_len = max(max_len, len(norm(L["label_original"]).split()))
        max_ngram = min(max_len, max(1, len(raw_tokens)))  # cap at sentence length
    print("ngram:", max_ngram)
    ngrams = generate_ngrams(raw_tokens, max_ngram)

    # collect candidates
    C = []  # (span, raw_ngram, canonical, variant, label_original, sim, mtype, label_len)
    for raw, i, j in ngrams:
        n_raw = norm(raw)
        for L in lexicon:
            d = normalized_edit_distance(n_raw, L["label_norm"])
            if d <= tau:
                sim = 1.0 - d
                mtype = match_type(raw, L["label_original"])
                C.append({
                    "span": (i, j),
                    "mention": raw,
                    "canonical": L["canonical"],
                    "variant": L["variant"],
                    "label_original": L["label_original"],
                    "similarity": sim,
                    "match_type": mtype,
                    "label_len": len(L["label_norm"])
                })

    if not C:
        return []

    # resolve conflicts per span: PRIORITY(EXACT > PUNCT_NORM > APPROX), then similarity, then longer label
    priority = {"EXACT": 3, "PUNCT_NORM": 2, "APPROX": 1}
    by_span: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}
    for c in C:
        by_span.setdefault(c["span"], []).append(c)

    resolved = []
    for span, cand_list in by_span.items():
        cand_list.sort(key=lambda x: (priority[x["match_type"]], x["similarity"], x["label_len"]), reverse=True)
        best = cand_list[0]
        best_out = {
            "span_tokens": span,
            "mention": best["mention"],
            "keyword": best["canonical"],
            "matched_variant": best["variant"],
            "label_used": best["label_original"],
            "similarity": round(best["similarity"], 3),
            "match_type": best["match_type"]
        }
        resolved.append(best_out)

    # stable order by start index
    resolved.sort(key=lambda x: (x["span_tokens"][0], -x["span_tokens"][1]))
    return resolved


if __name__ == "__main__":
    # Example usage (your sample):
    keywords = ["MotionDevice", "hasComponent"]
    sentences = [
        "MotionDevice is good",
        "Motiondevice is good",
        "MotionDevce is CAT"
    ]

    lex = build_lexicon(keywords)
    TAU = 0.20  # <= 20% normalized edit distance

    for s in sentences:
        matches = match_im_entities(s, lex, tau=TAU)
        print(f'\nSentence: "{s}"')
        if not matches:
            print("  -> no matches")
            continue
        for m in matches:
            i, j = m["span_tokens"]
            print(f'  -> {m["mention"]}  ==>  {m["keyword"]} '
                  f'[{m["match_type"]}, sim={m["similarity"]}]  '
                  f'(span={i}:{j}, variant={m["matched_variant"]})')
