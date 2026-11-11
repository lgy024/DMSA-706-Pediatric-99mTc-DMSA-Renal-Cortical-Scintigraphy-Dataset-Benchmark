"""Text metrics: BLEU-4, ROUGE-L F1, CIDEr and token-overlap F1.

Tokenisation uses jieba when available and falls back to Chinese character
level plus alphanumeric runs, after full-width-to-half-width normalisation and
lowercasing. The same tokenizer must be used for every model in a comparison.

Metric definitions follow the paper:

* token-overlap F1 = 2 * sum_w min(n_A(w), n_B(w)) / (|A| + |B|)
* ROUGE-L F1     = 2 * LCS(A, B) / (|A| + |B|)
* BLEU-4         = corpus-level, equally weighted modified 1-4-gram
                   precisions, standard brevity penalty, no smoothing;
                   zero if any required precision or the hypothesis length is zero.
* CIDEr          = TF-IDF-weighted 1-4-gram cosine similarity, with document
                   frequencies computed over references in the same cohort.

Tokenization defaults to jieba word segmentation, which matches the reported
scores more closely than character-level tokens for Chinese reports. Set
``DMSA706_TOKENIZER=char`` (or call ``set_tokenizer("char")``) for a
character-level fallback. The tokenizer must be identical for all models in a
comparison.
"""

from __future__ import annotations

import math
import os
import re
from collections import Counter
from typing import Dict, Iterable, List, Sequence, Tuple

_TOKEN_RE = re.compile(r"[一-鿿]|[a-zA-Z]+|\d+(?:\.\d+)?")

try:  # optional dependency; character fallback keeps the package usable
    import jieba

    _HAS_JIEBA = True
except ImportError:  # pragma: no cover - depends on the environment
    jieba = None
    _HAS_JIEBA = False

_TOKENIZER = os.environ.get("DMSA706_TOKENIZER", "jieba" if _HAS_JIEBA else "char")
if _TOKENIZER == "jieba" and not _HAS_JIEBA:
    _TOKENIZER = "char"


def set_tokenizer(backend: str) -> None:
    """Switch between ``"auto"``, ``"jieba"`` and ``"char"`` tokenisation."""
    global _TOKENIZER
    if backend == "auto":
        _TOKENIZER = "jieba" if _HAS_JIEBA else "char"
        return
    if backend not in {"jieba", "char"}:
        raise ValueError(f"unknown tokenizer: {backend}")
    if backend == "jieba" and not _HAS_JIEBA:
        raise RuntimeError("jieba is not installed: pip install jieba")
    _TOKENIZER = backend


def get_tokenizer() -> str:
    return _TOKENIZER


def normalize(text: str) -> str:
    """Lowercase, strip, and convert full-width ASCII to half-width."""
    if text is None:
        return ""
    characters = []
    for ch in str(text):
        code = ord(ch)
        if code == 0x3000:
            characters.append(" ")
        elif 0xFF01 <= code <= 0xFF5E:
            characters.append(chr(code - 0xFEE0))
        else:
            characters.append(ch)
    return "".join(characters).lower().strip()


def tokenize(text: str) -> List[str]:
    normalized = normalize(text)
    if _TOKENIZER == "jieba" and _HAS_JIEBA:
        return [token for token in jieba.lcut(normalized) if token.strip()]
    return _TOKEN_RE.findall(normalized)


def _ngrams(tokens: Sequence[str], n: int) -> Counter:
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def token_overlap_f1(prediction: str, reference: str) -> float:
    pred, ref = tokenize(prediction), tokenize(reference)
    if not pred or not ref:
        return 0.0
    pred_counts, ref_counts = Counter(pred), Counter(ref)
    overlap = sum(min(count, ref_counts[token]) for token, count in pred_counts.items())
    return 2.0 * overlap / (len(pred) + len(ref))


def lcs_length(a: Sequence[str], b: Sequence[str]) -> int:
    if not a or not b:
        return 0
    previous = [0] * (len(b) + 1)
    for x in a:
        current = [0] * (len(b) + 1)
        for j, y in enumerate(b, start=1):
            if x == y:
                current[j] = previous[j - 1] + 1
            else:
                current[j] = max(previous[j], current[j - 1])
        previous = current
    return previous[-1]


def rouge_l_f1(prediction: str, reference: str) -> float:
    pred, ref = tokenize(prediction), tokenize(reference)
    if not pred or not ref:
        return 0.0
    return 2.0 * lcs_length(pred, ref) / (len(pred) + len(ref))


def bleu4(pairs: Iterable[Tuple[str, str]]) -> float:
    """Corpus-level BLEU-4. ``pairs`` is an iterable of (prediction, reference)."""
    matches = [0] * 4
    totals = [0] * 4
    hyp_length = 0
    ref_length = 0

    for prediction, reference in pairs:
        pred, ref = tokenize(prediction), tokenize(reference)
        hyp_length += len(pred)
        ref_length += len(ref)
        for n in range(1, 5):
            totals[n - 1] += max(len(pred) - n + 1, 0)
            if len(pred) < n:
                continue
            pred_counts = _ngrams(pred, n)
            ref_counts = _ngrams(ref, n)
            matches[n - 1] += sum(
                min(count, ref_counts[gram]) for gram, count in pred_counts.items()
            )

    if hyp_length == 0:
        return 0.0
    precisions = []
    for n in range(4):
        if totals[n] == 0:
            return 0.0
        precisions.append(matches[n] / totals[n])
    if any(p == 0.0 for p in precisions):
        return 0.0
    brevity = (
        1.0
        if hyp_length > ref_length
        else math.exp(1.0 - ref_length / float(hyp_length))
    )
    return brevity * math.exp(sum(math.log(p) for p in precisions) / 4.0)


def _tf_idf(ngrams: Counter, df: Counter, n_docs: int) -> Dict[tuple, float]:
    vector = {}
    for gram, count in ngrams.items():
        idf = math.log((n_docs + 1.0) / (df.get(gram, 0) + 1.0)) + 1.0
        vector[gram] = count * idf
    return vector


def _cosine(v1: Dict[tuple, float], v2: Dict[tuple, float]) -> float:
    if not v1 or not v2:
        return 0.0
    dot = sum(value * v2.get(gram, 0.0) for gram, value in v1.items())
    norm1 = math.sqrt(sum(value * value for value in v1.values()))
    norm2 = math.sqrt(sum(value * value for value in v2.values()))
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0
    return dot / (norm1 * norm2)


def cider(
    pairs: Sequence[Tuple[str, str]],
    max_n: int = 4,
    reduction: str = "sum",
    scale: float = 1.0,
) -> float:
    """CIDEr over a cohort.

    ``reduction="sum"`` sums the four n-gram cosines and can exceed 1, matching
    the paper's note that CIDEr may be above 1; ``reduction="mean"`` is the
    normalised cosine form in [0, 1]. ``scale`` multiplies the result for
    comparison with a different reporting convention.
    """
    if not pairs:
        return 0.0

    pred_tokens = [tokenize(pred) for pred, _ in pairs]
    ref_tokens = [tokenize(ref) for _, ref in pairs]
    n_docs = len(pairs)

    ref_ngrams = [
        [_ngrams(tokens, n) for tokens in ref_tokens] for n in range(1, max_n + 1)
    ]
    dfs = []
    for n in range(1, max_n + 1):
        df = Counter()
        for grams in ref_ngrams[n - 1]:
            df.update(grams.keys())
        dfs.append(df)

    per_pair = []
    for i in range(n_docs):
        score = 0.0
        for n in range(1, max_n + 1):
            pred_vec = _tf_idf(_ngrams(pred_tokens[i], n), dfs[n - 1], n_docs)
            ref_vec = _tf_idf(ref_ngrams[n - 1][i], dfs[n - 1], n_docs)
            score += _cosine(pred_vec, ref_vec)
        if reduction == "mean":
            score /= max_n
        elif reduction != "sum":
            raise ValueError(f"unknown CIDEr reduction: {reduction}")
        per_pair.append(score * scale)

    return sum(per_pair) / len(per_pair)


def evaluate_text_pairs(
    pairs: Sequence[Tuple[str, str]],
    cider_reduction: str = "sum",
    cider_scale: float = 1.0,
) -> Dict[str, float]:
    """Compute all four T1 metrics for already-aligned (pred, ref) pairs."""
    if not pairs:
        return {"BLEU4": 0.0, "ROUGE_L": 0.0, "CIDEr": 0.0, "F1_token": 0.0}
    n = len(pairs)
    return {
        "BLEU4": bleu4(pairs),
        "ROUGE_L": sum(rouge_l_f1(p, r) for p, r in pairs) / n,
        "CIDEr": cider(pairs, reduction=cider_reduction, scale=cider_scale),
        "F1_token": sum(token_overlap_f1(p, r) for p, r in pairs) / n,
    }
