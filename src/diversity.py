"""Text-diversity metrics used by evaluation.

Design goals:
  * Single canonical word-tokenization (normalized whitespace split) used
    everywhere so that distinct-n, vocab size, MTLD, and Yule's K are
    directly comparable across model families and across experiments.
  * Optional prompt-prefix stripping using the *generating* model's
    tokenizer, so diversity reflects only the generated continuation
    rather than a fixed prompt that never changes across generations.
  * Sample-size-invariant complementary metrics (MTLD, Yule's K,
    vocab/sqrt(N)) so that 'distinct-2' style numbers can be cross-checked.
  * Standard-library metrics with an optional external tokenizer.

The exact word definition (one source of truth):
    text -> lowercase -> whitespace-split -> for each token,
    strip leading/trailing punctuation/_; drop empty tokens.

"""

from __future__ import annotations

import math
import re
from statistics import median
from collections import Counter
from typing import Iterable, Sequence

# ---------------------------------------------------------------------------
# Word tokenization (the canonical "word" definition)
# ---------------------------------------------------------------------------
_PUNCT_STRIP = re.compile(r"^[\W_]+|[\W_]+$", re.UNICODE)


def normalize_words(text: str) -> list[str]:
    """Canonical word-tokenizer used for all diversity metrics.

    Lowercases, splits on whitespace, strips leading/trailing punctuation
    (Unicode-aware), and drops empty tokens. Designed to be model-agnostic.
    """
    if not text:
        return []
    out: list[str] = []
    for raw in text.lower().split():
        w = _PUNCT_STRIP.sub("", raw)
        if w:
            out.append(w)
    return out


# ---------------------------------------------------------------------------
# Prompt-prefix stripping (uses the generating model's tokenizer)
# ---------------------------------------------------------------------------
def strip_prompt_prefix(text: str, tokenizer, n_tokens: int) -> str:
    """Re-tokenize `text` with the generating model's tokenizer, drop the
    first `n_tokens`, and decode the remainder. Returns "" if the text is
    shorter than the prompt prefix.

    This recovers the *generated continuation* from a synthetic example of
    the form (prompt + continuation) that was saved during data generation.
    """
    if not text or n_tokens <= 0 or tokenizer is None:
        return text or ""
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) <= n_tokens:
        return ""
    return tokenizer.decode(ids[n_tokens:], skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Diversity metrics (all operate on lists of normalized-word lists)
# ---------------------------------------------------------------------------
def distinct_n(tokenized_texts: Sequence[Sequence[str]], ns=(1, 2, 3)) -> dict:
    """Distinct-n ratios on the canonical word stream.

    Pooled across examples (i.e. n-grams from all examples are unioned
    before counting). This matches the convention used in the original
    Li et al. distinct-n metric and what the existing eval scripts did
    (modulo the new word definition + prompt stripping).
    """
    out: dict = {}
    for n in ns:
        all_ng: list[tuple] = []
        for toks in tokenized_texts:
            if len(toks) < n:
                continue
            all_ng.extend(tuple(toks[i : i + n]) for i in range(len(toks) - n + 1))
        if all_ng:
            out[f"distinct_{n}"] = len(set(all_ng)) / len(all_ng)
            out[f"unique_{n}grams"] = len(set(all_ng))
            out[f"total_{n}grams"] = len(all_ng)
        else:
            out[f"distinct_{n}"] = 0.0
            out[f"unique_{n}grams"] = 0
            out[f"total_{n}grams"] = 0
    return out


def vocab_per_sqrt_n(tokenized_texts: Sequence[Sequence[str]]) -> dict:
    """Herdan-style size-corrected vocabulary measure.

    Returns the unique vocabulary size, total token count, raw type-token
    ratio, and vocab / sqrt(total_tokens) which is approximately invariant
    to corpus size under Heaps' law.
    """
    counter: Counter = Counter()
    total = 0
    for toks in tokenized_texts:
        counter.update(toks)
        total += len(toks)
    vocab = len(counter)
    return {
        "vocab_size": vocab,
        "num_tokens": total,
        "type_token_ratio": (vocab / total) if total else 0.0,
        "vocab_per_sqrt_n": (vocab / math.sqrt(total)) if total else 0.0,
    }


def yule_k(tokenized_texts: Sequence[Sequence[str]]) -> dict:
    """Yule's K characteristic. Lower = more diverse vocabulary.

    K = 10000 * (sum_i (i^2 * V_i) - N) / N^2
    where V_i is the number of types occurring exactly i times and N is the
    total token count. Yule's K is approximately corpus-size invariant.
    """
    counter: Counter = Counter()
    for toks in tokenized_texts:
        counter.update(toks)
    n = sum(counter.values())
    if n == 0:
        return {"yule_k": 0.0}
    freqs_of_freqs: Counter = Counter(counter.values())
    s2 = sum((i * i) * v_i for i, v_i in freqs_of_freqs.items())
    k = 10000.0 * (s2 - n) / (n * n)
    return {"yule_k": k}


def mtld(tokenized_texts: Sequence[Sequence[str]], threshold: float = 0.72) -> dict:
    """Measure of Textual Lexical Diversity (McCarthy & Jarvis 2010).

    Concatenates all texts, then walks forward maintaining running TTR;
    every time TTR falls to `threshold` a new "factor" starts and the
    counters reset. MTLD = total_tokens / total_factors. Forward and
    backward passes are averaged. Robust to corpus length.
    """
    stream: list[str] = []
    for toks in tokenized_texts:
        stream.extend(toks)
    if not stream:
        return {"mtld": 0.0}

    def _one_direction(tokens: list[str]) -> float:
        n_total = len(tokens)
        if n_total == 0:
            return 0.0
        factors = 0.0
        types: set = set()
        token_count = 0
        for tok in tokens:
            types.add(tok)
            token_count += 1
            if token_count > 0:
                ttr = len(types) / token_count
                if ttr <= threshold:
                    factors += 1
                    types = set()
                    token_count = 0
        if token_count > 0:
            ttr_last = len(types) / token_count if token_count > 0 else 1.0
            partial = (1 - ttr_last) / (1 - threshold) if (1 - threshold) > 0 else 0.0
            factors += max(0.0, min(1.0, partial))
        return n_total / factors if factors > 0 else float("inf")

    fwd = _one_direction(stream)
    bwd = _one_direction(list(reversed(stream)))
    if math.isinf(fwd) and math.isinf(bwd):
        score = float("inf")
    elif math.isinf(fwd):
        score = bwd
    elif math.isinf(bwd):
        score = fwd
    else:
        score = (fwd + bwd) / 2.0
    return {"mtld": score}


def length_stats(tokenized_texts: Sequence[Sequence[str]]) -> dict:
    """Per-example length stats (in normalized words). Useful to separate
    the 'verbosity' confound from genuine diversity collapse: a model that
    rambles longer at later generations will look less diverse on
    pooled distinct-n even with constant per-example diversity."""
    lengths = [len(t) for t in tokenized_texts]
    if not lengths:
        return {"mean_length_words": 0.0, "median_length_words": 0,
                "min_length_words": 0, "max_length_words": 0,
                "num_continuations": 0}
    return {
        "mean_length_words": sum(lengths) / len(lengths),
        "median_length_words": median(lengths),
        "min_length_words": min(lengths),
        "max_length_words": max(lengths),
        "num_continuations": len(lengths),
    }


# ---------------------------------------------------------------------------
# Top-level convenience
# ---------------------------------------------------------------------------
def compute_diversity(
    texts: Iterable[str],
    *,
    tokenizer=None,
    strip_prompt_tokens: int = 0,
    sample_limit: int | None = None,
    include_mtld: bool = True,
    include_yule: bool = True,
) -> dict:
    """Compute the full unified diversity panel on a list of texts.

    Steps:
      1. Optionally truncate to `sample_limit` examples (deterministic, head).
      2. If `tokenizer` and `strip_prompt_tokens` > 0, strip the first
         `strip_prompt_tokens` *model* tokens from each text, then keep
         only the continuation.
      3. Apply the canonical word normalization to every continuation.
      4. Compute distinct-{1,2,3}, vocab / sqrt(N), Yule's K, MTLD, and
         per-example length stats.

    Returned dict uses a flat schema so it can be merged into eval JSONs
    or summary rows directly.
    """
    texts = list(texts)
    if sample_limit is not None and len(texts) > sample_limit:
        texts = texts[:sample_limit]

    if tokenizer is not None and strip_prompt_tokens > 0:
        continuations = [strip_prompt_prefix(t, tokenizer, strip_prompt_tokens)
                         for t in texts]
    else:
        continuations = list(texts)

    tokenized = [normalize_words(t) for t in continuations]
    tokenized = [t for t in tokenized if t]

    out: dict = {}
    out.update(distinct_n(tokenized))
    out.update(vocab_per_sqrt_n(tokenized))
    if include_yule:
        out.update(yule_k(tokenized))
    if include_mtld:
        out.update(mtld(tokenized))
    out.update(length_stats(tokenized))
    out["diversity_sample_limit"] = sample_limit
    out["diversity_strip_prompt_tokens"] = strip_prompt_tokens
    out["diversity_word_definition"] = "lower+wssplit+strip_punct"
    out["num_examples_after_strip"] = len(tokenized)
    out["num_examples_input"] = len(texts)
    return out
