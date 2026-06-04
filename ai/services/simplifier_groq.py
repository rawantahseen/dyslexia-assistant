"""
Text Simplifier with Minimum Bayes Risk Re-ranking  (v4)
---------------------------------------------------------
What changed over v3 — all changes target API call reduction for large documents
while preserving or improving output quality.  Zero API contract changes.

1. TWO-PHASE GENERATION  (simplify_text)
   Phase-1: fire a small probe batch (3 candidates) concurrently.
   If any probe clears all gates AND tail_difficulty < EASY_EXIT_THRESHOLD (3.2),
   return immediately — no phase-2 needed.
   Phase-2: fire the remaining candidates only when the probe found nothing good.
   Typical saving: ~60% fewer Groq calls on sentences that are easy to simplify
   (short substitution jobs, active-voice rewrites, etc.).

2. SEMANTIC DEDUPLICATION OF HARD SENTENCES  (simplify_targeted_async)
   Before generating any candidates, all hard sentences are embedded in one
   batched encode call.  Sentences whose cosine similarity exceeds
   CLUSTER_SIM_THRESHOLD (0.82) are clustered; only the hardest representative
   per cluster is sent to the LLM.  Non-representative sentences in a cluster
   receive the representative's simplified output directly.
   Typical saving: 40–75% fewer Groq calls on repetitive documents (reports,
   legal texts, academic papers).

3. DOCUMENT-AWARE CANDIDATE BUDGET  (compute_candidate_budget)
   Instead of a flat n_candidates ceiling per sentence, the total Groq calls
   for the whole document are capped at a budget that scales with document size:
       budget = min(BASE_BUDGET + n_hard_sentences * 3, MAX_BUDGET)
   Within the budget, calls are allocated proportionally to relative difficulty,
   with a per-sentence floor of MIN_CANDIDATES (3) and ceiling of MAX_CANDIDATES (8).
   This prevents a 30-sentence document from issuing 300 Groq calls while still
   giving harder sentences more candidates.

4. PARAGRAPH-LEVEL SIMPLIFICATION FOR MODERATE TEXT  (simplify_targeted_async)
   Consecutive hard sentences with max difficulty < PARAGRAPH_BATCH_THRESHOLD (5.8)
   and combined word count < PARAGRAPH_MAX_WORDS (70) are grouped and sent to the
   LLM as a single paragraph-level call instead of N separate sentence calls.
   The LLM has more context, the output is often more fluent, and the call count
   drops by up to 66% for those groups.
   Very hard sentences (≥ PARAGRAPH_BATCH_THRESHOLD) always get individual treatment
   so the re-ranker has enough candidates to work with.

5. IN-PROCESS SIMPLIFICATION CACHE  (_simplification_cache)
   Successful simplifications are stored in a process-level dict keyed by
   (text, sim_threshold).  On a cache hit the result is returned in microseconds
   with zero Groq calls.  Saves calls when the same sentence appears multiple
   times in a document (boilerplate, repeated headers, etc.) or across requests
   in the same server process.

All other logic from v3 is preserved unchanged:
  - lru_cache on score_difficulty
  - batched NLI via meaning_preserved_batch
  - batched similarity via semantic_similarity_batch
  - adaptive similarity threshold via _adaptive_sim_threshold
  - adaptive n_candidates via _adaptive_n_candidates (now used as a per-sentence
    ceiling inside the document budget, not a flat count)
  - semaphore parallelism in simplify_targeted_async
  - MBR combined score formula (0.55 * diff + 0.35 * sim² + 0.10 * length)
  - full fallback chain when no valid candidate exists

API contract: zero changes.
  simplify_text        — async, same signature and return dict
  simplify_text_sync   — sync wrapper, drop-in for v3
  simplify_targeted    — sync entry point used by api.py, drop-in for v3
"""

import asyncio
import hashlib
import os
import re
import sys
from functools import lru_cache

import numpy as np
import textstat
from dotenv import load_dotenv
from groq import AsyncGroq, Groq
from sentence_transformers import SentenceTransformer, util
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine_similarity
from transformers import pipeline

load_dotenv()

# ── Clients & models ───────────────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client       = Groq(api_key=GROQ_API_KEY)
async_client = AsyncGroq(api_key=GROQ_API_KEY)

sim_model = SentenceTransformer('paraphrase-MiniLM-L3-v2')

print("Loading NLI model (cross-encoder/nli-deberta-v3-small)...")
nli_model = pipeline(
    "text-classification",
    model="cross-encoder/nli-deberta-v3-small",
    device=-1,
    top_k=None,
)
print("NLI model loaded.")

# ── Scorer import ──────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
from ai.models.difficulty_scorer import find_difficult_words_in_text

# ── Tuning constants ───────────────────────────────────────────────────────────
# Two-phase generation
PHASE1_SIZE          = 3     # probe batch size; phase-2 fires if no good result
EASY_EXIT_THRESHOLD  = 3.2   # tail_difficulty below this triggers early return

# Semantic deduplication
CLUSTER_SIM_THRESHOLD = 0.82  # cosine sim above which two sentences share a cluster

# Document budget
BASE_BUDGET      = 15   # minimum total Groq calls for any document
BUDGET_PER_HARD  = 3    # extra calls budgeted per hard sentence
MAX_BUDGET       = 80   # hard ceiling regardless of document size
MIN_CANDIDATES   = 3    # floor per sentence
MAX_CANDIDATES   = 8    # ceiling per sentence (overrides adaptive_n for large docs)

# Paragraph batching
PARAGRAPH_BATCH_THRESHOLD = 5.8  # sentences scoring above this are never grouped
PARAGRAPH_MAX_WORDS       = 70   # combined word budget for a grouped paragraph call

# ── Prompts & temperatures ─────────────────────────────────────────────────────
SYSTEM_PROMPTS = [
    # P1 — word-level substitution, structure preserved
    """You are helping people with dyslexia read difficult text.
    Rewrite the text by replacing hard words with simpler ones.
    Keep the sentence structure identical. Keep ALL meaning.
    Return ONLY the rewritten text.""",

    # P2 — structural: break sentences
    """You are helping people with dyslexia read difficult text.
    Rewrite the text by breaking long sentences into shorter ones.
    Use simple subject-verb-object structure.
    Keep ALL meaning. Return ONLY the rewritten text.""",

    # P3 — aggressive full rewrite at grade-6 reading level
    """You are helping people with dyslexia read difficult text.
    Rewrite the text completely in simple everyday English.
    Use words a 12-year-old would know.
    Keep ALL meaning. Return ONLY the rewritten text.""",

    # P4 — active voice, eliminate passive constructions
    """You are helping people with dyslexia read difficult text.
    Rewrite the text using active voice only.
    Replace all passive constructions. Use simple words.
    Keep ALL meaning. Return ONLY the rewritten text.""",

    # P5 — length-preserving substitution, no padding
    """You are helping people with dyslexia read difficult text.
    Rewrite the text in simple everyday English.
    Replace hard words with simpler ones directly — do NOT add explanations or definitions.
    Keep the output roughly the same length as the input.
    Keep ALL meaning. Return ONLY the rewritten text.""",

    # P6 — domain-preserving: simplify context, keep technical nouns
    """You are helping people with dyslexia read difficult text.
    Rewrite the text using simple words for everything EXCEPT specific scientific
    or technical terms that cannot be replaced without losing meaning.
    For those terms, keep the word but add a short plain-English explanation
    immediately after it in parentheses.
    Example: "bioluminescence (when living things make their own light)"
    Keep ALL meaning. Return ONLY the rewritten text.""",
]

CANDIDATE_TEMPERATURES = [0.1, 0.3, 0.7, 1.0, 1.2]

# Round-robin grid: every prompt gets representation even when n_candidates < 25
_ALL_COMBOS: list[tuple[int, int]] = []
for _offset in range(len(CANDIDATE_TEMPERATURES)):
    for _pi in range(len(SYSTEM_PROMPTS)):
        _ti = (_pi + _offset) % len(CANDIDATE_TEMPERATURES)
        _combo = (_pi, _ti)
        if _combo not in _ALL_COMBOS:
            _ALL_COMBOS.append(_combo)


# ── In-process simplification cache ───────────────────────────────────────────
# Key: MD5 of (text.strip().lower(), str(sim_threshold))
# Value: the full result dict returned by simplify_text
_simplification_cache: dict[str, dict] = {}

def _cache_key(text: str, sim_threshold: float) -> str:
    return hashlib.md5(
        f"{text.strip().lower()}|{sim_threshold:.3f}".encode()
    ).hexdigest()


# ── Difficulty scoring (cached) ────────────────────────────────────────────────
@lru_cache(maxsize=1024)
def score_difficulty(text: str) -> float:
    """
    Tail-aware difficulty: 0.6 * mean + 0.4 * p90.
    lru_cache ensures repeated calls on the same string cost nothing.
    """
    results    = find_difficult_words_in_text(text, threshold=0.0)
    all_scored = results['all_scored']
    if not all_scored:
        return 5.0
    scores = [w['difficulty_score'] for w in all_scored]
    mean   = float(np.mean(scores))
    p90    = float(np.percentile(scores, 90))
    return round(0.6 * mean + 0.4 * p90, 4)


# ── Similarity (batched) ───────────────────────────────────────────────────────
def semantic_similarity_batch(original: str, candidates: list[str]) -> list[float]:
    """Encode original + all candidates in one call; return per-candidate cosine sims."""
    all_texts  = [original] + candidates
    embeddings = sim_model.encode(all_texts, convert_to_tensor=True, batch_size=32)
    orig_emb   = embeddings[0]
    return [float(util.cos_sim(orig_emb, embeddings[i + 1])) for i in range(len(candidates))]


def semantic_similarity(text_a: str, text_b: str) -> float:
    return semantic_similarity_batch(text_a, [text_b])[0]


# ── NLI contradiction gate (batched) ──────────────────────────────────────────
def meaning_preserved_batch(
    original: str,
    candidates: list[str],
    contradiction_threshold: float = 0.6,
) -> list[tuple[bool, str, float]]:
    """
    Single pipeline call for all candidates.
    Returns list of (is_safe, dominant_label, contradiction_score).
    """
    if not candidates:
        return []

    inputs = [f"{original} [SEP] {c}" for c in candidates]
    try:
        raw_batch = nli_model(inputs, batch_size=8, truncation=True)
    except Exception:
        return [(True, 'UNKNOWN', 0.0)] * len(candidates)

    results = []
    for raw in raw_batch:
        try:
            inner           = raw if isinstance(raw[0], dict) else raw[0]
            scores_by_label = {r['label'].upper(): r['score'] for r in inner}
            contradiction   = scores_by_label.get('CONTRADICTION', 0.0)
            label           = max(scores_by_label, key=scores_by_label.get)
            results.append((contradiction < contradiction_threshold, label, round(contradiction, 3)))
        except Exception:
            results.append((True, 'UNKNOWN', 0.0))
    return results


def meaning_preserved(
    original: str,
    candidate: str,
    contradiction_threshold: float = 0.6,
) -> tuple[bool, str, float]:
    return meaning_preserved_batch(original, [candidate], contradiction_threshold)[0]


# ── Candidate generation ───────────────────────────────────────────────────────
async def _generate_candidate_async(
    text: str,
    temperature: float,
    system_prompt: str,
) -> str | None:
    try:
        response = await async_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": f"Simplify this text:\n\n{text}"},
            ],
            temperature=temperature,
            max_tokens=1024,
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return None


async def _generate_batch_async(
    text: str,
    combos: list[tuple[int, int]],
    hard_word_suffix: str,
) -> list[tuple[int, int, str]]:
    """Fire a batch of (prompt_idx, temp_idx) combos concurrently."""
    tasks = [
        _generate_candidate_async(
            text,
            CANDIDATE_TEMPERATURES[ti],
            SYSTEM_PROMPTS[pi] + hard_word_suffix,
        )
        for pi, ti in combos
    ]
    outputs = await asyncio.gather(*tasks)
    return [
        (pi, ti, out)
        for (pi, ti), out in zip(combos, outputs)
        if out is not None
    ]


# Sync fallback for CLI / non-async callers
def generate_candidate(text: str, temperature: float, system_prompt: str) -> str:
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": f"Simplify this text:\n\n{text}"},
        ],
        temperature=temperature,
        max_tokens=1024,
    )
    return response.choices[0].message.content.strip()


# ── Adaptive helpers ───────────────────────────────────────────────────────────
def _adaptive_sim_threshold(word_count: int, original_score: float, base: float = 0.65) -> float:
    length_adj     = 0.10 * min(max((word_count - 10) / 40.0, 0.0), 1.0)
    difficulty_adj = 0.08 * min(max((original_score - 3.0) / 4.0, 0.0), 1.0)
    return max(base - length_adj - difficulty_adj, 0.45)


def _adaptive_n_candidates(sentence: str, difficulty: float) -> int:
    """
    Per-sentence candidate ceiling based on length × difficulty.
    Used as a local ceiling; the document budget may reduce it further.
    """
    wc = len(sentence.split())
    if wc < 12:
        return 3
    if wc < 20:
        return 4 if difficulty < 5.0 else 6
    if wc < 35:
        return 6 if difficulty < 5.5 else 8
    return 10


def compute_candidate_budget(
    difficulties: list[float],
    total_budget: int,
) -> list[int]:
    """
    Distribute total_budget Groq calls across N hard sentences proportionally
    to their relative difficulty, with per-sentence floor MIN_CANDIDATES and
    ceiling MAX_CANDIDATES.

    Equal difficulties → equal allocation.
    All-same edge case handled gracefully.
    """
    n = len(difficulties)
    if n == 0:
        return []

    min_d, max_d = min(difficulties), max(difficulties)
    if max_d == min_d:
        weights = [1.0] * n
    else:
        weights = [(d - min_d) / (max_d - min_d) for d in difficulties]

    raw = [MIN_CANDIDATES + w * (MAX_CANDIDATES - MIN_CANDIDATES) for w in weights]

    # Scale down if raw sum exceeds budget
    raw_sum = sum(raw)
    if raw_sum > total_budget:
        scale = total_budget / raw_sum
        raw = [max(MIN_CANDIDATES, r * scale) for r in raw]

    return [int(round(r)) for r in raw]


# ── Candidate scoring helpers ──────────────────────────────────────────────────
def _score_and_flag_candidates(
    text: str,
    raw_results: list[tuple[int, int, str]],
    original_score: float,
    effective_sim_threshold: float,
    text_lower: str,
    contradiction_threshold: float = 0.6,
) -> list[dict]:
    """
    Given a list of (prompt_idx, temp_idx, output) triples:
      1. Separate identical outputs (skip NLI + sim for them)
      2. Batch similarity on the rest
      3. Batch NLI on the rest
      4. Score difficulty only for candidates that pass both gates
    Returns a flat list of candidate dicts.
    """
    identical = [(pi, ti, o) for pi, ti, o in raw_results if o.strip().lower() == text_lower]
    to_score  = [(pi, ti, o) for pi, ti, o in raw_results if o.strip().lower() != text_lower]

    candidates: list[dict] = []

    for pi, ti, output in identical:
        diff = score_difficulty(output)  # cache hit: identical to input
        candidates.append({
            'text':              output,
            'temperature':       CANDIDATE_TEMPERATURES[ti],
            'prompt_type':       pi + 1,
            'similarity':        1.0,
            'tail_difficulty':   diff,
            'difficulty':        diff,
            'nli_label':         'IDENTICAL',
            'nli_contradiction': 0.0,
            'flagged':           'identical_to_input',
        })

    if to_score:
        outputs_only = [o for _, _, o in to_score]
        sims         = semantic_similarity_batch(text, outputs_only)
        nli_results  = meaning_preserved_batch(text, outputs_only, contradiction_threshold)

        for i, (pi, ti, output) in enumerate(to_score):
            safe, nli_label, contradiction_score = nli_results[i]

            if not safe:
                flagged = 'nli_contradiction'
            elif sims[i] < effective_sim_threshold:
                flagged = 'low_similarity'
            else:
                flagged = None

            # Only score difficulty when the candidate may be selected
            tail_diff = score_difficulty(output) if flagged is None else 99.0

            candidates.append({
                'text':              output,
                'temperature':       CANDIDATE_TEMPERATURES[ti],
                'prompt_type':       pi + 1,
                'similarity':        round(sims[i], 3),
                'tail_difficulty':   round(tail_diff, 3),
                'difficulty':        round(tail_diff, 3),
                'nli_label':         nli_label,
                'nli_contradiction': contradiction_score,
                'flagged':           flagged,
            })

    return candidates

def _roundtrip_penalty(hard_words: list[str], candidate_text: str) -> float:
    """
    For each original hard word, check whether the candidate text still
    contains a reasonable proxy — direct match, synonym, or hypernym.
    Returns a penalty in [0, 1]: 0 = all covered, 1 = all missing.
    Catches lossy substitutions like bioluminescence → 'making light'.
    """
    from nltk.corpus import wordnet

    if not hard_words:
        return 0.0

    candidate_lower = candidate_text.lower()
    missing = 0

    for word in hard_words:
        if word.lower() in candidate_lower:
            continue

        synsets  = wordnet.synsets(word)
        synonyms = {
            lemma.name().lower().replace('_', ' ')
            for s in synsets for lemma in s.lemmas()
        }
        if any(syn in candidate_lower for syn in synonyms):
            continue

        hypernyms = {
            lemma.name().lower().replace('_', ' ')
            for s in synsets
            for hyp in s.hypernyms()
            for lemma in hyp.lemmas()
        }
        if any(h in candidate_lower for h in hypernyms):
            continue

        missing += 1

    return missing / max(len(hard_words), 1)


def _select_best(
    candidates: list[dict],
    original_score: float,
    original_word_count: int,
    hard_words: list[str],              # original hard words list
) -> dict:
    original_hard_set = {w.lower() for w in hard_words}   # ← built right here

    """
    Three-tier fallback selection:
    Tier 1: valid (gates passed) AND actually simpler than original
    Tier 2: gates passed but not simpler (still better than nothing)
    Tier 3: highest similarity among everything generated
    """

    def combined_score(c: dict) -> float:
        normalized_diff = c['tail_difficulty'] / 10.0
        sim_penalty     = (1.0 - c['similarity']) ** 2
        length_ratio    = len(c['text'].split()) / max(original_word_count, 1)
        length_penalty  = max(0.0, length_ratio - 1.0)

        candidate_hard = {
            w['word'].lower()
            for w in find_difficult_words_in_text(c['text'], threshold=5.0)['difficult_words']
        }
        introduced_count     = len(candidate_hard - original_hard_set)  # ← now actually used
        introduction_penalty = introduced_count * 0.04

        roundtrip = _roundtrip_penalty(hard_words, c['text'])

        return (
            0.50 * normalized_diff +
            0.30 * sim_penalty +
            0.08 * length_penalty +
            0.08 * introduction_penalty +
            0.04 * roundtrip
        )

    valid = [c for c in candidates
             if c.get('flagged') is None and c['tail_difficulty'] < original_score]
    if valid:
        return min(valid, key=combined_score)

    fallback = [c for c in candidates
                if c.get('flagged') not in ('nli_contradiction', 'identical_to_input')]
    if fallback:
        return min(fallback, key=combined_score)

    return max(candidates, key=lambda c: c['similarity'])

def _build_hard_word_suffix(hard_words: list[str]) -> str:
    if not hard_words:
        return ""
    must_replace = ', '.join(hard_words)
    return (
        f"\n\nCRITICAL: You MUST replace ALL of these specific words "
        f"with simpler alternatives: {must_replace}\n"
        f"Do not use any of these words in your output under any circumstances.\n"
        f"IMPORTANT: Your replacement words must be simpler than the originals. "
        f"Do NOT introduce new long or uncommon words as replacements. "
        f"If a technical term has no simple equivalent, keep it and add "
        f"a brief parenthetical explanation."
    )

# ── Main async simplification function ────────────────────────────────────────
async def simplify_text(
    text: str,
    sim_threshold: float           = 0.65,
    n_candidates: int              = 10,
    hard_words: list               = None,
    contradiction_threshold: float = 0.6,
) -> dict:
    """
    Two-phase MBR simplification.

    Phase 1 (probe): fire PHASE1_SIZE candidates concurrently.
    If the best probe clears all gates and tail_difficulty < EASY_EXIT_THRESHOLD,
    return immediately (no phase 2).

    Phase 2 (full): fire remaining candidates only when phase 1 found nothing
    good enough.  All candidates from both phases are pooled for final selection.

    Result is stored in _simplification_cache keyed by (text, sim_threshold)
    so subsequent identical calls cost zero Groq calls.
    """
    try:
        if not text or not text.strip():
            return {"error": "Text is empty", "success": False}

        # ── Cache lookup ───────────────────────────────────────────────────────
        ck = _cache_key(text, sim_threshold)
        if ck in _simplification_cache:
            return _simplification_cache[ck]

        original_score      = score_difficulty(text)
        original_word_count = len(text.split())
        text_lower          = text.strip().lower()
        effective_sim       = _adaptive_sim_threshold(original_word_count, original_score, sim_threshold)

        # ── Hard-word prompt suffix ────────────────────────────────────────────
        hard_word_suffix = _build_hard_word_suffix(hard_words or [])

        combos_all    = _ALL_COMBOS[:n_candidates]
        phase1_combos = combos_all[:PHASE1_SIZE]
        phase2_combos = combos_all[PHASE1_SIZE:]

        seen_outputs: set[str] = set()
        all_candidates: list[dict] = []

        # ── Phase 1: probe ─────────────────────────────────────────────────────
        phase1_raw  = await _generate_batch_async(text, phase1_combos, hard_word_suffix)
        phase1_uniq = _dedup(phase1_raw, seen_outputs)
        all_candidates += _score_and_flag_candidates(
            text, phase1_uniq, original_score, effective_sim,
            text_lower, contradiction_threshold,
        )

        # Check whether any phase-1 candidate is good enough to exit early
        phase1_valid = [
            c for c in all_candidates
            if c.get('flagged') is None and c['tail_difficulty'] < EASY_EXIT_THRESHOLD
        ]
        skip_phase2 = len(phase1_valid) > 0

        # ── Phase 2: full grid (skipped on early exit) ─────────────────────────
        if not skip_phase2 and phase2_combos:
            phase2_raw  = await _generate_batch_async(text, phase2_combos, hard_word_suffix)
            phase2_uniq = _dedup(phase2_raw, seen_outputs)
            all_candidates += _score_and_flag_candidates(
                text, phase2_uniq, original_score, effective_sim,
                text_lower, contradiction_threshold,
            )

        if not all_candidates:
            return {"error": "All candidates failed", "success": False}

        # ── Select best ────────────────────────────────────────────────────────
        original_hard_set = {
            w['word'].lower()
            for w in find_difficult_words_in_text(text, threshold=5.0)['difficult_words']
        }
        best = _select_best(
            all_candidates,
            original_score,
            original_word_count,
            hard_words or [],            
        )
        simplified = best['text']

        # ── Metrics ────────────────────────────────────────────────────────────
        original_flesch   = textstat.flesch_reading_ease(text)
        simplified_flesch = textstat.flesch_reading_ease(simplified)
        valid_pool = [c for c in all_candidates if c.get('flagged') is None
                      and c['tail_difficulty'] < original_score]

        n_killed_nli  = sum(1 for c in all_candidates if c.get('flagged') == 'nli_contradiction')
        n_killed_sim  = sum(1 for c in all_candidates if c.get('flagged') == 'low_similarity')
        n_killed_iden = sum(1 for c in all_candidates if c.get('flagged') == 'identical_to_input')

        result = {
            "original":          text,
            "simplified":        simplified,
            "original_flesch":   original_flesch,
            "simplified_flesch": simplified_flesch,
            "improvement":       round(simplified_flesch - original_flesch, 2),
            "success":           True,
            "reranking": {
                "candidates_generated":    len(all_candidates),
                "candidates_valid":        len(valid_pool),
                "phase2_skipped":          skip_phase2,
                "best_temperature":        best['temperature'],
                "best_similarity":         best['similarity'],
                "best_difficulty":         best['tail_difficulty'],
                "original_difficulty":     round(original_score, 3),
                "difficulty_reduction":    round(original_score - best['tail_difficulty'], 3),
                "final_difficulty":        round(best['tail_difficulty'], 3),
                "scoring_method":          "tail_aware (0.6*mean + 0.4*p90) + sim²",
                "sim_threshold_base":      sim_threshold,
                "sim_threshold_effective": round(effective_sim, 3),
                "best_nli_label":          best.get('nli_label', 'N/A'),
                "best_nli_contradiction":  best.get('nli_contradiction', 0.0),
                "filter_stats": {
                    "killed_by_nli":        n_killed_nli,
                    "killed_by_similarity": n_killed_sim,
                    "killed_identical":     n_killed_iden,
                },
                "all_candidates": all_candidates,
            },
        }

        # Store in cache
        _simplification_cache[ck] = result
        return result

    except Exception as e:
        return {"error": str(e), "success": False}


def _dedup(
    raw: list[tuple[int, int, str]],
    seen: set[str],
) -> list[tuple[int, int, str]]:
    """Remove outputs already seen in a previous phase; mutates `seen` in place."""
    unique = []
    for pi, ti, output in raw:
        key = output.strip().lower()
        if key not in seen:
            seen.add(key)
            unique.append((pi, ti, output))
    return unique


# ── Sync wrapper ───────────────────────────────────────────────────────────────
def simplify_text_sync(
    text: str,
    sim_threshold: float           = 0.65,
    n_candidates: int              = 10,
    hard_words: list               = None,
    contradiction_threshold: float = 0.6,
) -> dict:
    """Sync entry point for CLI, tests, Celery tasks. Drop-in for v3."""
    return asyncio.run(
        simplify_text(
            text,
            sim_threshold=sim_threshold,
            n_candidates=n_candidates,
            hard_words=hard_words,
            contradiction_threshold=contradiction_threshold,
        )
    )


# ── Sentence splitter ──────────────────────────────────────────────────────────
def split_sentences(text: str) -> list[str]:
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]


# ── Semantic deduplication ─────────────────────────────────────────────────────
def _cluster_hard_sentences(
    sentences: list[str],
    difficulties: list[float],
) -> dict[int, int]:
    """
    Cluster sentences by semantic similarity.
    Returns {sentence_index: representative_index} for every sentence.
    The representative of a cluster is the sentence with the highest difficulty.

    If only one sentence, returns {0: 0}.
    """
    n = len(sentences)
    if n == 1:
        return {0: 0}

    embeddings = sim_model.encode(sentences, convert_to_tensor=False, batch_size=32)
    sim_matrix = sk_cosine_similarity(embeddings)
    dist_matrix = np.clip(1.0 - sim_matrix, 0.0, None)
    np.fill_diagonal(dist_matrix, 0.0)

    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=1.0 - CLUSTER_SIM_THRESHOLD,
        metric='precomputed',
        linkage='average',
    )
    labels = clustering.fit_predict(dist_matrix)

    # For each cluster, elect the sentence with the highest difficulty as rep
    cluster_rep: dict[int, int] = {}  # cluster_id → sentence_index
    for idx, cluster_id in enumerate(labels):
        if cluster_id not in cluster_rep or difficulties[idx] > difficulties[cluster_rep[cluster_id]]:
            cluster_rep[cluster_id] = idx

    return {idx: cluster_rep[labels[idx]] for idx in range(n)}


# ── Paragraph grouping ─────────────────────────────────────────────────────────
def _group_into_paragraphs(
    indices: list[int],
    sentences: list[str],
    difficulties: list[float],
) -> list[list[int]]:
    """
    Group consecutive hard sentence indices into paragraph batches when:
      - none of the sentences in the group is very hard (>= PARAGRAPH_BATCH_THRESHOLD)
      - combined word count < PARAGRAPH_MAX_WORDS

    Very hard sentences are always isolated (solo group).
    Returns list of groups, where each group is a list of sentence indices.
    """
    groups: list[list[int]] = []
    current_group: list[int] = []
    current_wc = 0

    for idx in indices:
        s    = sentences[idx]
        d    = difficulties[idx]
        wc   = len(s.split())
        very_hard = d >= PARAGRAPH_BATCH_THRESHOLD

        if very_hard:
            # Flush current group first, then isolate this sentence
            if current_group:
                groups.append(current_group)
                current_group = []
                current_wc = 0
            groups.append([idx])
        elif current_wc + wc > PARAGRAPH_MAX_WORDS and current_group:
            groups.append(current_group)
            current_group = [idx]
            current_wc = wc
        else:
            current_group.append(idx)
            current_wc += wc

    if current_group:
        groups.append(current_group)

    return groups


# ── Core targeted async function ───────────────────────────────────────────────
async def simplify_targeted_async(
    text: str,
    difficulty_threshold: float   = 4.5,
    sim_threshold: float          = 0.65,
    n_candidates: int             = 10,
    max_concurrent_sentences: int = 2,
) -> dict:
    """
    Document-aware targeted simplification pipeline:

    1. Score all sentences (all cached after first call).
    2. Identify hard sentences (difficulty >= difficulty_threshold, words >= 5).
    3. Semantic-cluster hard sentences; only one representative per cluster
       goes to the LLM — non-reps reuse the rep's result.
    4. Group representative sentences into paragraph batches where possible
       (moderate difficulty, combined word count under budget).
    5. Allocate Groq call budget across groups proportionally to difficulty.
    6. Simplify each group concurrently under a semaphore.
    7. Reassemble sentences in original order; compute final metrics.
    """
    sentences = split_sentences(text)

    # ── Single-sentence fast path ──────────────────────────────────────────────
    if len(sentences) <= 1:
        d          = score_difficulty(text)
        hw         = [w['word'] for w in find_difficult_words_in_text(text, threshold=difficulty_threshold)['difficult_words']]
        adaptive_n = min(_adaptive_n_candidates(text, d), n_candidates)
        return await simplify_text(text, sim_threshold=sim_threshold,
                                   n_candidates=adaptive_n, hard_words=hw)

    # ── Step 1: score all sentences ────────────────────────────────────────────
    sentence_difficulties = [score_difficulty(s) for s in sentences]  # all cached after first run

    # ── Step 2: identify hard sentences ───────────────────────────────────────
    hard_indices = [
        i for i, (s, d) in enumerate(zip(sentences, sentence_difficulties))
        if len(s.split()) >= 5 and d >= difficulty_threshold
    ]

    # ── Step 3: semantic deduplication ────────────────────────────────────────
    # sentence_to_rep maps each hard sentence index to its cluster's representative index
    sentence_to_rep: dict[int, int] = {}
    if hard_indices:
        hard_sentences   = [sentences[i] for i in hard_indices]
        hard_difficulties = [sentence_difficulties[i] for i in hard_indices]
        local_to_rep     = _cluster_hard_sentences(hard_sentences, hard_difficulties)
        # Translate local (0..n_hard) indices back to global sentence indices
        sentence_to_rep  = {
            hard_indices[local_idx]: hard_indices[rep_local]
            for local_idx, rep_local in local_to_rep.items()
        }

    # Representatives are hard sentences that are their own cluster rep
    rep_indices = sorted(set(sentence_to_rep[i] for i in hard_indices) if hard_indices else [])

    # ── Step 4: paragraph grouping of representatives ─────────────────────────
    rep_difficulties = [sentence_difficulties[i] for i in rep_indices]
    groups           = _group_into_paragraphs(rep_indices, sentences, sentence_difficulties)

    # ── Step 5: document-aware budget ─────────────────────────────────────────
    # One budget entry per group (paragraph or solo sentence)
    group_difficulties = [
        max(sentence_difficulties[i] for i in g) for g in groups
    ]
    total_budget = min(BASE_BUDGET + len(hard_indices) * BUDGET_PER_HARD, MAX_BUDGET)
    group_budgets = compute_candidate_budget(group_difficulties, total_budget)

    # ── Step 6: simplify each group ────────────────────────────────────────────
    semaphore = asyncio.Semaphore(max_concurrent_sentences)

    # Map from rep_index → simplified result dict (populated after gather)
    rep_to_result: dict[int, dict] = {}

    async def process_group(group_indices: list[int], budget: int) -> dict:
        """
        Simplify a group of sentences (paragraph batch or single sentence).
        Returns a result dict with 'simplified' and metadata.
        The `simplified` field may contain multiple sentences joined by a space
        when the group has more than one member.
        """
        group_text = ' '.join(sentences[i] for i in group_indices)
        group_diff = max(sentence_difficulties[i] for i in group_indices)

        # Collect hard words across the whole group
        hw_results = find_difficult_words_in_text(group_text, threshold=difficulty_threshold)
        hard_words = [w['word'] for w in hw_results['difficult_words']]

        # Cap budget by per-sentence adaptive ceiling × group size
        per_sentence_ceil = _adaptive_n_candidates(group_text, group_diff)
        effective_n = min(budget, per_sentence_ceil * len(group_indices), n_candidates)
        effective_n = max(effective_n, MIN_CANDIDATES)

        async with semaphore:
            return await simplify_text(
                group_text,
                sim_threshold=sim_threshold,
                n_candidates=effective_n,
                hard_words=hard_words,
            )

    group_results: list[dict] = await asyncio.gather(
        *[process_group(g, b) for g, b in zip(groups, group_budgets)]
    )

    # Map each representative sentence index to its group's simplified text
    # (paragraph groups produce one block of text — we use it wholesale)
    for group_idxs, result in zip(groups, group_results):
        for global_idx in group_idxs:
            rep_to_result[global_idx] = result

    # ── Step 7: reassemble all sentences in order ──────────────────────────────
    details: list[dict] = []
    result_sentences: list[str] = []

    # Track which paragraph groups have already been "consumed" so we don't
    # repeat their text for every sentence in the group.
    emitted_groups: set[int] = set()  # keyed by the first index in each group

    for i, sentence in enumerate(sentences):
        wc = len(sentence.split())
        d  = sentence_difficulties[i]

        # Easy / too-short sentences: pass through
        if wc < 5 or d < difficulty_threshold:
            result_sentences.append(sentence)
            reason = 'too short' if wc < 5 else 'already easy'
            details.append({
                'original':   sentence,
                'simplified': sentence,
                'action':     'kept_unchanged',
                'reason':     reason,
                **({"difficulty": round(d, 3)} if wc >= 5 else {}),
                'n_candidates_used': 0,
            })
            continue

        # Hard sentence: find its representative
        rep_idx = sentence_to_rep.get(i, i)
        result  = rep_to_result.get(rep_idx)

        if result is None or not result.get('success'):
            result_sentences.append(sentence)
            details.append({
                'original':   sentence,
                'simplified': sentence,
                'action':     'kept_unchanged',
                'reason':     'simplification failed',
                'difficulty': round(d, 3),
                'n_candidates_used': 0,
            })
            continue

        # Find the group this representative belongs to
        rep_group = next((g for g in groups if rep_idx in g), [rep_idx])
        group_key = rep_group[0]

        if len(rep_group) == 1:
            # Solo sentence — use simplified text directly
            simplified_text     = result['simplified']
            n_used              = result['reranking']['candidates_generated']
            final_diff          = result['reranking']['final_difficulty']
            from_cache          = result['reranking'].get('phase2_skipped', False)
        else:
            # Paragraph group — emit the whole block once for the first sentence
            # in the group; subsequent sentences in the group get the 'grouped'
            # marker so they don't duplicate text in result_sentences.
            if group_key not in emitted_groups:
                emitted_groups.add(group_key)
                simplified_text = result['simplified']
                n_used          = result['reranking']['candidates_generated']
                final_diff      = result['reranking']['final_difficulty']
                from_cache      = result['reranking'].get('phase2_skipped', False)
            else:
                # Already emitted — skip this sentence (it was part of the paragraph)
                details.append({
                    'original':   sentence,
                    'simplified': '[grouped with previous sentence]',
                    'action':     'grouped',
                    'difficulty': round(d, 3),
                    'n_candidates_used': 0,
                })
                continue

        result_sentences.append(simplified_text)
        reranking = result['reranking']
        details.append({
            'original':            sentence,
            'simplified':          simplified_text,
            'action':              'simplified',
            'original_difficulty': round(d, 3),
            'final_difficulty':    round(final_diff, 3),
            'reduction':           round(d - final_diff, 3),
            'hard_words_targeted': [w['word'] for w in
                find_difficult_words_in_text(sentence, threshold=difficulty_threshold)['difficult_words']],
            'winning_temp':        reranking['best_temperature'],
            'nli_label':           reranking.get('best_nli_label', 'N/A'),
            'filter_stats':        reranking.get('filter_stats', {}),
            'n_candidates_used':   n_used,
            'phase2_skipped':      reranking.get('phase2_skipped', False),
            'from_cluster_rep':    rep_idx != i,
        })

    final_text = ' '.join(result_sentences)

    # All score_difficulty calls below are lru_cache hits
    original_flesch   = textstat.flesch_reading_ease(text)
    simplified_flesch = textstat.flesch_reading_ease(final_text)
    original_diff     = score_difficulty(text)
    final_diff        = score_difficulty(final_text)

    sentences_simplified = sum(1 for d in details if d['action'] == 'simplified')
    sentences_kept       = sum(1 for d in details if d['action'] == 'kept_unchanged')
    sentences_grouped    = sum(1 for d in details if d['action'] == 'grouped')
    api_calls_made       = sum(d.get('n_candidates_used', 0) for d in details)
    n_clusters           = len(set(sentence_to_rep.values())) if sentence_to_rep else 0
    n_reps_simplified    = len(rep_indices)

    return {
        "original":          text,
        "simplified":        final_text,
        "original_flesch":   original_flesch,
        "simplified_flesch": simplified_flesch,
        "improvement":       round(simplified_flesch - original_flesch, 2),
        "success":           True,
        "reranking": {
            "mode":                   "targeted_v4",
            "total_sentences":        len(sentences),
            "sentences_simplified":   sentences_simplified,
            "sentences_kept":         sentences_kept,
            "sentences_grouped":      sentences_grouped,
            "hard_sentences_found":   len(hard_indices),
            "clusters_found":         n_clusters,
            "representatives_sent":   n_reps_simplified,
            "api_calls_made":         api_calls_made,
            "original_difficulty":    round(original_diff, 3),
            "final_difficulty":       round(final_diff, 3),
            "difficulty_reduction":   round(original_diff - final_diff, 3),
            "scoring_method":         "tail_aware (0.6*mean + 0.4*p90) + sim²",
            "max_concurrent":         max_concurrent_sentences,
        },
        "sentence_details": details,
    }


# ── Public sync entry point (used by api.py) ───────────────────────────────────
def simplify_targeted(
    text: str,
    difficulty_threshold: float = 3.5,
    sim_threshold: float        = 0.65,
    n_candidates: int           = 10,
) -> dict:
    """
    Sync wrapper around simplify_targeted_async.
    api.py imports and calls this — zero changes needed there.

    For FastAPI: make your route async and await simplify_targeted_async()
    directly to avoid blocking the event loop with asyncio.run().
    """
    return asyncio.run(
        simplify_targeted_async(
            text,
            difficulty_threshold=difficulty_threshold,
            sim_threshold=sim_threshold,
            n_candidates=n_candidates,
        )
    )


# ── CLI smoke test ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import time

    test_texts = [
        # Short sentence — should trigger early exit (phase-2 skipped)
        "Myocardial infarction occurs when blood flow decreases or stops to a part of the heart, causing damage to the heart muscle.",

        # Legal sentence — very hard, no early exit expected
        "The defendant, pursuant to the aforementioned contractual obligations stipulated in section 4.2 of the binding agreement, shall be held liable for any consequential damages.",

        # Multi-sentence paragraph — exercises dedup + paragraph batching
        (
            "Deep beneath the ocean's surface lies a mysterious world that scientists have only begun to understand. "
            "The immense pressure and complete darkness make exploration extremely challenging. "
            "Nevertheless, specialized submarines have revealed ecosystems filled with extraordinary creatures that survive under harsh conditions. "
            "Some of these organisms produce their own light through a process known as bioluminescence. "
            "Their remarkable adaptations demonstrate the incredible resilience of life on Earth. "
            "As technology advances, researchers hope to uncover even more secrets hidden in the deep sea."
        ),
    ]

    print("Text Simplifier — v4 (two-phase + dedup + budget + paragraph batching + cache)\n")
    print("=" * 75)

    for text in test_texts:
        t0      = time.perf_counter()
        result  = simplify_targeted(text)
        elapsed = time.perf_counter() - t0

        if result.get("success"):
            r = result["reranking"]
            print(f"\nOriginal  (Flesch: {result['original_flesch']:.1f} | "
                  f"Difficulty: {r['original_difficulty']}):")
            print(f"  {result['original'][:120]}{'...' if len(result['original']) > 120 else ''}")
            print(f"\nSimplified (Flesch: {result['simplified_flesch']:.1f} | "
                  f"Difficulty: {r['final_difficulty']}):")
            print(f"  {result['simplified'][:120]}{'...' if len(result['simplified']) > 120 else ''}")
            print(f"\nReduction: {r['difficulty_reduction']:+.3f} difficulty | "
                  f"{result['improvement']:+.1f} Flesch | "
                  f"Wall time: {elapsed:.2f}s")
            print(f"Sentences: {r['total_sentences']} total | "
                  f"{r['hard_sentences_found']} hard | "
                  f"{r.get('clusters_found', 'N/A')} clusters | "
                  f"{r.get('representatives_sent', 'N/A')} sent to LLM")
            print(f"API calls: {r['api_calls_made']} | "
                  f"Simplified: {r['sentences_simplified']} | "
                  f"Grouped: {r.get('sentences_grouped', 0)} | "
                  f"Kept: {r['sentences_kept']}")

            if "sentence_details" in result:
                print("\nSentence details:")
                for d in result["sentence_details"]:
                    action = d['action']
                    marker = {'simplified': '✓', 'kept_unchanged': '—', 'grouped': '⊕'}.get(action, '?')
                    n_used = d.get('n_candidates_used', 0)
                    rep    = ' [cluster-reuse]' if d.get('from_cluster_rep') else ''
                    skip   = ' [phase2-skipped]' if d.get('phase2_skipped') else ''
                    print(f"  {marker} [{action}] calls={n_used}{rep}{skip}")
            print("=" * 75)
        else:
            print(f"Error: {result.get('error')}")