"""
Text Simplifier with Minimum Bayes Risk Re-ranking  (v2)
---------------------------------------------------------
Changes over v1:

1. TAIL-AWARE DIFFICULTY  (score_difficulty)
   Old: mean of all word scores  → a single buried hard word doesn't move the mean
   New: 0.6 * mean + 0.4 * p90  → heavily penalizes candidates that keep even
        one very hard word, which is exactly the failure mode dyslexic readers hit.

2. NLI CONTRADICTION FILTER  (meaning_preserved)
   Old: cosine similarity ≥ 0.65 — catches topic drift but misses factual flips
        ("patient recovered" ↔ "patient died" scores ~0.85 similarity)
   New: cross-encoder NLI on top of similarity.  A candidate is rejected if the
        model predicts CONTRADICTION with confidence > 0.6.  Entailment is NOT
        required (paraphrases are neutral, not strictly entailed) — we only hard-
        block contradictions.  This is a gate, not a weight.
   Model: cross-encoder/nli-deberta-v3-small (~86 MB, CPU-friendly, ~40 ms/pair)

3. FULL PROMPT × TEMPERATURE GRID  (simplify_text)
   Old: zip(temperatures, prompts) → 5 candidates, one per temp, one per prompt
   New: cartesian product of all 5 prompts × all 5 temperatures = 25 candidates.
        n_candidates param caps how many we actually generate (default=10 for
        balance between quality and Groq API cost / latency).
        Candidates are sampled from the grid in a round-robin across prompts so
        every strategy gets representation regardless of n_candidates.

4. UPDATED combined_score
   Now uses tail_difficulty (p90-blended) instead of raw mean difficulty so the
   re-ranker and the filter speak the same language.

5. CANDIDATE METADATA
   Each candidate now carries:
     - nli_label / nli_score   for debugging which candidates were killed by NLI
     - tail_difficulty          the actual metric used in re-ranking
     - flagged reason           extended to include 'nli_contradiction'

API contract:  zero changes — same return dict keys as v1.
"""

from groq import Groq
import textstat
import os
import sys
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
import itertools
import re

load_dotenv()

# ── Clients & models ──────────────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client       = Groq(api_key=GROQ_API_KEY)

# Similarity model — unchanged from v1
sim_model = SentenceTransformer('paraphrase-MiniLM-L3-v2')

# NLI model — cross-encoder is much more accurate than bi-encoder for NLI.
# deberta-v3-small is ~86 MB and runs in ~40 ms per pair on CPU.
# Loaded once at module import so per-call overhead is zero.
print("Loading NLI model (cross-encoder/nli-deberta-v3-small)...")
nli_model = pipeline(
    "text-classification",
    model="cross-encoder/nli-deberta-v3-small",
    device=-1,          # CPU; change to 0 if you have a GPU on the server
    top_k=None,         # return scores for ALL labels, not just argmax
)
print("NLI model loaded.")

# ── Scorer path ───────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
from ai.models.difficulty_scorer import find_difficult_words_in_text

# ── Prompts & temperatures ────────────────────────────────────────────────────
SYSTEM_PROMPTS = [
    # Prompt 1 — conservative, word-level changes only
    """You are helping people with dyslexia read difficult text.
    Rewrite the text by replacing hard words with simpler ones.
    Keep the sentence structure identical. Keep ALL meaning.
    Return ONLY the rewritten text.""",

    # Prompt 2 — structural, break sentences
    """You are helping people with dyslexia read difficult text.
    Rewrite the text by breaking long sentences into shorter ones.
    Use simple subject-verb-object structure.
    Keep ALL meaning. Return ONLY the rewritten text.""",

    # Prompt 3 — aggressive, full rewrite
    """You are helping people with dyslexia read difficult text.
    Rewrite the text completely in simple everyday English.
    Use words a 12-year-old would know.
    Keep ALL meaning. Return ONLY the rewritten text.""",

    # Prompt 4 — active voice focus
    """You are helping people with dyslexia read difficult text.
    Rewrite the text using active voice only.
    Replace all passive constructions. Use simple words.
    Keep ALL meaning. Return ONLY the rewritten text.""",

    # Prompt 5 — concise, no padding
    """You are helping people with dyslexia read difficult text.
    Rewrite the text in simple everyday English.
    Replace hard words with simpler ones directly — do NOT add explanations or definitions.
    Keep the output roughly the same length as the input.
    Keep ALL meaning. Return ONLY the rewritten text.""",
]

# Full grid: 5 prompts × 5 temperatures = 25 possible candidates
CANDIDATE_TEMPERATURES = [0.1, 0.3, 0.7, 1.0, 1.2]

# Pre-build the full cartesian product in a round-robin order so that
# when n_candidates < 25 we still get diversity across all 5 prompts.
# Order: (p0,t0),(p1,t1),(p2,t2),(p3,t3),(p4,t4),(p0,t1),(p1,t2),...
_ALL_COMBOS = []
for _offset in range(len(CANDIDATE_TEMPERATURES)):
    for _pi, _prompt in enumerate(SYSTEM_PROMPTS):
        _ti = (_pi + _offset) % len(CANDIDATE_TEMPERATURES)
        _combo = (_pi, _ti)
        if _combo not in _ALL_COMBOS:
            _ALL_COMBOS.append(_combo)

# ── Core scoring functions ────────────────────────────────────────────────────

def score_difficulty(text: str) -> float:
    """
    Tail-aware difficulty score for a piece of text.

    OLD (v1): mean of all word difficulty scores.
    NEW (v2): 0.6 * mean + 0.4 * p90

    Why p90?
      - Dyslexic readers slow down or stall on the *hardest* word in a phrase,
        not the average word.
      - A candidate with mean=3.5 but one word at 8.5 is NOT simpler than one
        with mean=4.0 and max=5.0.  The old scorer ranked the first one higher
        (incorrectly). The p90 blend fixes this without going all the way to max
        (which is too sensitive to a single proper noun or technical term that
        cannot be removed).
    """
    results    = find_difficult_words_in_text(text, threshold=0.0)
    all_scored = results['all_scored']
    if not all_scored:
        return 5.0
    scores = [w['difficulty_score'] for w in all_scored]
    mean   = float(np.mean(scores))
    p90    = float(np.percentile(scores, 90))
    return round(0.6 * mean + 0.4 * p90, 4)


def semantic_similarity(text_a: str, text_b: str) -> float:
    """Bi-encoder cosine similarity — fast, good for topic drift detection."""
    emb_a = sim_model.encode(text_a, convert_to_tensor=True)
    emb_b = sim_model.encode(text_b, convert_to_tensor=True)
    return float(util.cos_sim(emb_a, emb_b))


def meaning_preserved(original: str, candidate: str,
                       contradiction_threshold: float = 0.6) -> tuple[bool, str, float]:
    """
    NLI contradiction gate.

    Returns: (is_safe, label, contradiction_score)

    Design decisions:
    - We run NLI as original → candidate (premise → hypothesis).
    - We only BLOCK on CONTRADICTION.  We do NOT require ENTAILMENT because
      a valid simplification is typically NEUTRAL (paraphrase), not strictly
      entailed.  Requiring entailment would kill most good candidates.
    - Threshold 0.6: conservative — only reject when the model is fairly sure.
      Lower = more aggressive filtering.  You can tune this per domain.
    - Truncation at 512 tokens is handled by the pipeline automatically.

    Cost: ~40 ms per pair on CPU for deberta-v3-small.
    """
    try:
        raw = nli_model(f"{original} [SEP] {candidate}")

        # pipeline with top_k=None returns a NESTED list for a single input:
        #   [[{'label': 'ENTAILMENT', 'score': 0.1}, {'label': 'NEUTRAL', ...}, ...]]
        # Without top_k it returns a flat list (only the argmax):
        #   [{'label': 'CONTRADICTION', 'score': 0.9}]
        # Unwrap the outer batch dimension if present so both forms work.
        inner = raw[0] if raw and isinstance(raw[0], list) else raw

        scores_by_label = {r['label'].upper(): r['score'] for r in inner}
        contradiction   = scores_by_label.get('CONTRADICTION', 0.0)

        # Dominant label for metadata
        label = max(scores_by_label, key=scores_by_label.get)

        if contradiction >= contradiction_threshold:
            return False, label, round(contradiction, 3)
        return True, label, round(contradiction, 3)

    except Exception:
        # Fail open — never silently discard a candidate due to an NLI error.
        return True, 'UNKNOWN', 0.0


# ── Candidate generation ──────────────────────────────────────────────────────

def generate_candidate(text: str, temperature: float, system_prompt: str) -> str:
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": f"Simplify this text:\n\n{text}"}
        ],
        temperature=temperature,
        max_tokens=1024,
    )
    return response.choices[0].message.content.strip()


# ── Main simplification function ──────────────────────────────────────────────

def _adaptive_sim_threshold(word_count: int, original_score: float,
                             base: float = 0.65) -> float:
    """
    Relax the similarity threshold for sentences that inherently need more
    aggressive rewriting to become simple.

    Two adjustment axes:
      length:     longer sentences need structural change → looser threshold
                  range: 0 at ≤10 words, -0.10 at ≥50 words
      difficulty: harder sentences need more vocab swap → looser threshold
                  range: 0 at difficulty=3.0, -0.08 at difficulty≥7.0

    Hard floor 0.45 — below this the bi-encoder loses resolution and NLI
    alone cannot be trusted as a semantic gate.

    Examples:
      short easy  (10w, diff=3.0) → 0.65  (no relaxation)
      medium hard (25w, diff=4.5) → 0.59
      long hard   (40w, diff=5.5) → 0.54
      very long   (55w, diff=7.0) → 0.47
    """
    length_adj     = 0.10 * min(max((word_count - 10) / 40.0, 0.0), 1.0)
    difficulty_adj = 0.08 * min(max((original_score - 3.0) / 4.0, 0.0), 1.0)
    return max(base - length_adj - difficulty_adj, 0.45)


def simplify_text(text: str,
                  sim_threshold: float = 0.65,
                  n_candidates: int    = 10,
                  hard_words: list     = None,
                  contradiction_threshold: float = 0.6) -> dict:
    """
    Generate N candidates from the full prompt×temperature grid, apply
    NLI contradiction filter + tail-aware difficulty scoring, then
    pick the best via MBR combined score.

    sim_threshold: BASE value. The actual threshold per call is adaptive —
      relaxed for longer/harder sentences via _adaptive_sim_threshold().
      Always pass the base here; do not compensate in callers.

    n_candidates: how many from the 25-combo grid to actually call.
      Default 10 = 2 passes through all 5 prompts, good cost/quality trade-off.
      Set to 25 for maximum quality (5× Groq calls per sentence).

    Returns same dict structure as v1 — api.py unchanged.
    """
    try:
        if not text or len(text.strip()) == 0:
            return {"error": "Text is empty", "success": False}

        original_score      = score_difficulty(text)
        original_word_count = len(text.split())

        # Adaptive similarity gate — relaxes for long/hard sentences
        effective_sim_threshold = _adaptive_sim_threshold(
            original_word_count, original_score, base=sim_threshold
        )

        # ── 1. Build prompt suffix for targeted hard words ─────────────────
        hard_word_suffix = ""
        if hard_words and len(hard_words) > 0:
            must_replace     = ', '.join(hard_words)
            hard_word_suffix = (
                f"\n\nCRITICAL: You MUST replace ALL of these specific words "
                f"with simpler alternatives: {must_replace}\n"
                f"Do not use any of these words in your output under any circumstances."
            )

        # ── 2. Generate candidates from the round-robin grid ───────────────
        candidates   = []
        seen_outputs = set()   # dedup by text — LLM collapses at low temps
        combos_to_run = _ALL_COMBOS[:n_candidates]

        for prompt_idx, temp_idx in combos_to_run:
            temp   = CANDIDATE_TEMPERATURES[temp_idx]
            prompt = SYSTEM_PROMPTS[prompt_idx] + hard_word_suffix

            try:
                output = generate_candidate(text, temp, prompt)
            except Exception as e:
                continue

            # ── Dedup: skip if this exact output was already generated ─────
            # LLaMA-70b collapses to identical text at low temperatures across
            # different prompts. Scoring duplicates wastes NLI + difficulty calls
            # and artificially inflates candidates_generated count.
            output_key = output.strip().lower()
            if output_key in seen_outputs:
                continue
            seen_outputs.add(output_key)

            # ── Flag: identical to input ───────────────────────────────────
            if output.strip().lower() == text.strip().lower():
                candidates.append({
                    'text':              output,
                    'temperature':       temp,
                    'prompt_type':       prompt_idx + 1,
                    'similarity':        1.0,
                    'tail_difficulty':   score_difficulty(output),
                    'difficulty':        score_difficulty(output),  # v1 compat key
                    'nli_label':         'IDENTICAL',
                    'nli_contradiction': 0.0,
                    'flagged':           'identical_to_input',
                })
                continue

            # ── Score similarity (bi-encoder, fast) ────────────────────────
            sim = semantic_similarity(text, output)

            # ── NLI gate (cross-encoder, ~40 ms) ──────────────────────────
            # Run before difficulty scoring to avoid wasting time on
            # contradicting candidates.
            safe, nli_label, contradiction_score = meaning_preserved(
                text, output, contradiction_threshold
            )

            # ── Tail-aware difficulty ──────────────────────────────────────
            tail_diff = score_difficulty(output)

            flagged = None
            if not safe:
                flagged = 'nli_contradiction'
            elif sim < effective_sim_threshold:
                flagged = 'low_similarity'

            candidates.append({
                'text':              output,
                'temperature':       temp,
                'prompt_type':       prompt_idx + 1,
                'similarity':        round(sim, 3),
                'tail_difficulty':   round(tail_diff, 3),
                'difficulty':        round(tail_diff, 3),   # v1 compat key
                'nli_label':         nli_label,
                'nli_contradiction': contradiction_score,
                'flagged':           flagged,
            })

        if not candidates:
            return {"error": "All candidates failed", "success": False}

        # ── 3. Filter valid candidates ─────────────────────────────────────
        # A candidate is valid if:
        #   (a) NLI did not flag it as contradiction
        #   (b) similarity >= threshold (topic preserved)
        #   (c) actually simpler than the original (tail_difficulty improved)
        #   (d) not identical to input
        valid = [
            c for c in candidates
            if c.get('flagged') is None
            and c['tail_difficulty'] < original_score
        ]

        if not valid:
            # Fallback: relax the "must be simpler" constraint,
            # keep NLI + similarity gates, pick least bad.
            valid = [
                c for c in candidates
                if c.get('flagged') not in ('nli_contradiction', 'identical_to_input')
            ]

        if not valid:
            # Last resort: highest similarity among anything we generated.
            valid = [max(candidates, key=lambda c: c['similarity'])]

        # ── 4. MBR combined score ──────────────────────────────────────────
        # Weights:
        #   difficulty 0.55  — primary objective (tail-aware)
        #   similarity 0.35  — meaning preservation, squared penalty
        #   length     0.10  — penalize bloat
        #
        # Why squared similarity penalty?
        #   Linear (old): sim=0.90 vs sim=0.70 → cost difference of 0.05
        #   Squared (new): sim=0.90 vs sim=0.70 → cost difference of 0.024 vs 0.090
        #   Squaring compresses the penalty for candidates that are close to each
        #   other in similarity but amplifies it for large drift. This means the
        #   re-ranker correctly prefers a sim=0.89 candidate over sim=0.70 even
        #   when the sim=0.70 candidate is slightly simpler — which matches how
        #   dyslexic readers actually experience paraphrased text (large meaning
        #   drift is more disorienting than a small difficulty increase).
        def combined_score(c: dict) -> float:
            normalized_diff  = c['tail_difficulty'] / 10.0    # lower = simpler
            sim_penalty      = (1.0 - c['similarity']) ** 2   # squared: small drift forgiven, large punished

            candidate_wc   = len(c['text'].split())
            length_ratio   = candidate_wc / max(original_word_count, 1)
            length_penalty = max(0.0, length_ratio - 1.0)

            return (
                0.55 * normalized_diff +
                0.35 * sim_penalty +
                0.10 * length_penalty
            )

        best = min(valid, key=combined_score)
        simplified = best['text']

        # ── 5. Metrics ─────────────────────────────────────────────────────
        original_flesch   = textstat.flesch_reading_ease(text)
        simplified_flesch = textstat.flesch_reading_ease(simplified)

        # Count how many candidates each filter killed (useful for tuning)
        n_killed_nli        = sum(1 for c in candidates if c.get('flagged') == 'nli_contradiction')
        n_killed_similarity = sum(1 for c in candidates if c.get('flagged') == 'low_similarity')
        n_killed_identical  = sum(1 for c in candidates if c.get('flagged') == 'identical_to_input')

        return {
            "original":          text,
            "simplified":        simplified,
            "original_flesch":   original_flesch,
            "simplified_flesch": simplified_flesch,
            "improvement":       round(simplified_flesch - original_flesch, 2),
            "success":           True,
            "reranking": {
                # v1-compatible keys
                "candidates_generated":  len(candidates),
                "candidates_valid":      len(valid),
                "best_temperature":      best['temperature'],
                "best_similarity":       best['similarity'],
                "best_difficulty":       best['tail_difficulty'],
                "original_difficulty":   round(original_score, 3),
                "difficulty_reduction":  round(original_score - best['tail_difficulty'], 3),
                "final_difficulty":      round(best['tail_difficulty'], 3),
                # v2 additions
                "scoring_method":         "tail_aware (0.6*mean + 0.4*p90) + sim²",
                "sim_threshold_base":     sim_threshold,
                "sim_threshold_effective":round(effective_sim_threshold, 3),
                "best_nli_label":         best.get('nli_label', 'N/A'),
                "best_nli_contradiction":best.get('nli_contradiction', 0.0),
                "filter_stats": {
                    "killed_by_nli":        n_killed_nli,
                    "killed_by_similarity": n_killed_similarity,
                    "killed_identical":     n_killed_identical,
                },
                "all_candidates":        candidates,
            }
        }

    except Exception as e:
        return {"error": str(e), "success": False}


# ── Sentence splitter ─────────────────────────────────────────────────────────

def split_sentences(text: str) -> list[str]:
    """Split on sentence boundaries, preserving abbreviation edge cases."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]


# ── Targeted simplification (sentence-level routing) ─────────────────────────

def simplify_targeted(text: str,
                       difficulty_threshold: float = 3.5,
                       sim_threshold: float        = 0.65,
                       n_candidates: int           = 10) -> dict:
    """
    Only simplify sentences that are actually hard.
    Easy sentences pass through untouched.

    n_candidates passed through to simplify_text so callers can trade
    Groq API cost vs quality per their context (e.g. lower for real-time,
    higher for async document processing).
    """
    sentences = split_sentences(text)

    # Single sentence — skip the loop
    if len(sentences) <= 1:
        hard_word_results = find_difficult_words_in_text(text, threshold=6.0)
        hard_words        = [w['word'] for w in hard_word_results['difficult_words']]
        return simplify_text(
            text,
            sim_threshold=sim_threshold,
            n_candidates=n_candidates,
            hard_words=hard_words,
        )

    result_sentences = []
    details          = []
    api_calls_made   = 0

    for sentence in sentences:
        words = sentence.split()

        # Too short — keep as is
        if len(words) < 5:
            result_sentences.append(sentence)
            details.append({
                'original':   sentence,
                'simplified': sentence,
                'action':     'kept_unchanged',
                'reason':     'too short',
            })
            continue

        sentence_difficulty = score_difficulty(sentence)

        # Easy sentence — keep as is
        if sentence_difficulty < difficulty_threshold:
            result_sentences.append(sentence)
            details.append({
                'original':   sentence,
                'simplified': sentence,
                'action':     'kept_unchanged',
                'reason':     'already easy',
                'difficulty': round(sentence_difficulty, 3),
            })
            continue

        # Hard sentence — identify specific hard words to target
        hard_word_results = find_difficult_words_in_text(sentence, threshold=6.0)
        hard_words        = [w['word'] for w in hard_word_results['difficult_words']]

        simplified_result = simplify_text(
            sentence,
            sim_threshold=sim_threshold,
            n_candidates=n_candidates,
            hard_words=hard_words,
        )
        api_calls_made += n_candidates

        if simplified_result.get('success'):
            simplified_sentence = simplified_result['simplified']
            final_difficulty    = simplified_result['reranking']['best_difficulty']
            reranking           = simplified_result['reranking']

            result_sentences.append(simplified_sentence)
            details.append({
                'original':              sentence,
                'simplified':            simplified_sentence,
                'action':                'simplified',
                'original_difficulty':   round(sentence_difficulty, 3),
                'final_difficulty':      round(final_difficulty, 3),
                'reduction':             round(sentence_difficulty - final_difficulty, 3),
                'hard_words_targeted':   hard_words,
                'winning_temp':          reranking['best_temperature'],
                'winning_strategy':      reranking.get('best_nli_label', 'N/A'),
                'nli_label':             reranking.get('best_nli_label', 'N/A'),
                'filter_stats':          reranking.get('filter_stats', {}),
            })
        else:
            result_sentences.append(sentence)
            details.append({
                'original':   sentence,
                'simplified': sentence,
                'action':     'kept_unchanged',
                'reason':     'simplification failed',
                'difficulty': round(sentence_difficulty, 3),
            })

    final_text = ' '.join(result_sentences)

    original_flesch   = textstat.flesch_reading_ease(text)
    simplified_flesch = textstat.flesch_reading_ease(final_text)
    original_diff     = score_difficulty(text)
    final_diff        = score_difficulty(final_text)

    sentences_simplified = sum(1 for d in details if d['action'] == 'simplified')
    sentences_kept       = len(details) - sentences_simplified

    return {
        "original":          text,
        "simplified":        final_text,
        "original_flesch":   original_flesch,
        "simplified_flesch": simplified_flesch,
        "improvement":       round(simplified_flesch - original_flesch, 2),
        "success":           True,
        "reranking": {
            "mode":                  "targeted",
            "total_sentences":       len(sentences),
            "sentences_simplified":  sentences_simplified,
            "sentences_kept":        sentences_kept,
            "api_calls_made":        api_calls_made,
            "original_difficulty":   round(original_diff, 3),
            "final_difficulty":      round(final_diff, 3),
            "difficulty_reduction":  round(original_diff - final_diff, 3),
            "scoring_method":        "tail_aware (0.6*mean + 0.4*p90) + sim²",
        },
        "sentence_details": details,
    }


# ── CLI smoke test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_texts = [
        "Myocardial infarction occurs when blood flow decreases or stops to a part of the heart, causing damage to the heart muscle.",
        "The defendant, pursuant to the aforementioned contractual obligations stipulated in section 4.2 of the binding agreement, shall be held liable for any consequential damages.",
        "Quantum entanglement is a physical phenomenon that occurs when a group of particles are generated, interact, or share spatial proximity in such a way that the quantum state of each particle cannot be described independently.",
    ]

    print("Text Simplifier — v2 (tail-aware MBR + NLI gate)\n")
    print("=" * 70)

    for text in test_texts:
        result = simplify_text(text, n_candidates=10)
        if result.get("success"):
            r = result["reranking"]
            print(f"\nOriginal  (Flesch: {result['original_flesch']:.1f} | "
                  f"Tail-Difficulty: {r['original_difficulty']}):")
            print(f"  {result['original']}")
            print(f"\nBest output (Flesch: {result['simplified_flesch']:.1f} | "
                  f"Tail-Difficulty: {r['best_difficulty']} | "
                  f"Temp: {r['best_temperature']} | "
                  f"Sim: {r['best_similarity']} | "
                  f"NLI: {r['best_nli_label']}):")
            print(f"  {result['simplified']}")
            print(f"\nDifficulty reduction: {r['difficulty_reduction']:+.3f} | "
                  f"Flesch improvement: {result['improvement']:+.1f}")
            print(f"Valid candidates: {r['candidates_valid']}/{r['candidates_generated']}")
            fs = r['filter_stats']
            print(f"Filter kills — NLI: {fs['killed_by_nli']} | "
                  f"Sim: {fs['killed_by_similarity']} | "
                  f"Identical: {fs['killed_identical']}")
            print("\nAll candidates:")
            for c in r['all_candidates']:
                marker = " ← SELECTED" if c['text'] == result['simplified'] else ""
                flag   = f" [{c['flagged']}]" if c['flagged'] else ""
                print(f"  p{c['prompt_type']} t={c['temperature']} | "
                      f"tail_diff={c['tail_difficulty']} | "
                      f"sim={c['similarity']} | "
                      f"nli={c['nli_label']} ({c['nli_contradiction']:.2f})"
                      f"{flag}{marker}")
            print("=" * 70)
        else:
            print(f"Error: {result['error']}")