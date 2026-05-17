# Dyslexia Assistant — NLP Text Simplification System

An end-to-end NLP pipeline that scores word-level reading difficulty for dyslexic users and simplifies text using LLM-based generation with Minimum Bayes Risk (MBR) re-ranking.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Components](#components)
  - [Difficulty Model](#1-difficulty-model)
  - [Difficulty Scorer](#2-difficulty-scorer)
  - [Text Simplifier](#3-text-simplifier)
  - [API](#4-api)
- [Setup & Installation](#setup--installation)
- [API Reference](#api-reference)
- [Model Training](#model-training)
- [Design Decisions](#design-decisions)
- [Known Limitations & Future Work](#known-limitations--future-work)

---

## Overview

The Dyslexia Assistant tackles a two-stage problem:

1. **Difficulty Estimation** — Score each word in a text on a 0–10 dyslexia difficulty scale using a fine-tuned BERT + linguistic feature hybrid model.
2. **Text Simplification** — Replace difficult sentences with LLM-generated alternatives, selected via MBR re-ranking across a prompt × temperature grid.

The system is grounded in dyslexia research literature (Ziegler & Goswami 2005, Snowling 2000, Dehaene 2009) and uses a composite difficulty label derived from orthographic, phonological, and psycholinguistic signals.

---

## Architecture

```
Input Text
    │
    ▼
┌─────────────────────────────┐
│     /analyze endpoint       │
│  find_difficult_words_in_   │
│  text() → word-level scores │
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│     /simplify endpoint      │
│  simplify_targeted()        │
│  • Split into sentences     │
│  • Skip easy sentences      │
│  • Score hard sentences     │
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────┐
│              Candidate Generation                   │
│  5 prompts × 5 temperatures = 25-combo grid        │
│  Round-robin sampling → n_candidates generated     │
└────────────┬────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────┐
│              Candidate Filtering                    │
│  1. NLI contradiction gate (DeBERTa cross-encoder) │
│  2. Bi-encoder semantic similarity threshold        │
│  3. Must be simpler than original (tail difficulty) │
└────────────┬────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────┐
│              MBR Re-ranking                         │
│  score = 0.55 * tail_difficulty/10                 │
│        + 0.35 * (1 - similarity)²                  │
│        + 0.10 * length_penalty                     │
└────────────┬────────────────────────────────────────┘
             │
             ▼
        Best Candidate → /process response
```

---

## Components

### 1. Difficulty Model

**File:** `train_difficulty_model.ipynb`

A hybrid regression model trained to predict a composite dyslexia difficulty score (0–10) for individual English words.

#### Architecture

```
BERT (bert-base-uncased)
    │
    ├── [CLS] embedding (768-dim)
    │
    └── concat with 11 linguistic features
            │
            ▼
        Linear(779, 256) → ReLU → Dropout(0.4)
        Linear(256, 64)  → ReLU → Dropout(0.2)
        Linear(64, 1)    → scalar score
```

#### Training Data

- **SUBTLEX-US** — word frequency + part-of-speech
- **AoA_51715** — age of acquisition, phoneme count (Nphon), frequency per million
- **Brysbaert Concreteness Ratings** — abstract vs. concrete word axis
- **WordNet** — context sentences for BERT's word-in-context input

Dataset size after merge: ~30,384 words

#### Composite Difficulty Label

The target label is a weighted sum of research-backed features, not human ratings. This makes the label deterministic and the model achieve R² ≈ 0.98.

| Component | Weight | Research Basis |
|---|---|---|
| Irregular grapheme (ough, tion…) | 0.15 | Ziegler & Goswami (2005) |
| Consonant clusters | 0.13 | Snowling (2000) |
| Silent letters (kn, gh, mb…) | 0.10 | BDA research |
| Syllable count | 0.10 | WM capacity |
| Word frequency (inverse) | 0.12 | SUBTLEX |
| Concreteness (inverse) | 0.11 | Paivio dual coding theory |
| Confusable letters (b/d/p/q) | 0.08 | Dehaene (2009) |
| Age of acquisition | 0.08 | Kuperman norms |
| Word length | 0.08 | Visual crowding |
| Vowel ratio (inverse) | 0.05 | Phonological chunking |

#### Training Configuration

| Parameter | Value |
|---|---|
| Base model | bert-base-uncased |
| Bottom 6 BERT layers | frozen |
| Optimizer | AdamW (encoder lr=2e-5, head lr=1e-3) |
| Scheduler | Linear warmup (20% steps) |
| Batch size | 64 |
| Epochs | 15 |
| Best val R² | 0.9849 |
| Best val MSE | 0.0163 |

---

### 2. Difficulty Scorer

**File:** `difficulty_scorer.py`

Runtime inference module. Loads the trained model once and exposes a cached word-scoring API.

#### Linguistic Features (11 total)

```python
LINGUISTIC_COLS = [
    'word_length', 'syllable_count', 'confusable_letters',
    'vowel_ratio', 'has_silent_letter', 'has_irregular_grapheme',
    'consonant_clusters', 'Dom_PoS_SUBTLEX', 'Nphon', 'Freq_pm',
    'concreteness'
]
```

Lookup tables used at runtime:
- `nphon_lookup` — phoneme count from AoA dataset
- `freq_lookup` — frequency per million from AoA dataset
- `pos_lookup` — dominant PoS from SUBTLEX
- `conc_lookup` — concreteness rating from Brysbaert et al.

Frequency fallback: `wordfreq` library (covers OOV words)

#### Caching

```python
@lru_cache(maxsize=4096)
def _score_word_cached(word_lower: str):
    ...
```

LRU cache with 4096 entries. Avoids redundant BERT forward passes for repeated words across a document. Cache key is lowercased word.

#### Public API

```python
score_word_bert(word: str) -> dict
# Returns: difficulty_score (0-10), difficulty_level, reasons, features

find_difficult_words_in_text(text: str, threshold: float = 5.0) -> dict
# Returns: all_scored (every content word), difficult_words (above threshold)
```

Stop words, punctuation, and words ≤ 2 characters are skipped automatically.

---

### 3. Text Simplifier

**File:** `simplifier_groq.py`

MBR-based simplification engine using LLaMA-3.3-70B via Groq.

#### Difficulty Scoring: Tail-Aware Metric

```python
score = 0.6 * mean(word_scores) + 0.4 * percentile(word_scores, 90)
```

Standard mean-based scoring misses sentences where one buried hard word stalls a dyslexic reader. The p90 blend penalises candidates that preserve even one very hard word without going to the variance-unstable max.

#### Prompt × Temperature Grid

```
5 prompts × 5 temperatures = 25 candidate slots

Prompts:
  P1 — Word-level substitution only (conservative)
  P2 — Sentence restructuring (break long sentences)
  P3 — Full rewrite, grade-8 vocabulary
  P4 — Active voice conversion
  P5 — Concise rewrite, no padding

Temperatures: [0.1, 0.3, 0.7, 1.0, 1.2]
```

Combos are sampled in round-robin order across prompts, so all 5 strategies are represented regardless of `n_candidates`.

#### Candidate Filtering

**Stage 1 — NLI Contradiction Gate**

Model: `cross-encoder/nli-deberta-v3-small` (~86 MB, ~40 ms/pair on CPU)

```python
# Only BLOCK on contradiction — not require entailment
# Valid simplifications are NEUTRAL (paraphrase), not strictly entailed
if contradiction_score >= 0.6:
    flagged = 'nli_contradiction'
```

**Stage 2 — Semantic Similarity**

Model: `paraphrase-MiniLM-L3-v2` (bi-encoder, fast)

Threshold is adaptive — relaxed for longer/harder sentences:

```python
threshold = base - 0.10 * length_factor - 0.08 * difficulty_factor
# Hard floor: 0.45
```

**Stage 3 — Must Improve**

Candidate is only valid if `tail_difficulty < original_score`.

#### MBR Re-ranking

```python
combined_score(c) = 0.55 * (tail_difficulty / 10.0)
                  + 0.35 * (1 - similarity) ** 2   # squared penalty
                  + 0.10 * max(0, length_ratio - 1.0)
```

Squared similarity penalty: compresses cost for near-similar candidates, amplifies it for large semantic drift — matching how dyslexic readers experience paraphrased text.

#### Targeted Mode

`simplify_targeted()` routes sentences through the pipeline selectively:

- Sentences < 5 words → pass through unchanged
- Sentence difficulty < `difficulty_threshold` (default 3.5) → pass through
- Hard sentences → identify specific hard words → inject as CRITICAL targets into prompts → simplify

This avoids unnecessary Groq API calls on already-easy text.

---

### 4. API

**File:** `api.py`  
**Framework:** FastAPI

#### Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/` | Health check |
| POST | `/analyze` | Word-level difficulty analysis |
| POST | `/simplify` | Text simplification only |
| POST | `/process` | Full pipeline: analyze → simplify → diff |

#### `/analyze` Response

```json
{
  "summary": {
    "total_content_words": 42,
    "hard_word_count": 8,
    "hard_word_density": 19.0,
    "weighted_difficulty": 6.83,
    "p90_score": 7.41,
    "hardest_word": "indemnification",
    "hardest_word_score": 8.23,
    "reading_level": "Challenging"
  },
  "hard_words": [...]
}
```

Reading level thresholds:

| hard_word_density | Level |
|---|---|
| < 5% | Easy |
| 5–15% | Moderate |
| 15–30% | Challenging |
| ≥ 30% | Very Difficult |

#### `/process` Response (developer mode)

```json
{
  "original": "...",
  "simplified": "...",
  "verdict": {
    "label": "Highly effective — ...",
    "hard_words_eliminated": 6,
    "hard_words_survived": 2,
    "hard_words_introduced": 1,
    "density_reduction": "18.5%",
    "difficulty_reduction": 1.72
  },
  "diff": {
    "eliminated": ["myocardial", "infarction"],
    "survived": [{"word": "cardiac", ...}],
    "introduced": ["emergency"]
  }
}
```

Verdict thresholds:

| Condition | Verdict |
|---|---|
| reduction ≥ 1.5 AND density drop ≥ 20% | Highly effective |
| reduction ≥ 0.8 OR density drop ≥ 15% | Effective |
| reduction ≥ 0.3 OR density drop ≥ 5% | Moderate |
| introduced > eliminated | Ineffective |
| else | Minimal |

---

## Setup & Installation

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (optional, CPU works for inference)
- Groq API key

### Install Dependencies

```bash
pip install fastapi uvicorn groq sentence-transformers transformers \
            torch pyphen wordfreq textstat pandas openpyxl scikit-learn \
            python-dotenv nltk
```

### Data Files Required

Place these in `ai/data/`:

| File | Source |
|---|---|
| `SUBTLEX-US frequency list with PoS and Zipf information.csv` | [SUBTLEX-US](http://crr.ugent.be/programs-data/subtitle-frequencies/subtlex-us) |
| `AoA_51715_words.xlsx` | [Kuperman AoA norms](http://crr.ugent.be/archives/806) |
| `concreteness.txt` | [Brysbaert et al.](https://github.com/ArtsEngine/concreteness) |

### Model Weights

Place `difficulty_model.pt` and `difficulty_config.json` in `ai/models/`.

To train from scratch, run `train_difficulty_model.ipynb` on Colab (T4 GPU, ~55 min).

### Environment Variables

```bash
# .env
GROQ_API_KEY=your_groq_api_key_here
```

### Run the API

```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

---

## API Reference

### POST `/analyze`

```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "Myocardial infarction occurs when blood flow decreases."}'
```

### POST `/simplify`

```bash
curl -X POST http://localhost:8000/simplify \
  -H "Content-Type: application/json" \
  -d '{"text": "The defendant, pursuant to the aforementioned contractual obligations..."}'
```

### POST `/process`

```bash
# Developer view (full diff)
curl -X POST "http://localhost:8000/process" \
  -H "Content-Type: application/json" \
  -d '{"text": "...", "user_view": false}'

# User view (plain language summary)
curl -X POST "http://localhost:8000/process" \
  -H "Content-Type: application/json" \
  -d '{"text": "...", "user_view": true}'
```

---

## Model Training

To retrain the difficulty model:

1. Open `train_difficulty_model.ipynb` in Google Colab
2. Upload the three data files (SUBTLEX, AoA, concreteness)
3. Run all cells (T4 GPU, ~55 min for 15 epochs)
4. Download `difficulty_model.pt` and `difficulty_config.json`
5. Place in `ai/models/`

Training outputs checkpointed at best val R² epoch.

---

## Design Decisions

### Why a Composite Label Instead of Human Ratings?

Human AoA/difficulty ratings introduce annotation noise and are sparse. The composite label is deterministic, covers all 30K words without gaps, and is directly interpretable in terms of research-validated dyslexia difficulty factors. The trade-off is that the model learns to predict a formula — not actual human reading difficulty — but the formula itself is grounded in peer-reviewed research.

### Why Tail-Aware Difficulty (p90 Blend)?

Dyslexic readers don't struggle with the average word — they stall on the hardest word in a phrase. A sentence scoring mean=3.5 with one word at 8.5 is not simpler than one scoring mean=4.0 with max=5.0. The p90 blend (rather than max) avoids oversensitivity to unavoidable proper nouns or technical terms.

### Why NLI as a Gate (Not a Score)?

Valid simplifications are paraphrases — semantically NEUTRAL, not ENTAILED. Requiring entailment would eliminate most good candidates. Blocking only CONTRADICTION (threshold=0.6) catches factual inversions ("patient recovered" → "patient died") that cosine similarity misses (~0.85 similarity on a medically opposite statement).

### Why Squared Similarity Penalty in MBR?

Linear penalty: small and large similarity gaps cost proportionally.  
Squared penalty: forgives minor drift (0.89 vs 0.87), strongly punishes large drift (0.90 vs 0.70). This matches the reading experience — slightly rephrased text is fine, heavily restructured text is disorienting.

### Why Adaptive Similarity Threshold?

Short simple sentences need minimal change → tight threshold.  
Long complex sentences inherently require structural changes that move embeddings → the bi-encoder threshold is relaxed. Without adaptation, the filter would kill all valid rewrites of long sentences.

---

## Known Limitations & Future Work

**Difficulty Model**
- MRC imageability/familiarity and Warriner valence/arousal use neutral fallbacks — integrating these would improve accuracy for emotional and abstract vocabulary
- Model trained on SUBTLEX-US → may underperform on domain-specific corpora (medical, legal)
- No cross-lingual support

**Simplifier**
- Groq API latency dominates end-to-end time; 10 candidates ≈ 10–15s per sentence
- NLI model runs on CPU; moving to GPU halves the gate latency
- No caching of candidate outputs across similar sentences
- Prompt suite is English-only

**Evaluation**
- No human evaluation loop; difficulty reduction is measured against the same composite formula used for training
- No A/B testing with actual dyslexic readers
- SARI score (precision/recall on word changes) not currently computed

**Personalization**
- No per-user difficulty calibration — threshold is global (5.0)
- No reading history or adaptive difficulty profile
