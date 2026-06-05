"""
difficulty_scorer.py  (v2)
--------------------------
Matches the retrained model exactly:
  - 15 linguistic features (up from 11)
  - Self-contained: no CSV/Excel at runtime — loads pkl lookup dicts only
  - count_syllables_final:  CMU → pyphen → vowel-group fallback
  - count_nphon_robust:     CMU → grapheme-to-phoneme approximation
  - New features: ortho_n, avg_bigram_freq, morpheme_count, zipf_score, aoa, pos_code
  - Dropped: Freq_pm (SUBTLEX subtitle freq), mrc_imag, mrc_fam, valence, arousal
  - Bucket boundaries recalibrated to v2 label distribution (mean ~4.37)

Public API — unchanged, drop-in replacement:
  score_word_bert(word)              → dict
  find_difficult_words_in_text(text) → dict
"""

import re
import os
import json
import pickle
import numpy as np
import pyphen
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from wordfreq import zipf_frequency
from functools import lru_cache
import nltk
nltk.download('wordnet',  quiet=True)
nltk.download('omw-1.4', quiet=True)
nltk.download('cmudict',  quiet=True)
from nltk.corpus import wordnet, cmudict

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR = os.path.join(BASE_DIR, 'ai', 'models')

WEIGHTS_PATH      = os.path.join(MODEL_DIR, 'difficulty_model_v2.pt')
CONFIG_PATH       = os.path.join(MODEL_DIR, 'difficulty_config_v2.json')
ORTHO_N_PATH      = os.path.join(MODEL_DIR, 'ortho_n_lookup.pkl')
BIGRAM_FREQ_PATH  = os.path.join(MODEL_DIR, 'bigram_freq_lookup.pkl')
POS_PATH          = os.path.join(MODEL_DIR, 'pos_lookup.pkl')
AOA_PATH          = os.path.join(MODEL_DIR, 'aoa_lookup.pkl')
CONC_PATH         = os.path.join(MODEL_DIR, 'concreteness_lookup.pkl')

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ── Model definition — must match training exactly ────────────────────────────
class WordDifficultyModel(nn.Module):
    def __init__(self, transformer_name, n_linguistic):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(transformer_name)
        hidden       = self.encoder.config.hidden_size
        self.head = nn.Sequential(
            nn.Linear(hidden + n_linguistic, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, input_ids, attention_mask, linguistic):
        cls = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        ).last_hidden_state[:, 0, :]
        return self.head(torch.cat([cls, linguistic], dim=1)).squeeze(-1)

# ── Load model + config ───────────────────────────────────────────────────────
print("Loading difficulty model v2...")
checkpoint        = torch.load(WEIGHTS_PATH, map_location=DEVICE)
config            = json.load(open(CONFIG_PATH))

_transformer_name = checkpoint['transformer_name']
_n_linguistic     = checkpoint['n_linguistic']       # 15
_max_len          = config['max_len']                # 64
_scaler_mean      = np.array(checkpoint['scaler_mean'],  dtype=np.float32)
_scaler_scale     = np.array(checkpoint['scaler_scale'], dtype=np.float32)

_model = WordDifficultyModel(_transformer_name, _n_linguistic).to(DEVICE)
_model.load_state_dict(checkpoint['model_state_dict'])
_model.eval()
_tokenizer = AutoTokenizer.from_pretrained(_transformer_name)
print("Model loaded.")

# ── Load lookup dicts (pkl only — no CSV/Excel at runtime) ────────────────────
print("Loading lookup dicts...")
with open(ORTHO_N_PATH,     'rb') as f: _ortho_n_lookup     = pickle.load(f)
with open(BIGRAM_FREQ_PATH, 'rb') as f: _bigram_freq_lookup = pickle.load(f)
with open(POS_PATH,         'rb') as f: _pos_lookup         = pickle.load(f)
with open(AOA_PATH,         'rb') as f: _aoa_lookup         = pickle.load(f)
with open(CONC_PATH,        'rb') as f: _conc_lookup        = pickle.load(f)
print("Lookups loaded.")

# ── Constants ─────────────────────────────────────────────────────────────────
STOP_WORDS = {
    'the','a','an','and','or','but','in','on','at','to','for',
    'of','with','by','from','is','was','are','were','be','been',
    'being','have','has','had','do','does','did','will','would',
    'could','should','may','might','shall','can','need','dare',
    'this','that','these','those','it','its','they','them','their',
    'he','she','we','you','i','my','your','his','her','our',
    'not','no','nor','so','yet','both','either','neither',
    'as','if','then','than','when','while','although','because',
    'into','onto','upon','about','above','below','between','through'
}

SILENT_PATTERNS    = ['kn','wr','gh','mb','bt','mn','cht','lm','sw','gn','ps','rh']
IRREGULAR_PATTERNS = ['ough','aigh','eigh','tion','sion','olo','queue','quay',
                       'eur','ieu','eau','ph','sch','chr']

POS_MAP = {
    'Noun':11,'Verb':12,'Adjective':13,'Adverb':14,
    'Preposition':5,'Conjunction':6,'Pronoun':7,
    'Article':8,'Determiner':9,'Number':10,
    'Interjection':11,'Name':12,'Unknown':0
}

# ── AoA fallback: linear approximation fitted during dataset build ─────────────
# Coefficients: AoA = 16.264 + (−1.825 × zipf)  — from notebook Cell 5
_AOA_INTERCEPT = 16.264
_AOA_SLOPE     = -1.825

# ── Shared objects ────────────────────────────────────────────────────────────
_dic = pyphen.Pyphen(lang='en')
_cmu = cmudict.dict()

# ── Feature functions — identical to notebook Cell 4 ─────────────────────────

def count_syllables_final(word: str) -> int:
    """CMU dict → pyphen → vowel-group fallback."""
    w = word.lower().strip()
    if not w:
        return 1
    # CMU: count vowel phonemes (phoneme strings ending in a digit)
    entries = _cmu.get(w)
    if entries:
        return max(sum(1 for ph in entries[0] if ph[-1].isdigit()), 1)
    # pyphen
    try:
        result = len(_dic.inserted(w).split('-'))
        if result > 1:
            return result
    except Exception:
        pass
    # vowel-group fallback — treat consonant-flanked y as vowel
    temp = re.sub(r'([^aeiou])y([^aeiou])', r'\1i\2', w)
    temp = re.sub(r'([^aeiou])y$', r'\1i', temp)
    groups = re.findall(r'[aeiou]+', temp)
    count  = len(groups)
    # silent e: consonant-consonant-e at end only
    if (w.endswith('e') and len(w) > 3
            and w[-2] not in 'aeiou'
            and w[-3] not in 'aeiou'
            and count > 1):
        count -= 1
    return max(count, 1)


def count_nphon_robust(word: str) -> int:
    """CMU dict phoneme count → grapheme-to-phoneme approximation."""
    entries = _cmu.get(word.lower())
    if entries:
        return len(entries[0])
    w     = word.lower()
    multi = ['ough','aigh','eigh','tion','sion','ph','th','ch',
             'sh','wh','ck','ng','qu','gh','kn','wr']
    count, i = 0, 0
    while i < len(w):
        matched = False
        for pat in sorted(multi, key=len, reverse=True):
            if w[i:i+len(pat)] == pat:
                count  += 1
                i      += len(pat)
                matched = True
                break
        if not matched:
            if w[i] not in ('e',) or i < len(w) - 1:
                count += 1
            i += 1
    return max(count, 1)


def _count_consonant_clusters(word: str) -> int:
    return len(re.findall(r'[bcdfghjklmnpqrstvwxyz]{2,}', word.lower()))


def _count_morphemes(word: str) -> int:
    PREFIXES = ['un','re','pre','mis','dis','over','under','out','up']
    SUFFIXES = ['tion','sion','ness','ment','ity','ous','ful','less',
                'ing','ed','er','est','ly','al','ic','ize','ise',
                'able','ible','ance','ence']
    w, count = word.lower(), 1
    for p in PREFIXES:
        if w.startswith(p) and len(w) - len(p) > 2:
            count += 1
            w = w[len(p):]
            break
    for s in SUFFIXES:
        if w.endswith(s) and len(w) - len(s) > 2:
            count += 1
            break
    return count


def _get_context(word: str) -> str:
    synsets = wordnet.synsets(word)
    if not synsets:
        return word
    for s in synsets:
        if s.examples():
            return s.examples()[0]
    return synsets[0].definition()


def syllabify(word: str) -> str:
    """Return dot-separated syllable display, e.g. 'cat·a·stroph·ic'."""
    if not isinstance(word, str) or not word.strip():
        return word
    w = word.lower()
    # Use CMU to get syllable breaks accurately
    entries = _cmu.get(w)
    if entries:
        # Fall back to pyphen for display — CMU gives phonemes not orthographic breaks
        pass
    try:
        return _dic.inserted(word).replace('-', '·')
    except Exception:
        return word


def should_skip_word(word: str) -> bool:
    if len(word) <= 2:
        return True
    if word.lower() in STOP_WORDS:
        return True
    if not re.match(r"^[a-zA-Z][a-zA-Z'\-]*[a-zA-Z]$", word):
        return True
    return False


# ── Build + normalise 15-feature vector ──────────────────────────────────────

def _build_linguistic(word: str):
    """
    Returns (normalised_vector: np.ndarray[15], feature_dict: dict).

    Feature order must match LINGUISTIC_COLS from the notebook exactly:
    word_length, syllable_count, confusable_letters, vowel_ratio,
    silent_letter_count, irregular_grapheme_count, consonant_clusters,
    nphon, ortho_n, avg_bigram_freq, morpheme_count,
    zipf_score, aoa, concreteness, pos_code
    """
    w = word.lower()
    length     = len(word)
    syllables  = count_syllables_final(word)
    confusable = sum(w.count(c) for c in 'bdpq')
    vowels     = sum(1 for c in w if c in 'aeiou')
    vowel_ratio = vowels / length if length > 0 else 0.0

    silent_count   = sum(1 for p in SILENT_PATTERNS    if p in w)
    irregular_count = sum(1 for p in IRREGULAR_PATTERNS if p in w)
    clusters       = _count_consonant_clusters(word)
    nphon          = count_nphon_robust(word)
    morphemes      = _count_morphemes(word)

    # Lookup-based features — fallback values match training-time fills
    ortho_n      = _ortho_n_lookup.get(w, 0)
    bigram_freq  = _bigram_freq_lookup.get(w, 0.005)   # ~median of training dist
    zipf         = zipf_frequency(w, 'en')
    aoa          = _aoa_lookup.get(w, float(
        np.clip(_AOA_INTERCEPT + _AOA_SLOPE * zipf, 1.0, 17.5)
    ))
    concreteness = _conc_lookup.get(w, 2.5)
    pos_raw      = _pos_lookup.get(w, _pos_lookup.get(word, 'Unknown'))
    pos_code     = POS_MAP.get(pos_raw if isinstance(pos_raw, str) else 'Unknown', 0)

    raw = np.array([
        length, syllables, confusable, vowel_ratio,
        silent_count, irregular_count, clusters,
        nphon, ortho_n, bigram_freq, morphemes,
        zipf, aoa, concreteness, pos_code
    ], dtype=np.float32)

    normalised = (raw - _scaler_mean) / _scaler_scale

    return normalised, {
        'syllables':          int(syllables),
        'length':             int(length),
        'confusable_letters': int(confusable),
        'silent_count':       int(silent_count),
        'irregular_count':    int(irregular_count),
        'clusters':           int(clusters),
        'zipf':               round(float(zipf), 3),
    }


# ── Cached inference ──────────────────────────────────────────────────────────

@lru_cache(maxsize=4096)
def _score_word_cached(word_lower: str):
    ling_vec, ling_dict = _build_linguistic(word_lower)
    context             = _get_context(word_lower)

    enc = _tokenizer(
        word_lower,
        context,
        max_length=_max_len,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    input_ids      = enc['input_ids'].to(DEVICE)
    attention_mask = enc['attention_mask'].to(DEVICE)
    ling_tensor    = torch.tensor(ling_vec, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        difficulty = _model(input_ids, attention_mask, ling_tensor).item()

    difficulty = float(np.clip(difficulty, 0.0, 10.0))
    return difficulty, ling_dict


# ── Public API ────────────────────────────────────────────────────────────────

def score_word_bert(word: str) -> dict:
    """
    Score a single word for dyslexic reading difficulty.

    Returns:
        word                str
        difficulty_score    float  0–10
        difficulty_level    str    Easy / Medium / Hard / Very Hard
        aoa                 float  backwards-compat alias for difficulty_score
        reasons             list[str]
        syllables_display   str    dot-separated syllables
        features            dict   raw feature values for inspection
    """
    word_lower            = word.lower()
    difficulty, ling_dict = _score_word_cached(word_lower)

    syllables  = ling_dict['syllables']
    length     = ling_dict['length']
    confusable = ling_dict['confusable_letters']
    silent     = ling_dict['silent_count']
    irregular  = ling_dict['irregular_count']
    clusters   = ling_dict['clusters']
    zipf       = ling_dict['zipf']

    # ── Difficulty level — recalibrated to v2 label distribution ─────────────
    # v2 mean=4.37, std=1.03 (vs v1 mean=3.91).
    # Boundaries shifted up by 0.5 at the hard end to avoid over-flagging.
    if difficulty < 2.5:
        level = "Easy"
    elif difficulty < 5.0:
        level = "Medium"
    elif difficulty < 7.0:
        level = "Hard"
    else:
        level = "Very Hard"

    # ── Human-readable reasons ────────────────────────────────────────────────
    reasons = []
    if syllables >= 4:
        reasons.append(f"{syllables} syllables")
    if length > 10:
        reasons.append(f"Long word ({length} letters)")
    if confusable >= 3:
        reasons.append(f"Contains {confusable} confusable letters (b/d/p/q)")
    if silent >= 1:
        reasons.append("Has silent letters")
    if irregular >= 1:
        reasons.append("Irregular spelling pattern")
    if clusters >= 2:
        reasons.append(f"{clusters} consonant clusters")
    if zipf < 2.0:
        reasons.append("Rarely encountered word")
    elif zipf < 3.5:
        reasons.append("Uncommon in everyday reading")
    if difficulty >= 7.0 and not reasons:
        reasons.append("Complex combination of difficulty factors")

    return {
        'word':              word,
        'difficulty_score':  round(difficulty, 2),
        'difficulty_level':  level,
        'aoa':               round(difficulty, 2),   # backwards compat
        'reasons':           reasons,
        'syllables_display': syllabify(word),
        'features': {
            'syllables':          syllables,
            'length':             length,
            'confusable_letters': confusable,
            'silent_letters':     silent,
            'irregular_graphemes':irregular,
            'consonant_clusters': clusters,
            'zipf':               zipf,
        }
    }


def find_difficult_words_in_text(text: str, threshold: float = 5.0) -> dict:
    """
    Identical signature and return structure to v1 — drop-in replacement.

    Returns:
        all_scored      list[dict]  every content word scored
        difficult_words list[dict]  words at or above threshold, sorted desc
    """
    all_words = re.findall(r'\b[a-zA-Z]+\b', text)

    # Count frequency of each content word in this text
    word_freq: dict[str, int] = {}
    for word in all_words:
        key = word.lower()
        if not should_skip_word(word):
            word_freq[key] = word_freq.get(key, 0) + 1

    all_scored:      list[dict] = []
    difficult_words: list[dict] = []
    seen:            set[str]   = set()

    for word in all_words:
        word_lower = word.lower()
        if word_lower in seen or should_skip_word(word):
            continue
        seen.add(word_lower)

        result = score_word_bert(word)
        result['frequency_in_text'] = word_freq.get(word_lower, 1)
        all_scored.append(result)

        if result['difficulty_score'] >= threshold:
            difficult_words.append(result)

    difficult_words.sort(key=lambda x: x['difficulty_score'], reverse=True)
    return {
        'all_scored':      all_scored,
        'difficult_words': difficult_words,
    }


# ── Smoke test ────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    test_words = [
        'the', 'cat', 'fire', 'table', 'knight', 'though',
        'understand', 'philosophy', 'straightforward',
        'myocardial', 'deoxyribonucleic', 'jurisdiction',
    ]
    print(f"\n{'word':25s} {'score':>6}  {'level':>10}  reasons")
    print('-' * 70)
    for w in test_words:
        r = score_word_bert(w)
        print(f"{r['word']:25s} {r['difficulty_score']:>6.2f}  "
              f"{r['difficulty_level']:>10}  {r['reasons']}")
