import os, sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
from ai.models.difficulty_scorer import find_difficult_words_in_text
from ai.services.simplifier_groq import simplify_targeted

app = FastAPI(title="Dyslexia Assistant API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "Dyslexia Assistant API is running"}

class TextInput(BaseModel):
    text: str


def _analyze_text(text: str) -> dict:
    results    = find_difficult_words_in_text(text, threshold=4.5)
    all_scored = results["all_scored"]
    hard_words = results["difficult_words"]

    all_scores   = [w["difficulty_score"] for w in all_scored]
    total_words  = len(all_scores)
    hard_count   = len(hard_words)
    hard_density = round(hard_count / total_words * 100, 1) if total_words > 0 else 0.0
    p90_score    = round(float(np.percentile(all_scores, 90)), 2) if all_scores else 0.0
    max_word     = hard_words[0] if hard_words else None

    total_frequency     = sum(w["frequency_in_text"] for w in hard_words)
    weighted_difficulty = round(
        sum(w["difficulty_score"] * w["frequency_in_text"] for w in hard_words) / total_frequency, 2
    ) if total_frequency > 0 else 0.0

    if hard_density < 5:
        reading_level = "Easy"
    elif hard_density < 15:
        reading_level = "Moderate"
    elif hard_density < 30:
        reading_level = "Challenging"
    else:
        reading_level = "Very Difficult"

    return {
        "summary": {
            "total_content_words": total_words,
            "hard_word_count":     hard_count,
            "hard_word_density":   hard_density,
            "weighted_difficulty": weighted_difficulty,
            "p90_score":           p90_score,
            "hardest_word":        max_word["word"] if max_word else None,
            "hardest_word_score":  max_word["difficulty_score"] if max_word else None,
            "reading_level":       reading_level,
        },
        "hard_words": hard_words,
    }


@app.post("/analyze")
def analyze(input: TextInput):
    """Standalone analysis — used independently, not after /process."""
    analysis  = _analyze_text(input.text)
    return analysis


@app.post("/simplify")
def simplify(input: TextInput):
    return simplify_targeted(input.text)

DIFFICULTY_THRESHOLD = 4.5

@app.post("/process")
def process(input: TextInput):
    # Step 1 — analyze original
    original_analysis = _analyze_text(input.text)

    # Step 2 — simplify
    simplified = simplify_targeted(input.text, difficulty_threshold=DIFFICULTY_THRESHOLD)
    if not simplified.get("success"):
        return {"error": simplified.get("error"), "success": False}

    simplified_text = simplified["simplified"]

    # Step 3 — analyze simplified
    simplified_analysis = _analyze_text(simplified_text)

    # Step 4 — diff
    original_hard   = {w["word"]: w for w in original_analysis["hard_words"]}
    simplified_hard = {w["word"]: w for w in simplified_analysis["hard_words"]}

    eliminated = [w for w in original_hard if w not in simplified_hard]

    survived = [
        {
            "word":       word,
            "difficulty": simplified_hard[word]["difficulty_level"],
            "why":        simplified_hard[word]["reasons"][0] if simplified_hard[word]["reasons"] else "Rarely encountered word",
        }
        for word in original_hard if word in simplified_hard
    ]

    introduced = [
        {"word": word, "difficulty": simplified_hard[word]["difficulty_level"]}
        for word in simplified_hard if word not in original_hard
    ]

    # Step 5 — improvement summary (held by frontend, shown on Analyze click)
    before_level = original_analysis["summary"]["reading_level"]
    after_level  = simplified_analysis["summary"]["reading_level"]

    original_density  = original_analysis["summary"]["hard_word_density"]
    simplified_density = simplified_analysis["summary"]["hard_word_density"]
    density_reduction = round(original_density - simplified_density, 1)

    original_diff  = simplified["reranking"]["original_difficulty"]
    final_diff     = simplified["reranking"]["final_difficulty"]
    diff_reduction = round(original_diff - final_diff, 3)

    parts = []
    if eliminated:
        parts.append(f"replaced {len(eliminated)} difficult word(s)")
    if density_reduction > 0:
        parts.append(f"reduced hard word density by {density_reduction}%")
    if before_level != after_level:
        parts.append(f"reading level improved from {before_level} to {after_level}")

    if not parts:
        summary_message = f"No simplification needed. Reading level is already {after_level}."
    else:
        summary_message = "We " + ", and ".join(parts) + "."
    #else:
        #parts.append(f"reading level stayed at {after_level}")

    #summary_message = "We " + ", and ".join(parts) + "."

    if survived:
        summary_message += (
            f" {len(survived)} word(s) could not be fully simplified: "
            f"{', '.join(w['word'] for w in survived)}."
        )
    if introduced:
        summary_message += f" {len(introduced)} new word(s) were introduced."

    return {
        # Texts
        "original":   input.text,
        "simplified": simplified_text,

        # Word lists for highlighting
        "original_hard_words": [
            {
                "word":             w["word"],
                "syllables":        w.get("syllables_display", w["word"]),
                "definition":       w.get("definition", ""),
                "difficulty_level": w["difficulty_level"],
                "reasons":          w["reasons"],
            }
            for w in original_analysis["hard_words"]
        ],
        "survived_words":      [w["word"] for w in survived],

        # Shown immediately after process
        "original_analysis":   original_analysis,
        "simplified_analysis": simplified_analysis,

        # Held by frontend, rendered only when user clicks Analyze
        "improvement_summary": {
            "message":          summary_message,
            "before_level":     before_level,
            "after_level":      after_level,
            "words_eliminated": len(eliminated),
            "words_survived":   len(survived),
            "words_introduced": len(introduced),
            "density_reduction": density_reduction,
            "diff_reduction":   diff_reduction,
            "survived":         survived,    # for "words to watch" panel
            "introduced":       introduced,
        },
    }
    
@app.post("/define")
def define_word(input: TextInput):
    from nltk.corpus import wordnet
    word = input.text.lower().strip()
    
    synsets = wordnet.synsets(word)
    if synsets:
        # Pick shortest definition — least cognitive load
        definition = min(
            (s.definition() for s in synsets),
            key=lambda d: len(d.split())
        )
        return {"success": True, "word": input.text, "definition": definition}
    
    return {"success": False, "word": input.text, "definition": ""}