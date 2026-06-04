from difficulty_scorer import score_word_bert
for w in ["because", "children", "difficulty", "phenomenon", 
          "myocardial", "catastrophic", "the", "understand", "mysterious", "associated "]:
    r = score_word_bert(w)
    print(f"{r['word']:20s} score={r['difficulty_score']} level={r['difficulty_level']}")

