import pandas as pd
from nltk.corpus import wordnet, words as nltk_words
from wordfreq import zipf_frequency
import nltk

# Source 1 — SUBTLEX (frequency + POS)
subtlex = pd.read_csv('SUBTLEX-US frequency list with PoS and Zipf information.csv')
subtlex_words = set(subtlex['Word'].str.lower().dropna())

# Source 2 — WordNet (broad vocabulary + definitions for context)
wordnet_words = set(w.lower() for w in wordnet.words())

# Source 3 — Brysbaert concreteness (40K words, high quality ratings)
brysbaert = pd.read_csv('concreteness.txt', sep='\t')
brysbaert_words = set(brysbaert['Word'].str.lower())

# Union of all sources
all_words = subtlex_words | wordnet_words | brysbaert_words
print(f"Total unique words: {len(all_words):,}")

# Filter 1 — only real alphabetic words
all_words = {w for w in all_words 
             if w.isalpha() and len(w) > 2}

# Filter 2 — minimum zipf threshold
# zipf < 1.5 means the word appears less than once per 30 million words
# These are so rare that no frequency dataset reliably covers them
# and your users will almost never encounter them
all_words = {w for w in all_words 
             if zipf_frequency(w, 'en') >= 1.5}

print(f"After filtering: {len(all_words):,}")

# Build base dataframe
df = pd.DataFrame({'Word': sorted(all_words)})