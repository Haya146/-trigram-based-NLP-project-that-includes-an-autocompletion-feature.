import nltk
from collections import defaultdict, Counter
nltk.download('punkt')

# Sample corpus
with open("corps.txt", "r", encoding="utf-8") as file:
    corpus = file.read()
tokens = nltk.word_tokenize(corpus.lower())

# Function to generate n-grams
def generate_ngrams(tokens, n):
    ngrams = zip(*[tokens[i:] for i in range(n)])
    return [" ".join(ngram) for ngram in ngrams]

# Generate trigrams
trigrams = generate_ngrams(tokens, 3)
print("Trigrams:", trigrams)

# Count trigrams
trigram_counts = Counter(trigrams)
print("Trigram Counts:", trigram_counts)

# حساب تكرارات التتابعين الأوليين لكل تتابع ثلاثي
bigram_counts = defaultdict(int)
for trigram in trigrams:
    words = trigram.split()
    bigram = (words[0], words[1])  # استخراج أول كلمتين فقط
    bigram_counts[bigram] += 1

# حساب احتمالات التتابعات الثلاثية
trigram_probabilities = {}
for trigram, count in trigram_counts.items():
    words = trigram.split()
    bigram = (words[0], words[1])
    trigram_probabilities[trigram] = count / bigram_counts[bigram]

# تطبيق Laplace Smoothing
vocab_size = len(set(tokens))
laplace_trigram_probabilities = {}

for trigram, count in trigram_counts.items():
    words = trigram.split()
    bigram = (words[0], words[1])
    laplace_trigram_probabilities[trigram] = (count + 1) / (bigram_counts[bigram] + vocab_size)

import math

def calculate_perplexity(sentence):
    sentence_tokens = nltk.word_tokenize(sentence.lower())
    N = len(sentence_tokens) - 2
    perplexity = 1
    for i in range(N):
        trigram = " ".join(sentence_tokens[i:i+3])
        prob = laplace_trigram_probabilities.get(trigram, 1 / vocab_size)  # تطبيق التنعيم
        perplexity *= 1 / prob
    perplexity = math.pow(perplexity, 1 / N)
    return perplexity

# Streamlit app code
import streamlit as st

st.title("Trigram Language Model")
input_sentence = st.text_input("Enter the beginning of a sentence:")

if st.button("Predict Next Word"):
    sentence_tokens = nltk.word_tokenize(input_sentence.lower())
    if len(sentence_tokens) >= 2:
        last_bigram = (sentence_tokens[-2], sentence_tokens[-1])
        predictions = {trigram.split()[2]: prob for trigram, prob in laplace_trigram_probabilities.items() if tuple(trigram.split()[:2]) == last_bigram}
        if predictions:
            sorted_predictions = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
            st.write("Suggestions:", [word for word, _ in sorted_predictions[:5]])
        else:
            st.write("No suggestions found.")
    else:
        st.write("Please enter at least two words.")
