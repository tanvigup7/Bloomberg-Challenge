# utils.py
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
import nltk
import pandas as pd

# Ensure nltk tokenizers are downloaded
nltk.download("punkt")

def visualize_word_cloud(data, column="text"):
    if column not in data.columns:
        print(f"Column '{column}' not found in data for word cloud generation.")
        return
    
    text = " ".join(data[column].dropna())
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)
    
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title("Word Cloud of Text Data")
    plt.show()

def visualize_word_frequency(data, column="text"):
    if column not in data.columns:
        print(f"Column '{column}' not found in data for word frequency visualization.")
        return
    
    all_words = []
    for text in data[column].dropna():
        words = nltk.word_tokenize(text.lower())
        all_words.extend(words)
    
    word_counts = Counter(all_words)
    common_words = word_counts.most_common(20)
    words, counts = zip(*common_words)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x=list(words), y=list(counts))
    plt.title("Top 20 Most Common Words")
    plt.xlabel("Words")
    plt.ylabel("Frequency")
    plt.xticks(rotation=45)
    plt.show()

def visualize_sentence_length(data, column="text"):
    if column not in data.columns:
        print(f"Column '{column}' not found in data for sentence length visualization.")
        return

    sentence_lengths = []
    for text in data[column].dropna():
        sentences = nltk.sent_tokenize(text)
        sentence_lengths.extend([len(sentence.split()) for sentence in sentences])
    
    plt.figure(figsize=(10, 6))
    sns.histplot(sentence_lengths, kde=True)
    plt.title("Distribution of Sentence Lengths")
    plt.xlabel("Number of Words in Sentence")
    plt.ylabel("Frequency")
    plt.show()
