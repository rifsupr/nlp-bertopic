# =========================================
# BERTopic - SMALL DATASET (CUSTOM TUNED)
# Dataset: BBC short news (title + description)
# =========================================

import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer

# =========================
# LOAD DATASET
# =========================
df = pd.read_csv("datasets/bbc-dataset.csv")

# Gabungkan title + description
df['text'] = df['title'].fillna('') + ' ' + df['description'].fillna('')
df = df[['text']].dropna()

docs = df['text'].astype(str).tolist()

print(f"Total documents: {len(docs)}")

# =========================
# EMBEDDING MODEL (CRUCIAL)
# =========================
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# =========================
# VECTORIZER (IMPORTANT)
# =========================
vectorizer_model = CountVectorizer(
    stop_words="english",
    ngram_range=(1,2),
    min_df=1   # kecil karena dataset kecil
)

# =========================
# BUILD BERTopic MODEL
# =========================
topic_model = BERTopic(
    embedding_model=embedding_model,
    vectorizer_model=vectorizer_model,
    min_topic_size=2,   # penting untuk dataset kecil
    calculate_probabilities=True,
    verbose=True
)

# =========================
# TRAIN MODEL
# =========================
print("🧠 Training BERTopic...")
topics, probs = topic_model.fit_transform(docs)

# =========================
# OUTPUT: TOPIC INFO
# =========================
print("\n===== TOPIC INFO =====")
print(topic_model.get_topic_info())

# =========================
# DETAIL TOPIC
# =========================
print("\n===== TOPIC DETAILS =====")
for topic_id in topic_model.get_topics():
    if topic_id == -1:
        continue
    print(f"\nTopic {topic_id}:")
    print(topic_model.get_topic(topic_id))

# =========================
# ASSIGN TOPIC KE DOKUMEN
# =========================
df['topic'] = topics
print("\n===== DOCUMENT WITH TOPIC =====")
print(df[['text', 'topic']].head())

# =========================
# SAVE VISUALIZATION
# =========================
print("\n Saving visualization...")

topic_model.visualize_topics().write_html("output/topics.html")
topic_model.visualize_barchart().write_html("output/barchart.html")

print("Done!")