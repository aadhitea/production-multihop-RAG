import pandas as pd
import numpy as np
# to generate embeddings for sentences and queries.
from sentence_transformers import SentenceTransformer
# calculate cosine similarity between query and document embeddings
from sklearn.metrics.pairwise import cosine_similarity

# Initialize the SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# a list of sentences to be embedded
sentence = ["My name is aditi",
            "I am from raipur",
            "I have studied in Delhi University",
"I am currently coding in python and learning RAG"
]
# The sentence embeddings are generated using the SentenceTransformer model. The embeddings are normalized to ensure that the cosine similarity calculations are accurate.
embeddings = model.encode(sentence, normalize_embeddings=True)

# user's query
query1 = "where has aditi studied from?"

# user's query is encoded to get the query embedding
query_embeddings = model.encode([query1], normalize_embeddings=True)

# Calculate similarity between query and every document
similarities = cosine_similarity(query_embeddings, embeddings)[0]

for doc, score in zip(sentence, similarities):
    print(f"Document: {doc} | Similarity Score: {score:.4f}")

top_index = similarities.argmax()
print("most similar doc to query is : ", sentence[top_index])
print("similarity score is :", similarities[top_index])

# print("query_embeddings shape :" , query_embeddings.shape )
# print("query_embeddings :" , query_embeddings)

top_k = 2 

top_indices = similarities.argsort()[-top_k:][::-1]

for i in top_indices:
    print(f"Top {top_k} similar document: {sentence[i]} | Similarity Score: {similarities[i]:.4f}")