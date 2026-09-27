import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

sentence = ["My name is aditi",
            "I am from raipur",
            "I have studied in Delhi University",
"I am currently coding in python and learning RAG"]


embeddings = model.encode(sentence)

print("embeddings shape :" , embeddings.shape )
print("embeddings :" , embeddings)


query = "where is aditi from?"

query_embedding = model.encode(query)

print("query embedding shape :" , query_embedding.shape )
print("query embedding :" , query_embedding)