import numpy as np 
import pandas as pd 
from sentence_transformers import SentenceTransformer

print("Python environment is working!")
      
model = SentenceTransformer('all-MiniLM-L6-v2')

sentences = ["Apple acquired Beats.",
             "The weather is sunny today."
             ]

embeddings = model.encode(sentences)

print("Embedding shape:", embeddings.shape)
print("Setup successful!")