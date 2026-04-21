from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

embedding =HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

texts = [
    "Artificial intelligence is transforming the way we interact with technology.",
    "Machine learning allows computers to learn patterns from data without explicit programming.",
    "The capital of India is New Delhi, which is known for its rich history and culture.",
    "Python is a popular programming language used for data science and machine learning."
]
query = 'tell me about india'

texts_embeddings=embedding.embed_documents(texts)
query_embedding=embedding.embed_query(query)

score =cosine_similarity([query_embedding],texts_embeddings)[0]

index , score =sorted(list(enumerate(score)),key=lambda x:x[1])[-1]
print(query)
print(texts[index])
print("similarity score is", score)

