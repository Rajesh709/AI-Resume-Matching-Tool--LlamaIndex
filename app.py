from llama_index import SimpleDirectoryReader, VectorStoreIndex
from llama_index.embeddings import HuggingFaceEmbedding
from llama_index.llms import OpenAI
import os

# Load resumes from a folder
documents = SimpleDirectoryReader(input_dir="resumes", required_exts=[".pdf", ".txt"]).load_data()

# Create index with custom embeddings (you can also use OpenAIEmbedding)
embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)

# Load job description
with open("job_description.txt", "r") as f:
    jd_text = f.read()

# Query the index with job description
query_engine = index.as_query_engine(similarity_top_k=5)
response = query_engine.query(jd_text)

# Show matched resumes
print("📌 Top Matching Resumes:\n")
print(response)
