import os
import re
import faiss
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from config import PROCESSED_DATA_DIR, VECTOR_STORE_DIR, EMBED_MODEL_NAME, MODEL_CACHE_DIR

# Initialize FAISS index
def initialize_faiss_index(dimension: int) -> faiss.IndexFlatL2:
    """Initialize a FAISS index for vector storage."""
    index = faiss.IndexFlatL2(dimension)  # L2 distance for similarity search
    return index

# Define text chunking strategy
def split_text_into_chunks(text: str, max_chunk_size: int = 512) -> list[str]:
    """Split text into smaller chunks based on chunk size."""
    sentences = re.split(r'(?<=[.!?])\s+', text)  # Split text into sentences
    chunks, current_chunk = [], ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= max_chunk_size:
            current_chunk += sentence + " "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + " "
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

# Hugging Face model setup
def load_embedding_model(model_name: str, cache_dir: str, device: str = "cpu"):
    """Load tokenizer and model from Hugging Face."""
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir).to(device)
    return tokenizer, model

def generate_embedding(text: str, tokenizer, model, device: str = "cpu") -> np.ndarray:
    """Generate embedding for a given text."""
    tokens = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    outputs = model(**tokens)
    embedding = outputs.last_hidden_state.mean(dim=1).detach().cpu().numpy().flatten()  # Ensure embeddings are on CPU
    return embedding

# Process documents and store embeddings
def process_documents_to_vector_store(processed_data_dir, vector_store_dir, model_name, cache_dir, device: str = "cpu"):
    """Process cleaned files, generate embeddings, and store in FAISS."""
    os.makedirs(vector_store_dir, exist_ok=True)

    # Load Hugging Face model with GPU/CPU device
    tokenizer, model = load_embedding_model(model_name, cache_dir, device)

    # Initialize FAISS index
    dimension = 384  # MiniLM embedding dimension
    faiss_index = initialize_faiss_index(dimension)

    metadata = []  # To store metadata for each embedding

    # Process each file
    for ticker_year in os.listdir(processed_data_dir):
        ticker, year = ticker_year.split("_")
        file_path = os.path.join(processed_data_dir, ticker_year, "content.txt")
        if not os.path.isfile(file_path):
            continue

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Split text into chunks
        chunks = split_text_into_chunks(content)

        # Generate embeddings for each chunk
        for chunk_id, chunk in enumerate(chunks):
            embedding = generate_embedding(chunk, tokenizer, model, device)
            faiss_index.add(np.array([embedding]))
            metadata.append({"ticker": ticker, "year": int(year), "chunk_id": chunk_id, "content": chunk})

    # Save FAISS index
    faiss.write_index(faiss_index, os.path.join(vector_store_dir, "vector_store.index"))

    # Save metadata
    metadata_file = os.path.join(vector_store_dir, "metadata.npy")
    np.save(metadata_file, metadata)

    print(f"FAISS index and metadata stored in {vector_store_dir}")

if __name__ == "__main__":
    # Determine if GPU is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Use constants from config.py
    process_documents_to_vector_store(
        processed_data_dir=PROCESSED_DATA_DIR,
        vector_store_dir=VECTOR_STORE_DIR,
        model_name=EMBED_MODEL_NAME,
        cache_dir=MODEL_CACHE_DIR,
        device=device
    )
