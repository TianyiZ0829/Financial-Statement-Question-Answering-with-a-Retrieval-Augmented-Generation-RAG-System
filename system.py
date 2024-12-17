import os
import faiss
import numpy as np
import torch
import json
from transformers import AutoTokenizer, AutoModel
from config import VECTOR_STORE_DIR, EMBED_MODEL_NAME, LLM_MODEL_NAME, TOGETHER_API_KEY
from together import Together
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    filename="rag_system_log.log",
    level=logging.INFO,
    format="%(asctime)s - %(message)s"
)

# Initialize Together AI client
client = Together(api_key=TOGETHER_API_KEY)


# HuggingFace Embedding Function
class HuggingFaceEmbeddingFunction:
    def __init__(self, model_name: str = EMBED_MODEL_NAME, device: str = "cpu"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.device = device

    def embed_text(self, text: str) -> np.ndarray:
        """Generate embeddings for a given text."""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(self.device)
        outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).detach().cpu().numpy().flatten()


# RAG System
class RAGSystem:
    def __init__(self, device: str = "cpu"):
        self.device = device
        self.embedding_function = HuggingFaceEmbeddingFunction(device=device)
        self.faiss_index = self.load_faiss_index()
        self.metadata = self.load_metadata()

    def load_faiss_index(self):
        """Load FAISS index."""
        index_path = os.path.join(VECTOR_STORE_DIR, "vector_store.index")
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"FAISS index not found at {index_path}")
        return faiss.read_index(index_path)

    def load_metadata(self):
        """Load metadata for embeddings."""
        metadata_path = os.path.join(VECTOR_STORE_DIR, "metadata.npy")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata not found at {metadata_path}")
        return np.load(metadata_path, allow_pickle=True).tolist()

    def retrieve(self, query: str, year: int = None, ticker: str = None) -> list:
        """Retrieve top-k relevant documents with metadata filtering."""
        query_embedding = self.embedding_function.embed_text(query).reshape(1, -1)
        distances, indices = self.faiss_index.search(query_embedding, RETRIEVER_SIMILARITY_TOP_K)
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx == -1:
                continue
            doc_metadata = self.metadata[idx]
            if (year and doc_metadata["year"] != year) or (ticker and doc_metadata["ticker"] != ticker):
                continue  # Filter by year or ticker
            doc_metadata["distance"] = dist
            results.append(doc_metadata)
        return results

    def rerank(self, retrieved_docs: list, query: str) -> list:
        """Rerank retrieved documents based on cosine similarity."""
        query_embedding = self.embedding_function.embed_text(query)
        for doc in retrieved_docs:
            doc_embedding = self.embedding_function.embed_text(doc["content"])
            doc["cosine_similarity"] = np.dot(query_embedding, doc_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
            )
        return sorted(retrieved_docs, key=lambda x: x["cosine_similarity"], reverse=True)

    def generate_response(self, query: str, retrieved_docs: list) -> str:
        """Generate response using Together AI."""
        context = "\n".join([doc["content"] for doc in retrieved_docs])
        prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
        response = client.chat.completions.create(
            model=LLM_MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.7
        )
        response_text = response.choices[0].message.content.strip()
        return response_text

    def query(self, query: str, year: int = None, ticker: str = None) -> str:
        """Main RAG query pipeline with metadata filtering and reranking."""
        print(f"Processing query: {query}")
        retrieved_docs = self.retrieve(query, year, ticker)
        if not retrieved_docs:
            return "No relevant documents found."
        reranked_docs = self.rerank(retrieved_docs, query)
        return self.generate_response(query, reranked_docs)


# Evaluation
TICKERS = ['FRT', 'CCL', 'IPG', 'WRB', 'IEX', 'TECH', 'PHM', 'LVS', 'ROP', 'ULTA']
QUERY_SET = {
    "FRT": ["What does FRT say about its real estate business?", "What risks are highlighted in FRT's filings?"],
    "CCL": ["What operational updates does CCL provide?", "What challenges does CCL highlight in recent filings?"],
    "IPG": ["What business activities does IPG focus on?", "What risks does IPG mention in its filings?"],
    "WRB": ["What does WRB say about its insurance offerings?", "What are the main risks WRB mentions?"],
    "IEX": ["What industries does IEX primarily serve?", "What challenges does IEX report in its filings?"],
    "TECH": ["What does TECH say about its research and development?", "What risks does TECH highlight in its filings?"],
    "PHM": ["What does PHM report about the housing market?", "What risks does PHM mention in recent filings?"],
    "LVS": ["What does LVS say about its casino operations?", "What are the key risks LVS highlights?"],
    "ROP": ["What is ROP's primary business focus?", "What risks does ROP highlight in its filings?"],
    "ULTA": ["What does ULTA say about its retail business?", "What challenges does ULTA mention in its filings?"],
}


def evaluate_rag_system():
    """Evaluate the RAG system with multiple queries for each ticker."""
    # Determine device for model computation
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Initialize the RAG system
    rag_system = RAGSystem(device=device)

    # Initialize results container
    evaluation_results = []

    # Iterate through each ticker and its queries
    for ticker in TICKERS:
        logging.info(f"Evaluating Ticker: {ticker}")
        ticker_results = {"ticker": ticker, "queries": []}
        queries = QUERY_SET.get(ticker, [])
        for query in queries:
            try:
                logging.info(f"Processing Query: {query}")
                response = rag_system.query(query, ticker=ticker)
                retrieved_docs = rag_system.retrieve(query, ticker=ticker)

                # Convert retrieved documents to JSON-serializable format
                retrieved_docs_serializable = []
                for doc in retrieved_docs:
                    doc_serializable = {k: (v.item() if isinstance(v, np.generic) else v) for k, v in doc.items()}
                    retrieved_docs_serializable.append(doc_serializable)

                # Save query results
                query_result = {
                    "query": query,
                    "response": response,
                    "retrieved_docs": retrieved_docs_serializable,
                }
                ticker_results["queries"].append(query_result)

                # Debug: Print the response
                print(f"Query: {query}\nResponse: {response}\n")

            except Exception as e:
                logging.error(f"Error processing query '{query}' for ticker '{ticker}': {e}")
                print(f"Error processing query '{query}' for ticker '{ticker}': {e}")

        evaluation_results.append(ticker_results)

    # Save results to JSON file
    try:
        with open("evaluation_results.json", "w") as f:
            json.dump(evaluation_results, f, indent=4)
        print("Evaluation results saved to 'evaluation_results.json'.")
    except Exception as e:
        print(f"Error saving evaluation results: {e}")



def generate_summary_report():
    """Generate a human-readable summary report."""
    try:
        # Load evaluation results
        with open("evaluation_results.json", "r") as f:
            evaluation_results = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading evaluation results: {e}")
        return

    # Generate a summary report
    with open("summary_report.txt", "w") as report:
        report.write("RAG System Evaluation Summary Report\n")
        report.write("=" * 60 + "\n\n")
        for ticker_results in evaluation_results:
            report.write(f"Ticker: {ticker_results['ticker']}\n")
            report.write("-" * 60 + "\n")
            for query in ticker_results["queries"]:
                report.write(f"Query: {query['query']}\n")
                report.write(f"Response: {query['response']}\n")
                for doc in query["retrieved_docs"]:
                    report.write(f"  - Document Preview: {doc['content'][:200]}...\n")
                report.write("\n")
            report.write("=" * 60 + "\n\n")
    print("Summary report saved to 'summary_report.txt'.")



# Run evaluation and report
if __name__ == "__main__":
    evaluate_rag_system()
    generate_summary_report()
    print("Evaluation and report generation completed.")

