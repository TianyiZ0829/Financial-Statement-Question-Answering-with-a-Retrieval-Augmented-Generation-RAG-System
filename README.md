# Financial Statement Question-Answering with a Retrieval-Augmented Generation (RAG) System

This project involves building a Retrieval-Augmented Generation (RAG) system to enable financial statement question-answering. Students will develop, test, and evaluate various components of the system, working with real-world financial data. The project culminates in the creation of a Streamlit application for querying financial statements.

## Files to Modify

Below is a list of files requiring modification, with a brief description of each:

1. **`sample_tickers.py`**
   - Task: Sample 10 company tickers based on your unique student ID.
   - This script fetches the current list of S&P 500 companies from Wikipedia, ensures reproducibility by using a student ID as the seed, and selects 10 random tickers. The output is saved as `sampled_tickers.txt`. The sampled tickers will be used to download financial reports in subsequent steps.

2. **`data_downloading.py`**
   - Task: Download 10-K filings for three selected companies (tickers).
   - Store the filings in the `ORIGINAL_DATA_DIR` as defined in `config.py`.

3. **`data_processing.py`**
   - Task: Process the downloaded 10-K filings, extract relevant text, and clean it (e.g., HTML tags, boilerplate removal).
   - Save the cleaned data in the specified format under `PROCESSED_DATA_DIR`.

4. **`vector_store_construction.py`**
   - Task: Chunk processed documents and store them in a vector database.
   - Use metadata to organize chunks by ticker and year, and implement a vector store index.

5. **`system.py`**
   - Task: Build the RAG pipeline, including embedding models, retrievers, metadata filtering, reranking, and response generation.
   - Ensure the pipeline handles queries effectively and integrates the various system components.
   - Note: Using a reranker might exceed the GPU constraints of Colab. If this occurs, you may provide commented-out code for the reranker and will not be deducted points.
   - Evaluation: 
     - Create 5 questions for each ticker based on the processed data.
     - Evaluate the system manually, focusing on both the retriever's ability to identify the most relevant node and the quality of the generated response.
     - Provide a detailed analysis of the evaluation, even though implementing automatic evaluation techniques is not required.

6. **`app.py`**
   - Task: Develop a Streamlit-based web app to enable users to interact with the RAG system.
   - Implement a user-friendly interface for selecting company tickers, years, and submitting queries.

7. **`config.py`**
   - Task: Configure the directory structure and specify the embedding and language model names (`EMBED_MODEL_NAME` and `LLM_MODEL_NAME`).
   - You will need to change the model names in this file to match the models you intend to use for the project.

**Note**: Detailed grading rubrics are provided in each file.

## Running on Colab

To run the system on Google Colab:

1. **Upload Files**: 
   - Run a cell in your Colab notebook to activate the directory tree.
   - Upload all project files (`.py`, `.sh`, etc.) to the `/content/` directory.

2. **Install Dependencies**:
   - Run the following command from run.sh to install necessary packages in a Colab cell:
     ```bash
     !pip install -r requirements.txt
     ```
   - If TensorFlow conflicts occur, use the additional commented-out two lines in `run.sh`.

3. **Execution**:
   - Once the code is implemented, ensure the system (excluding the Streamlit app) is error-free by running:
     ```bash
     source run.sh
     ```

## Notes on Hardware and Execution

- **GPU Requirement**:
  - The full system, including the app, requires a GPU to run efficiently. If you do not have GPU access, you will not be able to execute the entire system locally.
  
- **Colab Restrictions**:
  - It is impossible to open the Streamlit app on Colab. Therefore, evaluation and testing of the app must be done locally, while system functionality can be verified on Colab.
