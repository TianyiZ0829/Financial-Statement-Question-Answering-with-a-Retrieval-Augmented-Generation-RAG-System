import os

# Define root directory
ROOT_DIR: str = '.'

# Define data directories
DATA_DIR: str = os.path.join(ROOT_DIR, 'data')
os.makedirs(DATA_DIR, exist_ok=True)

ORIGINAL_DATA_DIR: str = os.path.join(DATA_DIR, 'original')
os.makedirs(ORIGINAL_DATA_DIR, exist_ok=True)

PROCESSED_DATA_DIR: str = os.path.join(DATA_DIR, 'processed')
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

VECTOR_STORE_DIR: str = os.path.join(DATA_DIR, 'vector_store')
os.makedirs(VECTOR_STORE_DIR, exist_ok=True)

# Define model cache directory
MODEL_CACHE_DIR: str = os.path.join(ROOT_DIR, 'model_cache')
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

# Together AI Configuration
TOGETHER_API_KEY: str = "27c24aed4e7bdf1b90c062574938843a973e161244efed42705c24bc7cd1224a"
os.environ["TOGETHER_API_KEY"] = TOGETHER_API_KEY

# Model configurations
EMBED_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"  # TODO
LLM_MODEL_NAME: str = "meta-llama/Llama-3-70b-chat-hf"  # Change to Together AI's model name
