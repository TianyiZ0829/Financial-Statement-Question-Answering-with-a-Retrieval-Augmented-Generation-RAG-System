import os
import re
import logging
from bs4 import BeautifulSoup
from config import ORIGINAL_DATA_DIR, PROCESSED_DATA_DIR

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

def clean_text_content(file_content):
    """Clean HTML content by removing boilerplate, tags, and formatting."""
    try:
        soup = BeautifulSoup(file_content, "lxml")  # Try parsing as HTML
    except Exception as e:
        logging.error(f"Error parsing content as HTML: {e}")
        # If parsing fails, treat content as plain text
        return file_content.strip()

    # Remove unnecessary tags like scripts and styles
    for tag in soup(["script", "style"]):
        tag.decompose()

    # Extract plain text
    text = soup.get_text(separator="\n")
    text = re.sub(r'\n+', '\n', text).strip()
    return text

if __name__ == "__main__":
    """
    Process raw 10-K files, clean the content, and save them as text files.
    """
    for ticker in os.listdir(ORIGINAL_DATA_DIR):
        ticker_dir = os.path.join(ORIGINAL_DATA_DIR, ticker, "10-K")
        if not os.path.exists(ticker_dir):
            logging.warning(f"No 10-K filings found for ticker {ticker}. Skipping...")
            continue

        for folder in os.listdir(ticker_dir):
            folder_path = os.path.join(ticker_dir, folder)
            if not os.path.isdir(folder_path):
                continue

            # Extract year
            year_match = re.search(r"-(\d{2})-", folder)
            if not year_match:
                logging.warning(f"Year not found in folder name: {folder}. Skipping...")
                continue
            year = int(f"20{year_match.group(1)}")

            # Locate full-submission file
            submission_path = os.path.join(folder_path, "full-submission.txt")
            if not os.path.isfile(submission_path):
                logging.warning(f"File not found: {submission_path}. Skipping...")
                continue

            # Read and clean content
            try:
                with open(submission_path, "r", encoding="utf-8") as f:
                    raw_content = f.read()
                cleaned_content = clean_text_content(raw_content)
            except Exception as e:
                logging.error(f"Error reading or cleaning file {submission_path}: {e}")
                continue

            # Save processed content
            try:
                processed_dir = os.path.join(PROCESSED_DATA_DIR, f"{ticker}_{year}")
                os.makedirs(processed_dir, exist_ok=True)
                processed_file_path = os.path.join(processed_dir, "content.txt")
                with open(processed_file_path, "w", encoding="utf-8") as f:
                    f.write(cleaned_content)
                logging.info(f"Processed file saved: {processed_file_path}")
            except Exception as e:
                logging.error(f"Error saving file {processed_file_path}: {e}")