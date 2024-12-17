from sec_edgar_downloader import Downloader
import os
import shutil
import re
from config import ORIGINAL_DATA_DIR

# Load tickers from the file
with open("sampled_tickers.txt", "r") as file:
    TICKERS = [ticker.strip() for ticker in file]

# Initialize the SEC Edgar Downloader
EMAIL_ADDRESS = "icyfancyy@gmail.com"
downloader = Downloader(ORIGINAL_DATA_DIR, email_address=EMAIL_ADDRESS)

# Year range for filtering
START_YEAR = 10
END_YEAR = 20

# Define the default download directory
DEFAULT_DOWNLOAD_DIR = os.path.join(os.getcwd(), "sec-edgar-filings")

def filter_10k_by_year(ticker, start_year, end_year):
    """
    Download and filter 10-K filings for a given ticker based on the year range.
    """
    print(f"Fetching 10-K filings for: {ticker}...")
    filings_dir = os.path.join(DEFAULT_DOWNLOAD_DIR, ticker, "10-K")
    downloader.get("10-K", ticker)

    # Filter downloaded files based on the year range
    if os.path.exists(filings_dir):
        for folder in os.listdir(filings_dir):
            year_match = re.search(r"-(\d{2})-", folder)
            if year_match:
                year = int(year_match.group(1))
                if year < start_year or year >= end_year:
                    folder_path = os.path.join(filings_dir, folder)
                    shutil.rmtree(folder_path)

# Download and filter filings
for ticker in TICKERS:
    filter_10k_by_year(ticker, START_YEAR, END_YEAR)

print(f"Download and filtering complete. Filings saved to: {ORIGINAL_DATA_DIR}")

# Copy files from the default directory to the target directory
print(f"Transferring files from {DEFAULT_DOWNLOAD_DIR} to {ORIGINAL_DATA_DIR}...")
for root, _, files in os.walk(DEFAULT_DOWNLOAD_DIR):
    for file in files:
        # Create corresponding directories in the target location
        relative_path = os.path.relpath(root, DEFAULT_DOWNLOAD_DIR)
        destination_dir = os.path.join(ORIGINAL_DATA_DIR, relative_path)
        os.makedirs(destination_dir, exist_ok=True)

        # Move the file to the target directory
        shutil.copy(os.path.join(root, file), os.path.join(destination_dir, file))

print(f"All files have been successfully transferred to {ORIGINAL_DATA_DIR}.")