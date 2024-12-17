import random
import requests
import pandas as pd
import os
current_dir = os.getcwd()

STUDENT_ID: int = 903880510
NUM_DOCS: int = 10  # Number of companies to sample
REQUIRED_YEARS = list(range(2010, 2020))  # Required filing years (2010-2019)
USER_AGENT = f"StudentID-{STUDENT_ID} (icyfancyy@gmail.com)"
SEC_API_BASE_URL = "https://data.sec.gov/submissions/CIK{}.json"  # SEC API URL


def get_sp500_tickers_wikipedia() -> list[str]:
    """
    Fetch the S&P 500 company tickers from Wikipedia.
    Returns:
        list[str]: A list of S&P 500 tickers.
    """
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = pd.read_html(url)
    sp500_table = tables[0]
    return sp500_table["Symbol"].tolist()


def get_cik_mapping() -> dict:
    """
    Fetch all tickers and their corresponding CIKs from SEC JSON.
    Returns:
        dict: A dictionary mapping tickers to CIKs ({ticker: CIK}).
    """
    headers = {"User-Agent": USER_AGENT}
    url = "https://www.sec.gov/files/company_tickers_exchange.json"
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch SEC company list: HTTP {response.status_code}")

    companies = response.json()
    fields = companies.get("fields", [])
    data = companies.get("data", [])

    # Determine field indices for CIK and ticker
    try:
        cik_index = fields.index("cik")
        ticker_index = fields.index("ticker")
    except ValueError as e:
        raise Exception("Could not find CIK or Ticker field indices.") from e

    # Construct the ticker-to-CIK mapping
    ticker_to_cik = {
        row[ticker_index].upper(): str(row[cik_index]).zfill(10) for row in data
    }
    return ticker_to_cik


def has_required_filings(cik: str, required_years: list[int]) -> bool:
    """
    Check if a company has all required 10-K filings for the given years.
    Args:
        cik (str): The company's CIK.
        required_years (list[int]): List of required filing years.
    Returns:
        bool: True if all required years are covered with 10-K filings, False otherwise.
    """
    url = SEC_API_BASE_URL.format(cik)
    headers = {"User-Agent": USER_AGENT}
    response = requests.get(url, headers=headers)

    if response.status_code != 200:
        print(f"Failed to fetch filings for company {cik}: HTTP {response.status_code}")
        return False

    data = response.json()
    filings = data.get("filings", {}).get("recent", {})
    filing_dates = filings.get("filingDate", [])
    filing_types = filings.get("form", [])

    # Extract years of all 10-K filings
    filing_years = {
        int(date[:4]) for date, form in zip(filing_dates, filing_types) if form == "10-K"
    }

    # Check if all required years are present
    return all(year in filing_years for year in required_years)


def sample_valid_tickers(tickers: list[str], ticker_to_cik: dict, num_docs: int, required_years: list[int]) -> list[str]:
    """
    Randomly sample valid tickers that meet the required filing criteria.
    Args:
        tickers (list): List of S&P 500 tickers.
        ticker_to_cik (dict): A dictionary mapping tickers to CIKs.
        num_docs (int): Number of companies to sample.
        required_years (list[int]): List of required filing years.
    Returns:
        list: A list of valid tickers.
    """
    random.seed(STUDENT_ID)
    sampled_tickers = random.sample(tickers, len(tickers))  # Shuffle tickers randomly
    valid_tickers = []

    for ticker in sampled_tickers:
        print(f"Validating ticker: {ticker}")
        cik = ticker_to_cik.get(ticker)
        if not cik:
            print(f"CIK not found for ticker {ticker}. Skipping...")
            continue

        if has_required_filings(cik, required_years):
            valid_tickers.append(ticker)
            print(f"Ticker {ticker} meets the requirements.")
            if len(valid_tickers) == num_docs:
                break
        else:
            print(f"Ticker {ticker} does not meet the requirements.")

    return valid_tickers


if __name__ == "__main__":
    # Step 1: Fetch S&P 500 tickers from Wikipedia
    sp500_tickers = get_sp500_tickers_wikipedia()

    # Step 2: Fetch ticker-to-CIK mapping
    ticker_to_cik_map = get_cik_mapping()

    # Step 3: Sample valid tickers
    valid_tickers = sample_valid_tickers(sp500_tickers, ticker_to_cik_map, NUM_DOCS, REQUIRED_YEARS)

    # Step 4: Save results to a file
    with open("sampled_tickers.txt", "w") as f:
        f.write("\n".join(valid_tickers))

    print(f"Final sampled tickers: {valid_tickers}")