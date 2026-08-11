import os
from edgar import set_identity, Company

# SEC mandates a User-Agent identity (Name and Email) to access public APIs
set_identity("Avala Jaithra avalajaithraprakarsh@gmail.com")

def fetch_latest_10k(ticker: str, save_dir: str = "data/filings"):
    """
    Connects to SEC EDGAR, retrieves the latest Form 10-K for a ticker,
    and extracts clean Markdown text.
    """
    os.makedirs(save_dir, exist_ok=True)
    print(f"[*] Connecting to SEC EDGAR for Ticker: {ticker}...")
    
    # Look up company
    company = Company(ticker)
    print(f"[*] Found Company: {company.name} (CIK: {company.cik})")
    
    # Fetch latest 10-K filing
    filing = company.get_filings(form="10-K").latest()
    print(f"[*] Latest 10-K Filed On: {filing.filing_date}")
    
    # Extract clean markdown text from the filing
    print("[*] Extracting Markdown text (this may take 15-30 seconds)...")
    clean_text = filing.markdown()
    
    # Save to disk
    file_path = os.path.join(save_dir, f"{ticker}_10K_{filing.filing_date}.md")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(clean_text)
        
    print(f"[+] Saved clean 10-K report to: {file_path}")
    return file_path

if __name__ == "__main__":
    fetch_latest_10k("MSFT")