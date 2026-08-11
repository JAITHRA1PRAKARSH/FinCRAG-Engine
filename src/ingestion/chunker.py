import os
import glob
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

def identify_company(filename: str) -> tuple[str, str]:
    """Identifies the company name and ticker from the filename."""
    base = os.path.basename(filename).upper()
    if "AAPL" in base or "APPLE" in base:
        return "Apple Inc.", "AAPL"
    elif "MSFT" in base or "MICROSOFT" in base:
        return "Microsoft Corporation", "MSFT"
    else:
        name = os.path.splitext(os.path.basename(filename))[0]
        return name, name

def create_sec_chunks(markdown_text: str, file_path: str = ""):
    """Splits markdown text and injects company/document metadata into every chunk."""
    company_name, ticker = identify_company(file_path)
    file_name = os.path.basename(file_path)
    
    headers_to_split_on = [
        ("#", "Header_1"),
        ("##", "Header_2"),
        ("###", "Header_3"),
    ]
    
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False
    )
    header_splits = markdown_splitter.split_text(markdown_text)

    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=200,
        separators=["\n\n|", "\n\n", "\n", ".", " ", ""]
    )
    
    raw_chunks = recursive_splitter.split_documents(header_splits)
    
    # Inject Contextual Metadata Header into EVERY Chunk
    enhanced_chunks = []
    for chunk in raw_chunks:
        # Contextual prefix ensures BM25 & Vector search always match the right company
        prefix = f"[Company: {company_name} ({ticker}) | Filing: SEC Form 10-K | Source: {file_name}]\n"
        enhanced_content = prefix + chunk.page_content
        
        metadata = dict(chunk.metadata)
        metadata.update({
            "company": company_name,
            "ticker": ticker,
            "source_file": file_name
        })
        
        enhanced_chunks.append(Document(page_content=enhanced_content, metadata=metadata))
        
    return enhanced_chunks

def load_all_filings(data_dir: str = "data/filings"):
    """Finds and loads ALL .md filings in data/filings/ with company attribution."""
    files = glob.glob(os.path.join(data_dir, "*.md"))
    if not files:
        raise FileNotFoundError(f"No markdown filings found in '{data_dir}'")
    
    all_chunks = []
    for file_path in files:
        print(f"[*] Loading and tagging filing: {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        file_chunks = create_sec_chunks(content, file_path=file_path)
        all_chunks.extend(file_chunks)
        
    print(f"[+] Total Context-Enriched Chunks: {len(all_chunks)}")
    return all_chunks