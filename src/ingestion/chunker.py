import os
import glob
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

def load_all_filings(data_dir: str = "data/filings"):
    """Finds and loads ALL .md filings in data/filings/."""
    files = glob.glob(os.path.join(data_dir, "*.md"))
    if not files:
        raise FileNotFoundError("No markdown filings found in data/filings/")
    
    all_chunks = []
    for file_path in files:
        print(f"[*] Loading filing: {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        file_chunks = create_sec_chunks(content)
        all_chunks.extend(file_chunks)
        
    print(f"[+] Total chunks combined across all filings: {len(all_chunks)}")
    return all_chunks

def create_sec_chunks(markdown_text: str):
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
    
    return recursive_splitter.split_documents(header_splits)

if __name__ == "__main__":
    # Test loading and chunking
    content, filename = load_latest_filing()
    chunks = create_sec_chunks(content)
    
    # Preview the very first chunk and its metadata
    print("\n" + "="*50)
    print("SAMPLE CHUNK PREVIEW:")
    print("="*50)
    print(f"METADATA: {chunks[10].metadata}")
    print(f"TEXT CONTENT (First 300 chars):\n{chunks[10].page_content[:300]}...")
    print("="*50)