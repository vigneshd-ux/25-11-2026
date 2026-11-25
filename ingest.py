import hashlib
from pathlib import Path
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from config import collection, generate_embeddings

DOCUMENT_PATH = Path("documents/leave_policy.pdf")


def load_file(file_path):
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    documents = []

    # Check if file is a valid PDF
    is_pdf = False
    try:
        with open(file_path, "rb") as f:
            header = f.read(5)
            if header.startswith(b"%PDF"):
                is_pdf = True
    except Exception:
        pass

    if is_pdf:
        try:
            reader = PdfReader(file_path)
            for page_number, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                documents.append({
                    "text": text,
                    "source": file_path.name,
                    "page": page_number + 1
                })
            return documents
        except Exception as e:
            print(f"Warning: Failed reading {file_path} as PDF ({e}). Falling back to text mode.")

    # Fallback to reading as text (for .txt, .md, or text files saved with .pdf extension)
    text = file_path.read_text(encoding="utf-8", errors="ignore")
    documents.append({
        "text": text,
        "source": file_path.name,
        "page": 1
    })
    return documents


def chunk_documents(documents):
    langchain_docs = []
    
    for doc in documents:
        if not doc["text"].strip():
            continue
        langchain_docs.append(
            Document(
                page_content=doc["text"],
                metadata={
                    "source": doc["source"],
                    "page": doc["page"],
                    "tenant_id": "tenant_001"
                }
            )
        )
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
        separators=["\n\n", "\n", " ", ""]
    )
    
    chunks = splitter.split_documents(langchain_docs)
    return chunks


def store_chunks(chunks):
    if not chunks:
        print("No chunks to store.")
        return

    texts = [
        chunk.page_content
        for chunk in chunks
    ]

    embeddings = generate_embeddings(texts)

    ids = []
    metadatas = []

    for chunk in chunks:
        metadata = chunk.metadata
        stable_content = (
            metadata["tenant_id"]
            + metadata["source"]
            + str(metadata["page"])
            + chunk.page_content
        )

        chunk_id = hashlib.sha256(
            stable_content.encode("utf-8")
        ).hexdigest()

        ids.append(chunk_id)
        metadatas.append(metadata)

    collection.upsert(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas
    )

    print(f"{len(chunks)} chunks stored successfully in ChromaDB!")


def run_ingestion(file_path=DOCUMENT_PATH):
    print(f"Starting ingestion for: {file_path}")
    raw_docs = load_file(file_path)
    print(f"Loaded {len(raw_docs)} document section(s)/page(s).")

    chunks = chunk_documents(raw_docs)
    print(f"Generated {len(chunks)} chunk(s).")

    store_chunks(chunks)
    print(f"Total documents in collection: {collection.count()}")


if __name__ == "__main__":
    run_ingestion()

    