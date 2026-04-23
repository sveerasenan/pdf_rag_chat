from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_and_split_pdf(pdf_path: str) -> list:
    """Load a PDF and split it into chunks, preserving page metadata."""
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
    )
    chunks = splitter.split_documents(documents)

    # Normalize page numbers to 1-indexed for display
    for chunk in chunks:
        raw_page = chunk.metadata.get("page", 0)
        chunk.metadata["page"] = int(raw_page) + 1

    return chunks
