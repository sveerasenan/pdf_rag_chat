from langchain_community.vectorstores import FAISS


def create_embeddings(provider: str, api_key: str):
    """Create an embeddings model for the given provider."""
    if provider == "gemini":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        return GoogleGenerativeAIEmbeddings(
            google_api_key=api_key,
            model="models/gemini-embedding-001",
        )
    else:
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(
            api_key=api_key,
            model="text-embedding-3-small",
        )


def create_vectorstore(chunks: list, embeddings) -> FAISS:
    """Build a FAISS vectorstore from document chunks."""
    return FAISS.from_documents(chunks, embeddings)


def create_retriever(vectorstore: FAISS, k: int = 3):
    """Create a retriever that returns the top-k most relevant chunks."""
    return vectorstore.as_retriever(search_kwargs={"k": k})
