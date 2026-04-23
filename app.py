import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from services.pdf_service import load_and_split_pdf
from services.embedding_service import create_embeddings, create_vectorstore, create_retriever
from services.llm_service import create_llm, get_answer

st.set_page_config(page_title="Chat with Your PDF")
st.title("Chat with Your PDF")

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower()

if LLM_PROVIDER == "gemini":
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        st.error("GOOGLE_API_KEY not found. Please set it in the .env file.")
        st.stop()
else:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("OPENAI_API_KEY not found. Please set it in the .env file.")
        st.stop()

st.caption(f"Using provider: **{LLM_PROVIDER}**")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    # Only re-process when a different file is uploaded
    if st.session_state.get("uploaded_file_name") != uploaded_file.name:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            temp_pdf_path = tmp_file.name

        try:
            with st.spinner("Processing PDF..."):
                chunks = load_and_split_pdf(temp_pdf_path)
                embeddings = create_embeddings(LLM_PROVIDER, api_key)
                vectorstore = create_vectorstore(chunks, embeddings)
                st.session_state.retriever = create_retriever(vectorstore)
                st.session_state.llm = create_llm(LLM_PROVIDER, api_key)
                st.session_state.uploaded_file_name = uploaded_file.name
        except Exception as e:
            st.error(f"Error processing PDF: {e}")
            st.stop()
        finally:
            if os.path.exists(temp_pdf_path):
                os.remove(temp_pdf_path)

    question = st.text_input("Ask a question about the PDF")

    if question:
        try:
            docs = st.session_state.retriever.invoke(question)
            answer = get_answer(st.session_state.llm, question, docs)

            st.subheader("Answer")
            st.write(answer)

            st.subheader("Retrieved Source Chunks")
            for i, doc in enumerate(docs, start=1):
                page = doc.metadata.get("page", "N/A")
                source = doc.metadata.get("source", "")
                with st.expander(f"Chunk {i} — Page {page}" + (f"  |  {os.path.basename(source)}" if source else "")):
                    st.write(doc.page_content)
        except Exception as e:
            st.error(f"Error generating answer: {e}")