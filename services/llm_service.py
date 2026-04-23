def create_llm(provider: str, api_key: str):
    """Create a chat LLM for the given provider."""
    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            google_api_key=api_key,
            model="gemini-2.5-flash",
        )
    else:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            api_key=api_key,
            model="gpt-4o-mini",
        )


def build_prompt(question: str, docs: list) -> str:
    """Build the prompt, embedding page numbers alongside each context chunk."""
    context_parts = []
    for doc in docs:
        page = doc.metadata.get("page", "?")
        context_parts.append(f"[Page {page}]\n{doc.page_content}")
    context = "\n\n".join(context_parts)

    return f"""Answer the user's question using only the context below.
Where relevant, cite the page number(s) from the context (e.g. "According to page 3...").
If the answer is not in the context, say "I don't know."

Context:
{context}

Question:
{question}
"""


def get_answer(llm, question: str, docs: list) -> str:
    """Run the LLM against the retrieved docs and return the response text."""
    prompt = build_prompt(question, docs)
    response = llm.invoke(prompt)
    return response.content
