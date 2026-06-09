from app.retrieval.schemas import RetrievedChunk

# The single source of truth for the grounding instruction (see AGENTS.md §4.5).
GROUNDING_SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions using ONLY the provided context.\n"
    "Rules:\n"
    "- Answer strictly from the context below. Do not use outside knowledge.\n"
    "- If the answer is not in the context, say you don't have that information. Never guess.\n"
    "- Be concise, and refer to the sources you used."
)

# Returned when retrieval finds nothing relevant — Claude is NOT called in that case.
FALLBACK_ANSWER = "I don't have that information in the provided documents."


def build_user_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    """Assemble the grounded user message from retrieved chunks and the question."""
    context = "\n\n".join(
        f"[Source {index + 1}: {chunk.source}]\n{chunk.text}" for index, chunk in enumerate(chunks)
    )
    return f"Context:\n{context}\n\nQuestion: {question}"
