"""OpenAI client for RAG answer generation."""

from typing import Any


def generate_answer(
    query: str,
    context: str,
    api_key: str | None = None,
) -> str:
    """
    Generate an answer using OpenAI given query and retrieved context.

    If no API key: returns placeholder message.
    """
    if not api_key:
        return "Enable OpenAI API key in settings to generate answers."

    try:
        from openai import OpenAI
    except ImportError:
        return "Install openai: pip install openai"

    client = OpenAI(api_key=api_key)
    prompt = f"""Context:
{context}

Question: {query}

Answer based only on the context above.
- If the context does not contain relevant information, say so.
- If the question is not related to the context, say so.
- If the question is related to the context, answer using the context; you may draw conclusions from it.
- If you cannot answer from the context, state that you are unable to answer rather than guessing. """

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Answer based only on the provided context."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=500,
        )
        return response.choices[0].message.content or "No response generated."
    except Exception as e:
        return f"Error: {str(e)}"
