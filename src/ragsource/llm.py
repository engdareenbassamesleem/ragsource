from typing import Protocol


class Generator(Protocol):
    def answer(self, question: str, context: str) -> str: ...


SYSTEM_INSTRUCTION = """You are a citation-first knowledge assistant.
Answer only from the supplied context. Cite supporting statements with the exact markers [S1],
[S2], and so on. If the context does not support an answer, say that the indexed documents do
not contain enough information. Do not use outside knowledge and do not invent citations."""


class GeminiGenerator:
    def __init__(self, api_key: str, model: str) -> None:
        from google import genai

        self.client = genai.Client(api_key=api_key)
        self.model = model

    def answer(self, question: str, context: str) -> str:
        prompt = f"{SYSTEM_INSTRUCTION}\n\nCONTEXT:\n{context}\n\nQUESTION:\n{question}"
        response = self.client.models.generate_content(model=self.model, contents=prompt)
        return response.text or "The indexed documents do not contain enough information."


class ExtractiveGenerator:
    """Safe fallback that returns retrieved evidence without calling an external LLM."""

    def answer(self, question: str, context: str) -> str:
        del question
        return f"Relevant evidence from the indexed documents:\n\n{context}"
