from ragsource.embeddings import Embedder
from ragsource.llm import Generator
from ragsource.models import AskResponse, Source
from ragsource.store import JsonVectorStore


class RagService:
    def __init__(
        self,
        store: JsonVectorStore,
        embedder: Embedder,
        generator: Generator,
        minimum_score: float,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.generator = generator
        self.minimum_score = minimum_score

    def ask(self, question: str, top_k: int) -> AskResponse:
        vector = self.embedder.encode([question])[0]
        matches = [
            item for item in self.store.search(vector, top_k) if item[1] >= self.minimum_score
        ]
        if not matches:
            return AskResponse(
                answer=(
                    "The indexed documents do not contain enough information "
                    "to answer this question."
                ),
                sources=[],
                grounded=False,
            )

        sources = [
            Source(
                citation=f"S{index}",
                document_id=chunk.document_id,
                filename=chunk.filename,
                page=chunk.page,
                score=round(score, 4),
                excerpt=chunk.text[:300],
            )
            for index, (chunk, score) in enumerate(matches, start=1)
        ]
        context = "\n\n".join(
            f"[S{index}] File: {chunk.filename}, page {chunk.page}\n{chunk.text}"
            for index, (chunk, _) in enumerate(matches, start=1)
        )
        return AskResponse(
            answer=self.generator.answer(question, context), sources=sources, grounded=True
        )
