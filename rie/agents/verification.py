from rie.llm.granite import GraniteClient
from rie.retrieval.vector_store import FaissVectorStore
from rie.state import ResearchState, VerifiedClaim


class VerificationAgent:
    def __init__(self, llm: GraniteClient | None = None, vector_store: FaissVectorStore | None = None):
        self.llm = llm or GraniteClient()
        self.vector_store = vector_store

    def run(self, state: ResearchState) -> ResearchState:
        verified = []
        for evidence in state.evidence:
            if self.vector_store is not None and self.vector_store.index is not None:
                matches = self.vector_store.search(evidence.claim, k=3)
                chunks = "\n---\n".join(chunk.text[:300] for chunk, _ in matches)
                ids = [chunk.chunk_id for chunk, _ in matches]
            else:
                chunks = evidence.chunk.text[:300]
                ids = [evidence.chunk.chunk_id]

            response = self.llm.generate(
                f"Does this claim have direct support in the evidence? Reply SUPPORTED or UNSUPPORTED.\n"
                f"Claim: {evidence.claim}\nEvidence: {chunks}",
                200,
            )
            supported = response.strip().upper().startswith("SUPPORTED")
            verified.append(VerifiedClaim(
                claim=evidence.claim,
                supported=supported,
                supporting_chunk_ids=ids if supported else [],
                flag="" if supported else f"UNSUPPORTED: {response.strip()[:120]}",
            ))
        state.verified_claims = verified
        return state
