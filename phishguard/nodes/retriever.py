"""Node 2 — Evidence Retriever (RAG).

Retrieval is scored against *phishing* neighbours only. A high similarity to a
legitimate-mail chunk is not evidence of phishing, so mixing both labels into one
score would let a convincing benign match inflate risk. Both are still retrieved
and shown, because a strong legitimate match is useful context for the analyst.
"""

from .. import config
from ..state import PhishGuardState, trace_entry
from ..vectorstore import collection_stats, query

NODE = "evidence_retriever"

# MiniLM cosine similarity over ordinary English prose is compressed into roughly
# 0.35-0.75, so a raw similarity of 0.55 is unremarkable and would hand every
# legitimate email a floor of risk. Two corrections, both deliberately linear and
# inspectable rather than learned:
#   absolute — rescale the useful band onto [0, 1]
#   margin   — how much closer the text sits to phishing than to legitimate prose
# The margin is what separates a genuine shipping notice from a fake one, since
# both are near the "package delivery" cluster in absolute terms.
SIM_FLOOR = 0.35
SIM_CEILING = 0.75
MARGIN_SPAN = 0.15


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _calibrate(
    max_phishing: float, max_legitimate: float, margin: float, available: bool
) -> tuple[float, str]:
    """Fold raw similarities into one inspectable 0-1 evidence score."""
    if not available:
        return 0.0, "Retrieval unavailable, so it contributes no evidence either way."

    absolute = _clamp((max_phishing - SIM_FLOOR) / (SIM_CEILING - SIM_FLOOR))
    if max_legitimate <= 0.0:
        # No legitimate neighbour in the top-k: every match is a phishing pattern.
        relative = absolute
        basis = "no legitimate pattern appeared in the top-k at all"
    else:
        relative = _clamp((margin + MARGIN_SPAN) / (2 * MARGIN_SPAN))
        basis = (
            f"nearest phishing pattern {max_phishing:.3f} vs nearest legitimate "
            f"{max_legitimate:.3f}, margin {margin:+.3f}"
        )

    evidence = round(0.5 * absolute + 0.5 * relative, 4)
    return evidence, (
        f"Retrieval evidence {evidence:.3f} = 0.5x absolute ({absolute:.3f}) "
        f"+ 0.5x relative ({relative:.3f}); {basis}."
    )


def retriever_node(state: PhishGuardState) -> dict:
    parsed = state.get("parsed", {})
    subject = parsed.get("subject", "") or ""
    body = parsed.get("body", "") or ""
    # Subject carries heavy signal for short bodies; retrieval query is derived
    # text, never instructions, so concatenating here is safe.
    query_text = f"{subject}\n{body}".strip()

    stats = collection_stats()
    chunks = query(query_text, k=config.RETRIEVAL_K)
    available = bool(chunks)

    phishing_hits = [c for c in chunks if c["label"] == "phishing"]
    legitimate_hits = [c for c in chunks if c["label"] != "phishing"]

    phishing_scores = sorted((c["similarity"] for c in phishing_hits), reverse=True)
    max_similarity = phishing_scores[0] if phishing_scores else 0.0
    top3 = phishing_scores[:3]
    mean_top3 = round(sum(top3) / len(top3), 4) if top3 else 0.0

    legitimate_scores = sorted((c["similarity"] for c in legitimate_hits), reverse=True)
    max_legitimate = legitimate_scores[0] if legitimate_scores else 0.0
    margin = round(max_similarity - max_legitimate, 4)

    evidence, evidence_note = _calibrate(max_similarity, max_legitimate, margin, available)

    if not available:
        reason = (
            "Retrieval unavailable — the Chroma index is missing or empty. "
            "Recorded as unavailable so confidence drops; it is NOT treated as "
            "'no phishing patterns matched'. Run: python scripts/ingest.py"
        )
    elif phishing_hits:
        best = phishing_hits[0]
        reason = (
            f"Nearest phishing pattern is {best['chunk_id']} ({best['pattern']}) at "
            f"cosine similarity {best['similarity']}. {evidence_note}"
        )
    else:
        reason = (
            f"All {len(chunks)} neighbours are labeled legitimate (best "
            f"{max_legitimate:.3f}); no phishing-pattern similarity recorded."
        )

    return {
        "retrieved": chunks,
        "max_similarity": max_similarity,
        "mean_top3_similarity": mean_top3,
        "max_legitimate_similarity": max_legitimate,
        "phishing_margin": margin,
        "retrieval_evidence": evidence,
        "retrieval_available": available,
        "trace": state.get("trace", [])
        + [
            trace_entry(
                node=NODE,
                inputs_seen={
                    "query_chars": len(query_text),
                    "k": config.RETRIEVAL_K,
                    "index_available": stats["available"],
                    "index_chunk_count": stats["count"],
                },
                output={
                    "retrieval_available": available,
                    "max_similarity_to_phishing": max_similarity,
                    "mean_top3_similarity_to_phishing": mean_top3,
                    "max_similarity_to_legitimate": max_legitimate,
                    "phishing_margin": margin,
                    "calibrated_retrieval_evidence": evidence,
                    "calibration": evidence_note,
                    "phishing_neighbours": len(phishing_hits),
                    "legitimate_neighbours": len(legitimate_hits),
                    "chunks": [
                        {
                            "chunk_id": c["chunk_id"],
                            "label": c["label"],
                            "pattern": c["pattern"],
                            "similarity": c["similarity"],
                            "excerpt": c["text"][:160],
                        }
                        for c in chunks
                    ],
                },
                reason=reason,
            )
        ],
    }
