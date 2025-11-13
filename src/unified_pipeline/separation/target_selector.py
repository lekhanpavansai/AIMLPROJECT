from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from unified_pipeline.separation.embedding import SpeakerEmbedding, SpeakerEncoder
from unified_pipeline.separation.separator import SeparationResult
from unified_pipeline.core_types import AudioFrame
from unified_pipeline.utils.logging import get_logger

log = get_logger(__name__)


@dataclass(slots=True)
class TargetSelectionResult:
    target_stem: Optional[AudioFrame]
    target_index: Optional[int]
    similarities: List[float]


class TargetSelector:
    """Compare separated stems to target embedding to pick best match."""

    def __init__(self, encoder: Optional[SpeakerEncoder] = None, threshold: float = 0.6) -> None:
        self.encoder = encoder or SpeakerEncoder()
        self.threshold = threshold

    async def select(self, separation: SeparationResult, target_embedding: SpeakerEmbedding) -> TargetSelectionResult:
        if not separation.stems:
            log.warning("target_selector.no_stems")
            return TargetSelectionResult(target_stem=None, target_index=None, similarities=[])

        embeddings = []
        for idx, stem in enumerate(separation.stems):
            embedding = await self.encoder.embed(stem, speaker_id=f"stem_{idx}")
            embeddings.append(embedding)

        similarities = self.encoder.match(target_embedding, embeddings)
        best_index = max(range(len(similarities)), key=lambda i: similarities[i])
        best_score = similarities[best_index]

        log.info(
            "target_selector.similarities",
            scores=[float(f"{s:.3f}") for s in similarities],
            best_index=best_index,
            best_score=float(f"{best_score:.3f}"),
        )

        if best_score < self.threshold:
            log.warning(
                "target_selector.below_threshold",
                best_score=float(f"{best_score:.3f}"),
                threshold=float(f"{self.threshold:.3f}"),
            )
            return TargetSelectionResult(target_stem=None, target_index=None, similarities=similarities)

        return TargetSelectionResult(
            target_stem=separation.stems[best_index],
            target_index=best_index,
            similarities=similarities,
        )


