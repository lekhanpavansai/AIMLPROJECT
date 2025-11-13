from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional

from unified_pipeline.core_types import TranscriptSegment
from unified_pipeline.visualization.timeline import TimelineBuilder, TimelineConfig, TimelineEntry

try:
    import plotly.graph_objects as go
except Exception:  # pragma: no cover - optional dependency
    go = None  # type: ignore[assignment]


@dataclass(slots=True)
class VisualizationConfig:
    timeline: TimelineConfig = field(default_factory=TimelineConfig)
    enable_html: bool = True
    theme: str = "plotly_dark"


class TimelineExporter:
    """Export timeline data to JSON and optional HTML visualization."""

    def __init__(self, config: VisualizationConfig | None = None) -> None:
        self.config = config or VisualizationConfig()
        self.builder = TimelineBuilder(self.config.timeline)

    def build_entries(self, segments: Iterable[TranscriptSegment]) -> List[TimelineEntry]:
        return self.builder.build(segments)

    def export(self, segments: Iterable[TranscriptSegment], output_dir: Path) -> List[TimelineEntry]:
        entries = self.build_entries(segments)
        output_dir.mkdir(parents=True, exist_ok=True)

        json_path = output_dir / "timeline.json"
        json_path.write_text(json.dumps(entries, indent=2), encoding="utf-8")

        if self.config.enable_html:
            html_path = output_dir / "timeline.html"
            self._export_html(entries, html_path)

        return entries

    def _export_html(self, entries: List[TimelineEntry], path: Path) -> None:
        if not entries:
            path.write_text("<html><body><p>No timeline data.</p></body></html>", encoding="utf-8")
            return

        if go is None:
            path.write_text(
                "<html><body><p>Plotly not installed; timeline visualization unavailable.</p></body></html>",
                encoding="utf-8",
            )
            return

        speakers = sorted({entry["speaker"] for entry in entries})
        speaker_to_y = {speaker: idx for idx, speaker in enumerate(speakers)}

        fig = go.Figure()
        for entry in entries:
            fig.add_trace(
                go.Bar(
                    x=[entry["duration"]],
                    y=[speaker_to_y[entry["speaker"]]],
                    base=[entry["start"]],
                    orientation="h",
                    name=entry["speaker"],
                    hovertext=(
                        f"{entry['speaker']}<br>"
                        f"{entry['start']}s → {entry['end']}s<br>"
                        f"{entry['text']}<br>confidence={entry['confidence']:.2f}"
                    ),
                    marker=dict(
                        color="#1f77b4" if entry["is_target"] else "#ff7f0e",
                        opacity=0.9 if entry["is_target"] else 0.6,
                    ),
                    showlegend=False,
                )
            )

        fig.update_layout(
            title="Speaker Timeline",
            barmode="overlay",
            xaxis=dict(title="Time (s)"),
            yaxis=dict(
                tickvals=list(speaker_to_y.values()),
                ticktext=speakers,
                title="Speaker",
            ),
            template=self.config.theme,
            height=max(400, 150 + 60 * len(speakers)),
        )

        fig.write_html(str(path), include_plotlyjs="cdn", full_html=True)


