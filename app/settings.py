from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class AppSettings:
    """
    Repository-wide path settings.

    Why a dataclass:
    - Makes the settings explicit and easy to inspect
    - Gives clean defaults while still allowing overrides in tests and CLI
    """

    project_root: Path | str = field(
        default_factory=lambda: Path(__file__).resolve().parents[1]
    )

    def __post_init__(self) -> None:
        """
        Normalize the project root to an absolute resolved Path.

        Why this matters:
        - CLI arguments often arrive as strings
        - Tests may pass either strings or Path objects
        - Converting here makes the rest of the code simpler and safer
        """
        self.project_root = Path(self.project_root).resolve()

    @property
    def app_dir(self) -> Path:
        return self.project_root / "app"

    @property
    def tests_dir(self) -> Path:
        return self.project_root / "tests"

    @property
    def configs_dir(self) -> Path:
        return self.project_root / "configs"

    @property
    def scripts_dir(self) -> Path:
        return self.project_root / "scripts"

    @property
    def data_dir(self) -> Path:
        return self.project_root / "data"

    @property
    def test_source_dir(self) -> Path:
        return self.data_dir / "test_source"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def indexes_dir(self) -> Path:
        return self.data_dir / "indexes"

    @property
    def bm25_index_dir(self) -> Path:
        return self.indexes_dir / "bm25"

    @property
    def vector_index_dir(self) -> Path:
        return self.indexes_dir / "vector"

    @property
    def evals_dir(self) -> Path:
        return self.data_dir / "evals"

    @property
    def logs_dir(self) -> Path:
        return self.data_dir / "logs"

    @property
    def exports_dir(self) -> Path:
        return self.data_dir / "exports"

    @property
    def tracking_dir(self) -> Path:
        return self.data_dir / "tracking"

    @property
    def source_snapshots_dir(self) -> Path:
        return self.tracking_dir / "source_snapshots"

    @property
    def source_manifest_current_path(self) -> Path:
        return self.tracking_dir / "source_manifest_current.json"

    @property
    def source_history_path(self) -> Path:
        return self.tracking_dir / "source_history.jsonl"

    @property
    def formatting_stats_path(self) -> Path:
        return self.tracking_dir / "formatting_stats.jsonl"

    @property
    def change_events_path(self) -> Path:
        return self.tracking_dir / "change_events.jsonl"

    @property
    def ingest_lineage_path(self) -> Path:
        return self.tracking_dir / "ingest_lineage.jsonl"

    @property
    def manual_edit_notes_path(self) -> Path:
        return self.tracking_dir / "manual_edit_notes.jsonl"

    @property
    def scan_manifest_path(self) -> Path:
        return self.processed_dir / "scan_manifest.json"

    @property
    def atomic_chunks_path(self) -> Path:
        return self.processed_dir / "atomic_chunks.jsonl"

    @property
    def parent_chunks_path(self) -> Path:
        return self.processed_dir / "parent_chunks.jsonl"

    @property
    def chunk_stats_path(self) -> Path:
        return self.processed_dir / "chunk_stats.json"

    @property
    def rag_runs_path(self) -> Path:
        return self.logs_dir / "rag_runs.jsonl"

    @property
    def feedback_logs_path(self) -> Path:
        return self.logs_dir / "feedback_logs.jsonl"

    @property
    def pipeline_events_path(self) -> Path:
        return self.logs_dir / "pipeline_events.jsonl"

    @property
    def gold_queries_path(self) -> Path:
        return self.evals_dir / "gold_queries.jsonl"

    @property
    def eval_runs_path(self) -> Path:
        return self.evals_dir / "eval_runs.jsonl"
    
    @property
    def bm25_index_path(self) -> Path:
        return self.bm25_index_dir / "bm25_index.pkl"

    @property
    def failure_cases_path(self) -> Path:
        return self.evals_dir / "failure_cases.jsonl"

    @property
    def pipeline_config_path(self) -> Path:
        return self.configs_dir / "pipeline.yaml"

    @property
    def chunking_config_path(self) -> Path:
        return self.configs_dir / "chunking.yaml"

    @property
    def retrieval_config_path(self) -> Path:
        return self.configs_dir / "retrieval.yaml"

    @property
    def logging_config_path(self) -> Path:
        return self.configs_dir / "logging.yaml"

    def required_directories(self) -> list[Path]:
        return [
            self.configs_dir,
            self.scripts_dir,
            self.data_dir,
            self.test_source_dir,
            self.processed_dir,
            self.indexes_dir,
            self.bm25_index_dir,
            self.vector_index_dir,
            self.evals_dir,
            self.logs_dir,
            self.exports_dir,
            self.tracking_dir,
            self.source_snapshots_dir,
        ]

    def ensure_directories(self) -> None:
        for directory in self.required_directories():
            directory.mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> dict[str, str]:
        return {
            "project_root": str(self.project_root),
            "data_dir": str(self.data_dir),
            "test_source_dir": str(self.test_source_dir),
            "processed_dir": str(self.processed_dir),
            "tracking_dir": str(self.tracking_dir),
            "logs_dir": str(self.logs_dir),
            "evals_dir": str(self.evals_dir),
            "configs_dir": str(self.configs_dir),
        }


settings = AppSettings()