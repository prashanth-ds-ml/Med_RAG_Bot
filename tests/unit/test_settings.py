from pathlib import Path

from app.settings import AppSettings


def test_settings_resolve_expected_core_paths(tmp_path: Path) -> None:
    """
    What this test checks:
    - Core repo paths are built correctly from the provided project root.

    Why this matters:
    - Every later module depends on stable, predictable path resolution.
    """
    app_settings = AppSettings(project_root=tmp_path)

    assert app_settings.data_dir == tmp_path / "data"
    assert app_settings.test_source_dir == tmp_path / "data" / "test_source"
    assert app_settings.processed_dir == tmp_path / "data" / "processed"
    assert app_settings.tracking_dir == tmp_path / "data" / "tracking"
    assert app_settings.logs_dir == tmp_path / "data" / "logs"
    assert app_settings.evals_dir == tmp_path / "data" / "evals"
    assert app_settings.configs_dir == tmp_path / "configs"


def test_settings_required_directories_include_key_repo_folders(tmp_path: Path) -> None:
    """
    What this test checks:
    - The bootstrap directory list includes the important repo folders.

    Why this matters:
    - Missing one required folder can break later pipeline steps in confusing ways.
    """
    app_settings = AppSettings(project_root=tmp_path)
    required = app_settings.required_directories()

    assert app_settings.test_source_dir in required
    assert app_settings.processed_dir in required
    assert app_settings.bm25_index_dir in required
    assert app_settings.vector_index_dir in required
    assert app_settings.logs_dir in required
    assert app_settings.tracking_dir in required
    assert app_settings.source_snapshots_dir in required


def test_settings_ensure_directories_creates_missing_folders(tmp_path: Path) -> None:
    """
    What this test checks:
    - ensure_directories actually creates the repo folders on disk.

    Why this matters:
    - Scripts should be able to bootstrap the local environment cleanly.
    """
    app_settings = AppSettings(project_root=tmp_path)

    app_settings.ensure_directories()

    for directory in app_settings.required_directories():
        assert directory.exists()
        assert directory.is_dir()


def test_settings_file_paths_are_under_expected_parent_dirs(tmp_path: Path) -> None:
    """
    What this test checks:
    - File artifact paths point into the correct parent directories.

    Why this matters:
    - Prevents logs, snapshots, and processed artifacts from being scattered.
    """
    app_settings = AppSettings(project_root=tmp_path)

    assert app_settings.source_manifest_current_path.parent == app_settings.tracking_dir
    assert app_settings.source_history_path.parent == app_settings.tracking_dir
    assert app_settings.formatting_stats_path.parent == app_settings.tracking_dir
    assert app_settings.change_events_path.parent == app_settings.tracking_dir

    assert app_settings.atomic_chunks_path.parent == app_settings.processed_dir
    assert app_settings.parent_chunks_path.parent == app_settings.processed_dir
    assert app_settings.chunk_stats_path.parent == app_settings.processed_dir

    assert app_settings.rag_runs_path.parent == app_settings.logs_dir
    assert app_settings.feedback_logs_path.parent == app_settings.logs_dir
    assert app_settings.pipeline_events_path.parent == app_settings.logs_dir


def test_settings_to_dict_contains_useful_debug_keys(tmp_path: Path) -> None:
    """
    What this test checks:
    - The debug dictionary exposes the important top-level paths.

    Why this matters:
    - Helpful when printing config summaries in CLI or manual runners.
    """
    app_settings = AppSettings(project_root=tmp_path)
    payload = app_settings.to_dict()

    assert "project_root" in payload
    assert "data_dir" in payload
    assert "test_source_dir" in payload
    assert "processed_dir" in payload
    assert "tracking_dir" in payload
    assert "logs_dir" in payload
    assert "evals_dir" in payload
    assert "configs_dir" in payload