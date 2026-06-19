from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import zipfile


SAFE_ENV_CONTENT = """GOOGLE_API_KEY=
GEMINI_MODEL=gemini-2.5-flash
OPENAI_API_KEY=
OPENAI_MODEL=gpt-4o-mini
"""

EXCLUDED_DIRS = {
    ".deps",
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".submission_build",
    ".submission_validation",
    ".venv",
    ".venv312",
    ".vscode",
    "__pycache__",
    "raw_data",
    "vector_store",
}
EXCLUDED_SUFFIXES = {".pyc", ".pyo", ".log", ".zip", ".exe", ".cfg", ".cache"}
EXCLUDED_FILES = {".env", "secrets.toml"}


def should_include(path: Path, root: Path) -> bool:
    relative = path.relative_to(root)
    if any(part in EXCLUDED_DIRS for part in relative.parts):
        return False
    if path.name in EXCLUDED_FILES:
        return False
    if path.suffix in EXCLUDED_SUFFIXES:
        return False
    return True


def create_submission_zip(root: str | Path, output: str | Path) -> Path:
    project_root = Path(root).resolve()
    output_path = Path(output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    staging_path = project_root / ".submission_build"

    if staging_path.exists():
        shutil.rmtree(staging_path)
    staging_path.mkdir(parents=True)

    try:
        for path in project_root.rglob("*"):
            if staging_path in path.parents or path.resolve() == output_path:
                continue
            if not path.is_file() or not should_include(path, project_root):
                continue
            destination = staging_path / path.relative_to(project_root)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)

        (staging_path / ".env").write_bytes(SAFE_ENV_CONTENT.encode("utf-8"))

        with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in staging_path.rglob("*"):
                if path.is_file():
                    archive.write(path, path.relative_to(staging_path))
    finally:
        shutil.rmtree(staging_path, ignore_errors=True)
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create a clean submission zip")
    parser.add_argument("--root", default=".", help="Project root")
    parser.add_argument("--output", default="Taiwan_Stock_System_submission_final.zip", help="Output zip path")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    path = create_submission_zip(args.root, args.output)
    print(f"Submission zip: {path}")
    print(
        "Included: generated empty .env. Excluded: local .env, .streamlit/secrets.toml, .venv, .venv312, .deps, "
        ".vscode, __pycache__, data/raw_data, data/vector_store, .pyc, .log, .zip"
    )


if __name__ == "__main__":
    main()
