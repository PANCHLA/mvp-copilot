# ingest/ingest.py
import os
import json
from pathlib import Path
from git import Repo, InvalidGitRepositoryError
from tqdm import tqdm

# --- make all paths absolute and project-root relative ---
SCRIPT_DIR = Path(__file__).resolve().parent       # ingest/
PROJECT_ROOT = SCRIPT_DIR.parent                   # mvp-copilot/
OUT_DIR = (PROJECT_ROOT / "data" / "chunks").resolve()
RAW_REPO_DIR = (PROJECT_ROOT / "data" / "raw_repos").resolve()
CHUNK_SIZE = 200
OVERLAP = 40

# safety thresholds
MAX_FILE_SIZE = 1_000_000  # bytes; skip files larger than 1MB by default
BINARY_CHECK_BYTES = 4096  # bytes to sample when checking for binary
BINARY_NON_TEXT_THRESHOLD = 0.30  # if >30% of sampled bytes are non-text, treat as binary

OUT_DIR.mkdir(parents=True, exist_ok=True)
RAW_REPO_DIR.mkdir(parents=True, exist_ok=True)


def clone_repo(git_url: str, target_name: str):
    target_path = (RAW_REPO_DIR / target_name).resolve()
    if target_path.exists():
        print(f"[+] Repo already cloned at {target_path}")
        return target_path
    print(f"[+] Cloning {git_url} -> {target_path}")
    Repo.clone_from(git_url, str(target_path))
    return target_path


def is_probably_binary(path: Path) -> bool:
    """
    Heuristic to detect binary files:
    - If file contains NUL bytes in the first BINARY_CHECK_BYTES -> binary
    - Or if a high proportion of bytes are non-text (control chars) -> binary
    """
    try:
        with open(path, "rb") as f:
            sample = f.read(BINARY_CHECK_BYTES)
            if not sample:
                return False
            if b"\x00" in sample:
                return True
            # compute non-text ratio
            text_chars = bytearray({7, 8, 9, 10, 12, 13, 27} | set(range(0x20, 0x7F)))
            non_text = sum(1 for b in sample if b not in text_chars)
            ratio = non_text / max(1, len(sample))
            return ratio > BINARY_NON_TEXT_THRESHOLD
    except Exception:
        # if we can't read, assume binary to be safe
        return True


def get_tracked_files(repo_path: Path):
    """
    Return a list of tracked file paths (relative to repo root) using git ls-files.
    If repo is not a git repo or listing fails, return None.
    """
    try:
        repo = Repo(str(repo_path))
        files_str = repo.git.ls_files()
        if not files_str:
            return []
        files = [repo_path / p for p in files_str.splitlines()]
        return files
    except InvalidGitRepositoryError:
        return None
    except Exception as e:
        print(f"[WARN] git ls-files failed: {e}")
        return None


def list_text_files(repo_path: Path, max_size: int = MAX_FILE_SIZE):
    """
    Produce a list of text file paths to ingest.
    Prefers git ls-files when available, otherwise walks the filesystem.
    Filters out binary files and very large files.
    """
    repo_path = Path(repo_path).resolve()
    tracked = get_tracked_files(repo_path)
    cand_files = []

    if tracked is not None:
        # use git-tracked files (clean)
        print(f"[+] Using git ls-files for {repo_path}")
        cand_files = tracked
    else:
        # fallback: scan filesystem for files
        print(f"[+] Using filesystem scan for {repo_path}")
        cand_files = [p for p in repo_path.rglob("*") if p.is_file()]

    # filter out big/binary files
    out_files = []
    for p in cand_files:
        try:
            if p.stat().st_size > max_size:
                # skip very large files
                continue
            if is_probably_binary(p):
                continue
            out_files.append(p)
        except Exception:
            continue
    return out_files


def chunk_file(filepath: Path, chunk_size=CHUNK_SIZE, overlap=OVERLAP):
    """
    Read a text file and split into chunks of lines with overlap.
    """
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
    except Exception:
        return []
    chunks = []
    i = 0
    n = len(lines)
    while i < n:
        chunk_lines = lines[i:i+chunk_size]
        text = "".join(chunk_lines).strip()
        if text:
            chunks.append({
                "file_path": str(filepath.resolve()),
                "start_line": i+1,
                "end_line": min(i+chunk_size, n),
                "text": text
            })
        i += (chunk_size - overlap)
    return chunks


def save_chunks(chunks, out_dir: Path, repo_name: str):
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    for idx, c in enumerate(chunks):
        fname = f"{repo_name}__chunk_{idx:06d}.json"
        path = out_dir / fname
        with open(path, "w", encoding="utf-8") as f:
            json.dump(c, f, ensure_ascii=False)
    print(f"[+] Saved {len(chunks)} chunks to {out_dir}")

# --- paste somewhere near save_chunks / ingest_repo functions ---

def create_folder_summary_chunks(files, repo_root: Path, repo_name: str):
    """
    Given a list of file Paths, create small summary chunks for each folder.
    Each chunk lists the files contained in that folder (relative paths).
    Returns: list of chunk dicts (same shape as chunk_file output).
    """
    from collections import defaultdict
    folders = defaultdict(list)
    for p in files:
        try:
            rel = p.resolve().relative_to(repo_root.resolve())
        except Exception:
            rel = p.name
        parent = Path(rel).parent
        folders[str(parent)].append(str(rel))

    folder_chunks = []
    for folder, file_list in folders.items():
        # skip root empty folder if it's just "."
        if folder in (".", ""):
            folder_name = repo_root.name
        else:
            folder_name = folder
        text_lines = [f"FOLDER: {folder_name}", "FILES:"]
        for fn in sorted(file_list):
            text_lines.append(f" - {fn}")
        text = "\n".join(text_lines)
        # use a synthetic file_path to indicate this is a folder-summary chunk
        chunk = {
            "file_path": str((repo_root / folder_name).resolve()) + f"::folder_summary::{repo_name}",
            "start_line": 1,
            "end_line": 1,
            "text": text
        }
        folder_chunks.append(chunk)
    return folder_chunks


def ingest_repo(git_url: str, repo_name: str, max_size: int = MAX_FILE_SIZE):
    repo_path = clone_repo(git_url, repo_name)
    print(f"[+] Inspecting files in {repo_path}")
    files = list_text_files(repo_path, max_size=max_size)
    print(f"[+] Found {len(files)} text files to chunk.")
    
    # create folder-summary chunks and include them
    folder_summary_chunks = create_folder_summary_chunks(files, repo_path, repo_name)
    all_chunks = []
    # add folder summaries first (so they get low index numbers)
    all_chunks.extend(folder_summary_chunks)

    for f in tqdm(files):
        file_chunks = chunk_file(f)
        if file_chunks:
            all_chunks.extend(file_chunks)
    save_chunks(all_chunks, OUT_DIR, repo_name)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--git", required=True, help="Git clone URL (HTTPS)")
    parser.add_argument("--name", required=False, help="Repo name (folder)", default="demo_repo")
    parser.add_argument("--max-size", type=int, default=MAX_FILE_SIZE, help="Max file size in bytes to ingest (default 1MB)")
    args = parser.parse_args()
    print(f"[+] PROJECT_ROOT = {PROJECT_ROOT}")
    print(f"[+] RAW_REPO_DIR = {RAW_REPO_DIR}")
    print(f"[+] OUT_DIR = {OUT_DIR}")
    ingest_repo(args.git, args.name, max_size=args.max_size)
