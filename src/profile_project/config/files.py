from __future__ import annotations

import contextlib
import errno
import json
import os
import shutil
import tempfile
from collections.abc import Iterator
from pathlib import Path

# ---------------------------------------------------------------------------
# Low-level primitives
# ---------------------------------------------------------------------------


def atomic_rename(tmp: Path, final: Path) -> None:
    """Replace *final* with *tmp* atomically (same-volume).

    On Windows a ``PermissionError`` with ``errno`` ``EPERM`` or ``EACCES``
    means the destination is locked by an AV scanner or shell extension; we
    fall back to a copy+unlink so the caller never sees a half-written file.
    Any other error is re-raised unchanged.
    """
    try:
        os.replace(os.fspath(tmp), os.fspath(final))
    except PermissionError as exc:
        if exc.errno in (errno.EPERM, errno.EACCES):
            shutil.copyfile(os.fspath(tmp), os.fspath(final))
            os.unlink(os.fspath(tmp))
        else:
            raise


def atomic_write_json(path: Path, data: object) -> None:
    """Serialize *data* to JSON and write it to *path* atomically.

    The temp file is created in the **same directory** as *path* so that the
    subsequent ``os.replace`` is guaranteed to stay on the same volume.
    Parent directories are created if absent.  The file descriptor is
    ``flush()``-ed and ``fsync()``-ed before the rename so a crash cannot
    leave a partial file at the final path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False)
    fd, tmp_name = tempfile.mkstemp(dir=os.fspath(path.parent), suffix=".tmp")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        atomic_rename(tmp, path)
    except BaseException:
        if tmp.exists():
            os.unlink(os.fspath(tmp))
        raise


def _atomic_write_text(path: Path, text: str) -> None:
    """Write arbitrary text to *path* atomically (same-dir temp + fsync + rename)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=os.fspath(path.parent), suffix=".tmp")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        atomic_rename(tmp, path)
    except BaseException:
        if tmp.exists():
            os.unlink(os.fspath(tmp))
        raise


def append_jsonl(path: Path, obj: dict[str, object]) -> None:
    """Append one JSON object followed by ``"\\n"`` to *path*.

    Creates the parent directory and file when absent.  The file descriptor
    is ``flush()``-ed and ``fsync()``-ed before close.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(obj, ensure_ascii=False) + "\n"
    with open(os.fspath(path), "a", encoding="utf-8") as handle:
        handle.write(line)
        handle.flush()
        os.fsync(handle.fileno())


# ---------------------------------------------------------------------------
# .gitignore helper
# ---------------------------------------------------------------------------


def ensure_gitignore_entry(root: Path, entry: str) -> bool:
    """Ensure *entry* appears as its own line in ``root/.gitignore``.

    Creates the file (and *root* if absent) when needed; appends the line
    when the file exists but does not already contain *entry*.  The write is
    performed via a temp file + ``fsync`` + atomic rename.

    Returns ``True`` if the entry was added, ``False`` if it was already
    present (idempotent).
    """
    root = Path(root)
    gitignore = root / ".gitignore"
    if gitignore.exists():
        existing = gitignore.read_text(encoding="utf-8")
        lines = existing.splitlines()
        if entry in lines:
            return False
        suffix = "" if not existing or existing.endswith("\n") else "\n"
        _atomic_write_text(gitignore, existing + suffix + entry + "\n")
        return True
    _atomic_write_text(gitignore, entry + "\n")
    return True


# ---------------------------------------------------------------------------
# Root-prefix rewriter (§6b.5 root-move re-homing)
# ---------------------------------------------------------------------------


def rewrite_root_prefix(run_data_root: Path, old_root: str, new_root: str) -> int:
    """Replace every occurrence of *old_root* with *new_root* in all
    ``*.json`` and ``*.jsonl`` files under *run_data_root* (recursive).

    Only files whose text actually contains *old_root* are rewritten; each
    rewrite uses a temp file + ``fsync`` + atomic rename.

    Returns the number of files rewritten.
    """
    run_data_root = Path(run_data_root)
    if old_root == new_root or not run_data_root.exists():
        return 0
    rewritten = 0
    for path in sorted(run_data_root.rglob("*")):
        if not path.is_file() or path.suffix not in (".json", ".jsonl"):
            continue
        original = path.read_text(encoding="utf-8")
        if old_root not in original:
            continue
        _atomic_write_text(path, original.replace(old_root, new_root))
        rewritten += 1
    return rewritten


# ---------------------------------------------------------------------------
# All-or-nothing transaction
# ---------------------------------------------------------------------------


class Transaction:
    """Staged write handle yielded by :func:`transaction`.

    Tracks every file and directory it creates so that :meth:`rollback` can
    remove them deepest-first, leaving zero residue.
    """

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.created_files: list[Path] = []
        self.created_dirs: list[Path] = []

    def _ensure_dirs(self, path: Path) -> None:
        """Create missing parent directories and record them for rollback."""
        missing: list[Path] = []
        current = path.parent
        while not current.exists():
            missing.append(current)
            current = current.parent
        for directory in reversed(missing):
            directory.mkdir()
            self.created_dirs.append(directory)

    def write_json(self, path: Path, data: object) -> None:
        """Write *data* as JSON to *path*, recording the file for rollback if new."""
        path = Path(path)
        self._ensure_dirs(path)
        existed = path.exists()
        atomic_write_json(path, data)
        if not existed:
            self.created_files.append(path)

    def append_jsonl(self, path: Path, obj: dict[str, object]) -> None:
        """Append *obj* as a JSONL line to *path*.

        Records the file for rollback if it is newly created by this transaction.
        """
        path = Path(path)
        self._ensure_dirs(path)
        existed = path.exists()
        append_jsonl(path, obj)
        if not existed:
            self.created_files.append(path)

    def rollback(self) -> None:
        """Remove every file and directory this transaction created."""
        for file_path in reversed(self.created_files):
            if file_path.exists():
                os.unlink(os.fspath(file_path))
        # deepest-first so children are removed before parents
        for directory in sorted(
            self.created_dirs, key=lambda p: len(p.parts), reverse=True
        ):
            if directory.exists():
                with contextlib.suppress(OSError):
                    directory.rmdir()


@contextlib.contextmanager
def transaction(root: Path) -> Iterator[Transaction]:
    """Context manager for all-or-nothing staged writes under *root*.

    On **any** exception inside the ``with`` block the transaction rolls back,
    removing all files and directories it created, before re-raising.  On a
    clean exit the staged writes are kept.
    """
    txn = Transaction(root)
    try:
        yield txn
    except BaseException:
        txn.rollback()
        raise
