"""Sub-agent input/output staging and dispatch helpers.

The sub-agent service writes output files under its per-session workdir
(default `/storage/subagent_work/<session_id>/...` when running via the
recommended docker-compose; legacy native runs may use `/tmp/work/...`).
The Node gateway only allows attachments whose real path is inside
`MEDIA_DIR` or `STICKERS_DIR` (see
`src/mediaHandler.js::resolveAllowedAttachmentPath`). To satisfy that sandbox
without weakening it, we copy each output file into

    <MEDIA_DIR>/subagent_out/<session_id>/<basename>

before dispatching it to WhatsApp. Defense in depth — Node still validates
the final path.

This module also stages **input** files: WhatsApp media that LLM2 wants the
sub-agent to operate on first gets copied into

    <SUBAGENT_INPUT_STAGING_DIR>/<session_id>/<basename>

before being passed to `/execute`. The path must be reachable by the
sub-agent process. In the docker-compose contract both sides bind-mount
`/storage` (set ``SUBAGENT_INPUT_STAGING_DIR=/storage/subagent_in``); for
native deployments the default is ``<project_root>/data/subagent_in``.

This module:
  - Stages output files into the Node sandbox dir, dropping files that are
    missing or larger than ``MAX_FILE_SIZE_BYTES``.
  - Stages input files into the cross-process exchange dir.
  - Detects the WhatsApp attachment ``kind`` from the file's MIME type.
  - Renders a human-readable file list to embed into the
    ``[SUBTASK FINISHED]`` system message so LLM2 can reference the files
    naturally in its text reply.
"""
from __future__ import annotations

import mimetypes
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

try:
  from ..log import setup_logging
except ImportError:
  import sys
  from pathlib import Path as _Path
  sys.path.append(str(_Path(__file__).resolve().parent.parent.parent))
  from bridge.log import setup_logging  # type: ignore


logger = setup_logging()


# 200 MB per file. Above this we drop the file rather than risk a WhatsApp
# rejection or a slow upload.
MAX_FILE_SIZE_BYTES = 200 * 1024 * 1024


def _project_root() -> Path:
  # python/bridge/subagent/output.py → up four levels = repo root
  return Path(__file__).resolve().parent.parent.parent.parent


def _media_dir() -> Path:
  raw = os.getenv("MEDIA_DIR")
  if raw:
    return Path(raw).expanduser().resolve()
  return (_project_root() / "data" / "media").resolve()


def staging_root() -> Path:
  """Root directory for staged sub-agent outputs (`<MEDIA_DIR>/subagent_out`)."""
  return _media_dir() / "subagent_out"


def input_staging_root() -> Path:
  """Root directory for staged sub-agent **inputs**.

  This must be a path that the sub-agent process can read. Set
  ``SUBAGENT_INPUT_STAGING_DIR`` to a host directory shared with the
  sub-agent — e.g. ``/storage/subagent_in`` for the docker-compose contract,
  or any other path co-located with the sub-agent process.

  When the env var is unset, fall back to ``<project_root>/data/subagent_in``,
  which is writable for native (non-docker) deployments and avoids assuming
  ``/storage`` exists with the right permissions on the host.
  """
  raw = os.getenv("SUBAGENT_INPUT_STAGING_DIR")
  if raw:
    return Path(raw).expanduser().resolve()
  return (_project_root() / "data" / "subagent_in").resolve()


@dataclass(frozen=True)
class StagedFile:
  path: str  # absolute, real path inside MEDIA_DIR
  name: str  # original basename
  size_bytes: int
  mime: str  # best-effort MIME type, may be ``application/octet-stream``
  kind: str  # one of: image, video, audio, document


@dataclass(frozen=True)
class SkippedFile:
  source_path: str
  name: str
  reason: str  # human-readable, embedded in [SUBTASK FINISHED]


@dataclass(frozen=True)
class StagedOutputs:
  staged: list[StagedFile]
  skipped: list[SkippedFile]


_EXT_MIME_OVERRIDES = {
  # ``mimetypes`` doesn't ship a webp mapping on every Python build.
  ".webp": "image/webp",
}


def detect_kind(path: str | os.PathLike[str]) -> tuple[str, str]:
  """Return ``(kind, mime)`` for the given file path.

  ``kind`` is one of ``image``, ``video``, ``audio``, ``document``. It maps to
  the WhatsApp/Baileys media kind expected by ``src/wa/outbound.js``.

  Stickers are intentionally not auto-detected — proper WhatsApp stickers
  require custom EXIF metadata that sub-agents won't produce. A bare ``.webp``
  is sent as an ``image`` instead.
  """
  path_str = str(path)
  ext = os.path.splitext(path_str)[1].lower()
  mime = _EXT_MIME_OVERRIDES.get(ext)
  if mime is None:
    guessed, _ = mimetypes.guess_type(path_str)
    mime = guessed or "application/octet-stream"
  if mime.startswith("image/"):
    return "image", mime
  if mime.startswith("video/"):
    return "video", mime
  if mime.startswith("audio/"):
    return "audio", mime
  return "document", mime


def _format_size(size_bytes: int) -> str:
  if size_bytes < 1024:
    return f"{size_bytes} B"
  if size_bytes < 1024 * 1024:
    return f"{size_bytes / 1024:.1f} KB"
  return f"{size_bytes / (1024 * 1024):.1f} MB"


def stage_output_files(
  session_id: str,
  raw_paths: Iterable[str],
  *,
  base_dir: Path | None = None,
) -> StagedOutputs:
  """Copy ``raw_paths`` into ``<base_dir>/<session_id>/`` and validate them.

  Files that are missing, not regular files, or larger than
  ``MAX_FILE_SIZE_BYTES`` are reported as ``SkippedFile`` instead of being
  copied. Copy errors are also captured as skips so a single bad file does
  not abort the rest of the batch.

  ``base_dir`` defaults to :func:`staging_root` and is overridable for tests.
  """
  staged: list[StagedFile] = []
  skipped: list[SkippedFile] = []

  if not session_id:
    logger.warning("stage_output_files called with empty session_id; nothing staged")
    return StagedOutputs(staged=[], skipped=[])

  paths = [str(p) for p in raw_paths if isinstance(p, (str, os.PathLike))]
  if not paths:
    return StagedOutputs(staged=[], skipped=[])

  target_root = (base_dir or staging_root()) / session_id
  try:
    target_root.mkdir(parents=True, exist_ok=True)
  except OSError as err:
    logger.exception(
      "stage_output_files: failed to create staging dir %s: %s",
      target_root, err,
    )
    for src in paths:
      skipped.append(SkippedFile(
        source_path=src,
        name=os.path.basename(src),
        reason=f"staging dir unavailable: {err}",
      ))
    return StagedOutputs(staged=[], skipped=skipped)

  used_names: set[str] = set()
  for src in paths:
    name = os.path.basename(src) or "unnamed"
    if not src or not os.path.exists(src):
      skipped.append(SkippedFile(source_path=src, name=name, reason="file not found"))
      continue
    if not os.path.isfile(src):
      skipped.append(SkippedFile(source_path=src, name=name, reason="not a regular file"))
      continue
    try:
      size = os.path.getsize(src)
    except OSError as err:
      skipped.append(SkippedFile(source_path=src, name=name, reason=f"stat failed: {err}"))
      continue
    if size > MAX_FILE_SIZE_BYTES:
      skipped.append(SkippedFile(
        source_path=src,
        name=name,
        reason=f"file too large ({_format_size(size)} > 200 MB)",
      ))
      continue

    # Avoid clobbering when two source files share a basename.
    final_name = name
    counter = 1
    while final_name in used_names or (target_root / final_name).exists():
      stem, ext = os.path.splitext(name)
      final_name = f"{stem}_{counter}{ext}"
      counter += 1
    used_names.add(final_name)

    dest = target_root / final_name
    try:
      shutil.copyfile(src, dest)
    except OSError as err:
      skipped.append(SkippedFile(source_path=src, name=name, reason=f"copy failed: {err}"))
      continue

    real_dest = str(dest.resolve())
    kind, mime = detect_kind(real_dest)
    staged.append(StagedFile(
      path=real_dest,
      name=final_name,
      size_bytes=size,
      mime=mime,
      kind=kind,
    ))

  return StagedOutputs(staged=staged, skipped=skipped)


def stage_input_files(
  session_id: str,
  raw_paths: Iterable[str],
  *,
  base_dir: Path | None = None,
) -> list[str]:
  """Copy ``raw_paths`` into ``<base_dir>/<session_id>/`` for the sub-agent.

  Returns the list of staged absolute paths in input order, omitting files
  that are missing / unreadable / oversized. Used to bridge the filesystem
  gap between the bridge process and a containerised sub-agent: the staged
  paths must live under a directory both sides have mounted (configured via
  ``SUBAGENT_INPUT_STAGING_DIR``; see :func:`input_staging_root`).
  """
  if not session_id:
    logger.warning("stage_input_files called with empty session_id; nothing staged")
    return []

  paths = [str(p) for p in raw_paths if isinstance(p, (str, os.PathLike))]
  if not paths:
    return []

  target_root = (base_dir or input_staging_root()) / session_id
  try:
    target_root.mkdir(parents=True, exist_ok=True)
  except OSError as err:
    logger.exception(
      "stage_input_files: failed to create staging dir %s: %s",
      target_root, err,
    )
    return []

  staged_paths: list[str] = []
  used_names: set[str] = set()
  for src in paths:
    name = os.path.basename(src) or "unnamed"
    if not src or not os.path.exists(src):
      logger.warning("stage_input_files: source missing, skipping: %s", src)
      continue
    if not os.path.isfile(src):
      logger.warning("stage_input_files: not a regular file, skipping: %s", src)
      continue
    try:
      size = os.path.getsize(src)
    except OSError as err:
      logger.warning("stage_input_files: stat failed for %s: %s", src, err)
      continue
    if size > MAX_FILE_SIZE_BYTES:
      logger.warning(
        "stage_input_files: %s is too large (%s); skipping",
        src, _format_size(size),
      )
      continue

    final_name = name
    counter = 1
    while final_name in used_names or (target_root / final_name).exists():
      stem, ext = os.path.splitext(name)
      final_name = f"{stem}_{counter}{ext}"
      counter += 1
    used_names.add(final_name)

    dest = target_root / final_name
    try:
      shutil.copyfile(src, dest)
    except OSError as err:
      logger.warning("stage_input_files: copy failed for %s: %s", src, err)
      continue
    staged_paths.append(str(dest.resolve()))

  return staged_paths


def cleanup_input_staging(session_id: str, *, base_dir: Path | None = None) -> None:
  """Remove ``<base_dir>/<session_id>/`` after the sub-agent finishes.

  Errors are logged and swallowed — leaking a staging dir is far less
  bad than crashing the post-task cleanup path.
  """
  if not session_id:
    return
  target_root = (base_dir or input_staging_root()) / session_id
  if not target_root.exists():
    return
  try:
    shutil.rmtree(target_root)
  except OSError as err:
    logger.warning(
      "cleanup_input_staging: failed to remove %s: %s",
      target_root, err,
    )


def format_file_list(staged: list[StagedFile], skipped: list[SkippedFile]) -> str:
  """Render staged + skipped files for embedding into ``[SUBTASK FINISHED]``.

  Returns an empty string when there is nothing to mention.
  """
  if not staged and not skipped:
    return ""
  lines: list[str] = []
  if staged:
    lines.append(f"Output files attached ({len(staged)}):")
    for f in staged:
      lines.append(f"- {f.name} ({f.kind}, {_format_size(f.size_bytes)})")
  if skipped:
    if lines:
      lines.append("")
    lines.append(f"Files skipped ({len(skipped)}):")
    for s in skipped:
      lines.append(f"- {s.name}: {s.reason}")
  return "\n".join(lines)
