"""Tests for sub-agent output staging, kind detection, and gateway dispatch."""
from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bridge.subagent.output import (
  MAX_FILE_SIZE_BYTES,
  StagedFile,
  SkippedFile,
  cleanup_input_staging,
  detect_kind,
  format_file_list,
  stage_input_files,
  stage_output_files,
)
from bridge.messaging.gateway import send_attachment


# ---------------------------------------------------------------------------
# detect_kind
# ---------------------------------------------------------------------------

class TestDetectKind:
  def test_image_jpeg(self):
    kind, mime = detect_kind("/tmp/foo.jpg")
    assert kind == "image"
    assert mime == "image/jpeg"

  def test_image_png(self):
    kind, _ = detect_kind("/tmp/foo.png")
    assert kind == "image"

  def test_video_mp4(self):
    kind, mime = detect_kind("/tmp/foo.mp4")
    assert kind == "video"
    assert mime == "video/mp4"

  def test_audio_mp3(self):
    kind, _ = detect_kind("/tmp/foo.mp3")
    assert kind == "audio"

  def test_pdf_is_document(self):
    kind, mime = detect_kind("/tmp/foo.pdf")
    assert kind == "document"
    assert mime == "application/pdf"

  def test_csv_is_document(self):
    kind, _ = detect_kind("/tmp/foo.csv")
    assert kind == "document"

  def test_unknown_extension_falls_back_to_document(self):
    kind, mime = detect_kind("/tmp/foo.unknownext")
    assert kind == "document"
    assert mime == "application/octet-stream"

  def test_no_extension_falls_back_to_document(self):
    kind, mime = detect_kind("/tmp/somefile")
    assert kind == "document"
    assert mime == "application/octet-stream"

  def test_webp_is_image_not_sticker(self):
    # Bare .webp from a sub-agent isn't a real WhatsApp sticker (no EXIF), so
    # we send it as an image rather than as a sticker.
    kind, _ = detect_kind("/tmp/foo.webp")
    assert kind == "image"


# ---------------------------------------------------------------------------
# stage_output_files
# ---------------------------------------------------------------------------

class TestStageOutputFiles:
  def _make_file(self, dirpath: Path, name: str, size: int = 16) -> Path:
    p = dirpath / name
    p.write_bytes(b"x" * size)
    return p

  def test_stages_single_file(self, tmp_path):
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    src = self._make_file(src_dir, "report.csv", size=128)
    base = tmp_path / "media"
    result = stage_output_files("sess1", [str(src)], base_dir=base)
    assert len(result.staged) == 1
    assert result.skipped == []
    f = result.staged[0]
    assert f.name == "report.csv"
    assert f.kind == "document"
    assert f.size_bytes == 128
    assert Path(f.path).exists()
    assert Path(f.path).read_bytes() == b"x" * 128
    # Staged copy is under base/sess1/, not the original location.
    assert str(base / "sess1") in f.path

  def test_stages_multiple_kinds(self, tmp_path):
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    files = [
      self._make_file(src_dir, "a.png", 10),
      self._make_file(src_dir, "b.mp4", 20),
      self._make_file(src_dir, "c.mp3", 30),
      self._make_file(src_dir, "d.pdf", 40),
    ]
    base = tmp_path / "media"
    result = stage_output_files("s2", [str(p) for p in files], base_dir=base)
    kinds = {f.name: f.kind for f in result.staged}
    assert kinds == {
      "a.png": "image",
      "b.mp4": "video",
      "c.mp3": "audio",
      "d.pdf": "document",
    }
    assert result.skipped == []

  def test_skips_missing_file(self, tmp_path):
    base = tmp_path / "media"
    result = stage_output_files("s3", ["/nonexistent/path/to/file.txt"], base_dir=base)
    assert result.staged == []
    assert len(result.skipped) == 1
    assert "not found" in result.skipped[0].reason

  def test_skips_oversized_file(self, tmp_path, monkeypatch):
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    big = self._make_file(src_dir, "big.bin", size=2048)
    # Lower the cap to something the file exceeds without writing 200MB to disk.
    monkeypatch.setattr("bridge.subagent.output.MAX_FILE_SIZE_BYTES", 1024)
    base = tmp_path / "media"
    result = stage_output_files("s4", [str(big)], base_dir=base)
    assert result.staged == []
    assert len(result.skipped) == 1
    assert "too large" in result.skipped[0].reason

  def test_skips_directory(self, tmp_path):
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    nested = src_dir / "nested"
    nested.mkdir()
    base = tmp_path / "media"
    result = stage_output_files("s5", [str(nested)], base_dir=base)
    assert result.staged == []
    assert len(result.skipped) == 1
    assert "regular file" in result.skipped[0].reason

  def test_handles_empty_input(self, tmp_path):
    base = tmp_path / "media"
    result = stage_output_files("s6", [], base_dir=base)
    assert result.staged == []
    assert result.skipped == []

  def test_handles_empty_session_id(self, tmp_path):
    src = self._make_file(tmp_path, "x.txt")
    base = tmp_path / "media"
    result = stage_output_files("", [str(src)], base_dir=base)
    assert result.staged == []
    assert result.skipped == []

  def test_resolves_collision_in_basenames(self, tmp_path):
    a_dir = tmp_path / "a"
    b_dir = tmp_path / "b"
    a_dir.mkdir()
    b_dir.mkdir()
    a_file = a_dir / "report.csv"
    b_file = b_dir / "report.csv"
    a_file.write_text("first")
    b_file.write_text("second")
    base = tmp_path / "media"
    result = stage_output_files("s7", [str(a_file), str(b_file)], base_dir=base)
    assert len(result.staged) == 2
    names = sorted(f.name for f in result.staged)
    assert names == ["report.csv", "report_1.csv"]
    # Originals must still be readable as separate copies.
    contents = sorted(Path(f.path).read_text() for f in result.staged)
    assert contents == ["first", "second"]

  def test_partial_failure_does_not_abort_batch(self, tmp_path):
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    good = self._make_file(src_dir, "ok.txt")
    base = tmp_path / "media"
    result = stage_output_files(
      "s8",
      [str(good), "/does/not/exist.bin"],
      base_dir=base,
    )
    assert len(result.staged) == 1
    assert result.staged[0].name == "ok.txt"
    assert len(result.skipped) == 1

  def test_default_size_cap_is_200mb(self):
    assert MAX_FILE_SIZE_BYTES == 200 * 1024 * 1024


# ---------------------------------------------------------------------------
# stage_input_files / cleanup_input_staging
# ---------------------------------------------------------------------------

class TestStageInputFiles:
  def _make_file(self, dirpath: Path, name: str, size: int = 16) -> Path:
    p = dirpath / name
    p.write_bytes(b"y" * size)
    return p

  def test_copies_files_into_session_subdir(self, tmp_path):
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    a = self._make_file(src_dir, "a.txt", size=8)
    b = self._make_file(src_dir, "b.bin", size=12)
    base = tmp_path / "exchange"
    paths = stage_input_files("sess1", [str(a), str(b)], base_dir=base)
    assert len(paths) == 2
    for p in paths:
      assert Path(p).exists()
      assert (base / "sess1") in Path(p).parents
    # Names preserved
    names = sorted(Path(p).name for p in paths)
    assert names == ["a.txt", "b.bin"]
    # Contents preserved
    assert Path(paths[0]).read_bytes() == b"y" * 8
    assert Path(paths[1]).read_bytes() == b"y" * 12

  def test_skips_missing_files(self, tmp_path):
    paths = stage_input_files(
      "s2", ["/nonexistent/foo.txt"], base_dir=tmp_path / "x"
    )
    assert paths == []

  def test_skips_directories(self, tmp_path):
    sub = tmp_path / "src" / "nested"
    sub.mkdir(parents=True)
    paths = stage_input_files("s3", [str(sub)], base_dir=tmp_path / "x")
    assert paths == []

  def test_skips_oversized(self, tmp_path, monkeypatch):
    src = self._make_file(tmp_path, "big.bin", size=4096)
    monkeypatch.setattr("bridge.subagent.output.MAX_FILE_SIZE_BYTES", 1024)
    paths = stage_input_files("s4", [str(src)], base_dir=tmp_path / "x")
    assert paths == []

  def test_handles_empty_inputs(self, tmp_path):
    assert stage_input_files("s5", [], base_dir=tmp_path) == []

  def test_handles_empty_session_id(self, tmp_path):
    src = self._make_file(tmp_path, "x.txt")
    assert stage_input_files("", [str(src)], base_dir=tmp_path / "x") == []

  def test_resolves_basename_collisions(self, tmp_path):
    a_dir = tmp_path / "a"
    b_dir = tmp_path / "b"
    a_dir.mkdir()
    b_dir.mkdir()
    a_file = a_dir / "img.jpg"
    b_file = b_dir / "img.jpg"
    a_file.write_bytes(b"first")
    b_file.write_bytes(b"second")
    base = tmp_path / "exchange"
    paths = stage_input_files("s6", [str(a_file), str(b_file)], base_dir=base)
    assert len(paths) == 2
    names = {Path(p).name for p in paths}
    assert "img.jpg" in names
    # Second file got renamed to avoid clobbering the first
    assert any(n.startswith("img_") and n.endswith(".jpg") for n in names)

  def test_returns_absolute_resolved_paths(self, tmp_path):
    src = self._make_file(tmp_path, "x.txt")
    paths = stage_input_files("s7", [str(src)], base_dir=tmp_path / "exchange")
    assert len(paths) == 1
    assert os.path.isabs(paths[0])
    # Path matches a real file
    assert os.path.isfile(paths[0])


class TestCleanupInputStaging:
  def test_removes_session_dir(self, tmp_path):
    base = tmp_path / "exchange"
    src = tmp_path / "x.txt"
    src.write_text("hi")
    paths = stage_input_files("sess1", [str(src)], base_dir=base)
    assert paths and Path(paths[0]).exists()
    cleanup_input_staging("sess1", base_dir=base)
    assert not (base / "sess1").exists()

  def test_no_op_for_missing_dir(self, tmp_path):
    # Should not raise
    cleanup_input_staging("never_existed", base_dir=tmp_path / "exchange")

  def test_no_op_for_empty_session_id(self, tmp_path):
    cleanup_input_staging("", base_dir=tmp_path)


# ---------------------------------------------------------------------------
# format_file_list
# ---------------------------------------------------------------------------

class TestFormatFileList:
  def test_empty_returns_empty_string(self):
    assert format_file_list([], []) == ""

  def test_only_staged(self):
    staged = [
      StagedFile(path="/x/a.png", name="a.png", size_bytes=2048, mime="image/png", kind="image"),
    ]
    text = format_file_list(staged, [])
    assert "Output files attached (1):" in text
    assert "a.png" in text
    assert "image" in text

  def test_with_skipped(self):
    skipped = [SkippedFile(source_path="/x/big.bin", name="big.bin", reason="file too large (250.0 MB > 200 MB)")]
    text = format_file_list([], skipped)
    assert "Files skipped (1):" in text
    assert "big.bin" in text
    assert "too large" in text


# ---------------------------------------------------------------------------
# send_attachment gateway helper
# ---------------------------------------------------------------------------

class FakeWS:
  def __init__(self):
    self.sent: list[str] = []

  async def send(self, payload: str) -> None:
    self.sent.append(payload)


class TestSendAttachment:
  def test_sends_send_message_payload_with_attachments(self):
    ws = FakeWS()
    asyncio.run(send_attachment(
      ws,
      chat_id="123@s.whatsapp.net",
      attachment_path="/data/media/subagent_out/sess/file.pdf",
      kind="document",
      request_id="req-1",
      file_name="file.pdf",
    ))
    assert len(ws.sent) == 1
    msg = json.loads(ws.sent[0])
    assert msg["type"] == "send_message"
    payload = msg["payload"]
    assert payload["chatId"] == "123@s.whatsapp.net"
    assert payload["requestId"] == "req-1"
    assert "text" not in payload
    assert payload["attachments"] == [{
      "kind": "document",
      "path": "/data/media/subagent_out/sess/file.pdf",
      "fileName": "file.pdf",
    }]

  def test_includes_reply_to_when_provided(self):
    ws = FakeWS()
    asyncio.run(send_attachment(
      ws,
      chat_id="g@g.us",
      attachment_path="/data/media/foo.png",
      kind="image",
      request_id="req-2",
      reply_to="000125",
    ))
    msg = json.loads(ws.sent[0])
    assert msg["payload"]["replyTo"] == "000125"

  def test_includes_caption_when_provided(self):
    ws = FakeWS()
    asyncio.run(send_attachment(
      ws,
      chat_id="g@g.us",
      attachment_path="/data/media/foo.png",
      kind="image",
      request_id="req-3",
      caption="hello",
    ))
    att = json.loads(ws.sent[0])["payload"]["attachments"][0]
    assert att["caption"] == "hello"

  def test_noop_when_path_missing(self):
    ws = FakeWS()
    asyncio.run(send_attachment(
      ws,
      chat_id="g@g.us",
      attachment_path="",
      kind="image",
      request_id="req-4",
    ))
    assert ws.sent == []

  def test_noop_when_kind_missing(self):
    ws = FakeWS()
    asyncio.run(send_attachment(
      ws,
      chat_id="g@g.us",
      attachment_path="/x.png",
      kind="",
      request_id="req-5",
    ))
    assert ws.sent == []
