import csv
import io
import tarfile
from pathlib import Path
import pytest
from src.common import ROOT, load_config, read_csv
from src.download import extract_needed
from src.metrics import score_pairs
from src.tokenizer import CharacterTokenizer
from src.train import verify_artifacts


def test_saved_artifacts_and_no_leakage():
    cfg = load_config("data/configs/training/proposal01.yaml")
    tok = verify_artifacts(cfg)
    assert len(tok) == 37
    train = read_csv(ROOT / cfg["data"]["train_csv"])
    vg = read_csv(ROOT / cfg["data"]["valid_general_csv"])
    vr = read_csv(ROOT / cfg["data"]["valid_religious_csv"])
    assert (len(train), len(vg), len(vr)) == (132396, 2396, 1194)
    assert not ({r["id"] for r in train} & {r["id"] for r in vg+vr})
    assert not ({r["speaker_id"] for r in train} & {r["speaker_id"] for r in vg})
    assert {r["book"] for r in vr} == {"JDG"}
    assert not any(r["book"] == "JDG" for r in train if r["source"] == "religious")
    for r in train+vg+vr:
        assert tok.encode(r["transcription"])


def test_tokenizer_repeated_letters_blank_and_tones():
    tok = CharacterTokenizer(ROOT / "data/tokenizers/fon_char_nfd_v1")
    assert tok.decode(tok.encode("nɔ́ ɖu"), ctc=False) == "nɔ́ ɖu"
    a, blank = tok.ids["a"], tok.blank_id
    assert tok.decode([a, a, blank, a]) == "aa"
    assert tok.decode([a, a], ctc=False) == "aa"
    assert tok.minimum_ctc_frames([a, a]) == 3


def test_metric_micro_not_macro_and_no_cap():
    # Syllabification of isolated vowels is unambiguous.
    score = score_pairs([("a", "a a a a"), ("a a a", "a a a")])
    assert score["syllable_wer"] == pytest.approx(3/4)
    assert score_pairs([("a", "a a a a")])["syllable_wer"] == 3
    assert score_pairs([("nɔ́", "nɔ̀")])["syllable_wer"] == 1
    assert score_pairs([("kɔ́xojí", "kɔ́ xo jí")])["syllable_wer"] == 0


def test_tar_rejects_traversal(tmp_path):
    archive = tmp_path / "bad.tar"
    with tarfile.open(archive, "w") as t:
        m = tarfile.TarInfo("../escape")
        m.size = 1
        t.addfile(m, io.BytesIO(b"x"))
    with pytest.raises(ValueError, match="Unsafe"):
        extract_needed(archive, tmp_path / "root", {"../escape"})


def test_tar_extracts_only_requested_regular_files(tmp_path):
    archive = tmp_path / "good.tar"
    with tarfile.open(archive, "w") as t:
        for name in ("audio/train/a.wav", "audio/train/b.wav"):
            m = tarfile.TarInfo(name); m.size = 1
            t.addfile(m, io.BytesIO(b"x"))
    root = tmp_path / "root"
    assert extract_needed(archive, root, {"audio/train/a.wav"}) == 1
    assert (root / "audio/train/a.wav").read_bytes() == b"x"
    assert not (root / "audio/train/b.wav").exists()


def _tar_bytes(files):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as t:
        for name, data in files.items():
            m = tarfile.TarInfo(name)
            m.size = len(data)
            t.addfile(m, io.BytesIO(data))
    return buffer.getvalue()


def _patch_remote(monkeypatch, payload, md5_of=None):
    import base64, hashlib
    import src.download as dl
    md5 = base64.b64encode(hashlib.md5(md5_of or payload).digest()).decode()
    monkeypatch.setattr(dl, "object_metadata", lambda url: {"size": len(payload), "md5": md5, "generation": "7"})
    monkeypatch.setattr(dl.urllib.request, "urlopen", lambda url, timeout=0: io.BytesIO(payload))
    monkeypatch.setattr(dl.time, "sleep", lambda s: None)
    return dl


def test_stream_fetch_extracts_only_needed_and_verifies_md5(tmp_path, monkeypatch):
    payload = _tar_bytes({"audio/train_x/a.wav": b"aaa", "audio/train_x/b.wav": b"bbb"})
    dl = _patch_remote(monkeypatch, payload)
    result = dl.fetch_archive("https://bucket/", "audio_train_x_000.tar", tmp_path, {"audio/train_x/a.wav"})
    assert result["extracted"] == 1 and result["generation"] == "7"
    assert (tmp_path / "audio/train_x/a.wav").read_bytes() == b"aaa"
    assert not (tmp_path / "audio/train_x/b.wav").exists()


def test_stream_fetch_md5_mismatch_deletes_files_and_fails(tmp_path, monkeypatch):
    payload = _tar_bytes({"audio/train_x/a.wav": b"aaa"})
    dl = _patch_remote(monkeypatch, payload, md5_of=b"something else")
    with pytest.raises(OSError, match="Integrity"):
        dl.fetch_archive("https://bucket/", "audio_train_x_000.tar", tmp_path, {"audio/train_x/a.wav"}, retries=2)
    assert not (tmp_path / "audio/train_x/a.wav").exists()


def test_generated_json_is_portable_lf(tmp_path):
    from src.common import atomic_json, identity
    import json
    p = tmp_path / "artifact.json"
    atomic_json(p, {"text": "nɔ́", "tokens": ["a", "b"]})
    assert b"\r\n" not in p.read_bytes()
    assert json.loads(p.read_text(encoding="utf-8"))["text"] == "nɔ́"
    cfg = load_config("data/configs/training/proposal01.yaml")
    assert all("\\" not in key for key in identity(cfg)["files"])


def test_preparation_reproduces_saved_artifacts(tmp_path):
    from src.prepare import prepare
    from src.common import sha256
    prepare(tmp_path)
    for rel in ("tokenizers/fon_char_nfd_v1/tokenizer.json",
                "splits/all_labels_valspk_jdg_v1/train.csv",
                "splits/all_labels_valspk_jdg_v1/valid_general.csv",
                "splits/all_labels_valspk_jdg_v1/valid_religious.csv"):
        assert sha256(tmp_path / rel) == sha256(ROOT / "data" / rel)
