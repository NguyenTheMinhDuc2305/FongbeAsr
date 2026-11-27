"""Saved character vocabulary, independent of model and Hugging Face Transformers."""
import json
import unicodedata
from pathlib import Path
from data.normalize import tonal_target


class CharacterTokenizer:
    def __init__(self, path=None, config=None):
        self.path = Path(path) if path is not None else None
        self.config = config if config is not None else json.loads((self.path / "tokenizer.json").read_text(encoding="utf-8"))
        self.tokens = self.config["tokens"]
        if len(set(self.tokens)) != len(self.tokens):
            raise ValueError("Duplicate vocabulary tokens")
        self.ids = {s: i for i, s in enumerate(self.tokens)}
        self.blank_id = self.ids["[PAD]"]
        self.unk_id = self.ids["[UNK]"]
        self.delimiter_id = self.ids["|"]
        if self.config["normalization"] != "tonal_target_digits_false_nfd":
            raise ValueError("Unsupported normalization")

    def __len__(self):
        return len(self.tokens)

    def encode(self, text, strict=True):
        text = tonal_target(text)
        out = []
        for c in text:
            t = "|" if c == " " else c
            if strict and t not in self.ids:
                raise ValueError(f"Out-of-vocabulary character {c!r}")
            out.append(self.ids.get(t, self.unk_id))
        return out

    def decode(self, ids, ctc=True):
        out, previous = [], None
        for item in ids:
            i = int(item)
            if i < 0 or i >= len(self):
                raise ValueError(f"Invalid token ID {i}")
            if i != self.blank_id and (not ctc or i != previous):
                # UNK is exposed, not silently turned into a correct letter.
                out.append(self.tokens[i])
            previous = i
        return unicodedata.normalize("NFC", "".join(out).replace("|", " ").strip())

    @staticmethod
    def minimum_ctc_frames(ids):
        return len(ids) + sum(a == b for a, b in zip(ids, ids[1:]))
