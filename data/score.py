#!/usr/bin/env python
"""Official score of the "Frugal Tonal ASR for Fongbe" competition.

    python score.py --reference Reference.csv --submission my_submission.csv [--usage Public|Private] [--output report.json]
    python score.py --reference Reference_syllables.csv --submission submission.csv --syllabified

The submission uses the Zindi format `ID, Target` (see SampleSubmission.csv); the columns `id, transcription`
of the training files are also accepted, so that local validation files can be scored as they are.
The reference has `id, transcription` (or `ID, Target`), and optionally `Usage`.
By default both files hold plain Fongbe text: each text goes through `syllabify()` (normalisation by `normalize.py`:
Unicode NFD, lower case, punctuation, digits and elision apostrophes removed; then syllables with tone digits).
With --syllabified, both files are already syllabified (like the leaderboard ground truth and your submission file).

Leaderboard metric = length-normalised weighted WER on syllables (a syllable = its letters AND its tone):
  for each utterance i with n_i reference syllables and E_i edits (substitutions + deletions + insertions):
      WER_i = E_i / n_i
  score = sum_i sqrt(n_i) * WER_i / sum_i sqrt(n_i)
Lower is better: 0 = perfect, 1 = empty submission, above 1 possible when a submission inserts many syllables.
Utterances whose reference is empty are skipped. In score.py, an id missing from the submission is scored as an empty
transcription (the leaderboard rejects a submission with missing ids).
Diagnostics (no effect on the ranking): corpus syllable error rate TSER (total edits / total syllables), segmental
syllable error without tones, matched syllable rate, conditional TER (share of wrong tones among the syllables matched
without looking at the tone) and TER per tone.
"""
import argparse
import collections
import csv
import json
import math
import re
import sys
import unicodedata as ud
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from normalize import tonal_target  # noqa: E402
from syllabify import syllabify  # noqa: E402

from rapidfuzz.distance import Levenshtein  # required, version pinned in requirements.txt

VOWELS = set("aeiouɛɔə")
TONES = {"́": "high", "̀": "low", "̌": "rising", "̂": "falling", "̄": "mid"}
TONE_NAMES = {"1": "high", "2": "low", "3": "rising", "4": "falling", "5": "mid"}


def units(word):
    """Base characters with their combining marks: [[char, [marks]], ...]."""
    out = []
    for c in word:
        if ud.category(c).startswith("M") and out:
            out[-1][1].append(c)
        elif not ud.category(c).startswith("M"):
            out.append([c, []])
    return out


def syllables(text):
    """List of (base, tone): base = onset + vowel (+ syllable-final n), tone = sorted tone marks ('' = unmarked)."""
    res = []
    for word in tonal_target(text or "").split():
        u = units(word); i = 0; onset = ""
        while i < len(u):
            b, m = u[i]
            tone = "".join(sorted(x for x in m if x in TONES)); other = "".join(x for x in m if x not in TONES)
            if b in VOWELS:
                base = onset + b + other; onset = ""
                if i + 1 < len(u) and u[i + 1][0] == "n" and not u[i + 1][1] and (i + 2 >= len(u) or u[i + 2][0] not in VOWELS):
                    base += "n"; i += 1
                res.append((base, tone)); i += 1
            elif tone:                                  # syllabic consonant carrying a tone (ń, m̀…)
                res.append((onset + b + other, tone)); onset = ""; i += 1
            else:
                onset += b + other; i += 1
        if onset:
            res.append((onset, ""))
    return res


def _distance(a, b):
    return Levenshtein.distance(a, b)


def _equal_pairs(a, b):
    """Pairs (i, j) matched identically in the rapidfuzz edit alignment."""
    for op in Levenshtein.opcodes(a, b):
        if op.tag == "equal":
            for k in range(op.src_end - op.src_start):
                yield op.src_start + k, op.dest_start + k


def tokens(text, syllabified=False):
    """Syllable tokens as on the leaderboard: letters + tone digits, e.g. 'un nɔ1 ɖu nu3'."""
    return (" ".join((text or "").split()) if syllabified else syllabify(text or "")).split()


def _base_tone(token):
    return re.sub(r"[0-9]", "", token), "".join(sorted(re.sub(r"[^0-9]", "", token)))


def measure(pairs, syllabified=False):
    """pairs: list of (reference, hypothesis). Score = sqrt(n)-weighted mean of the per-utterance syllable WER."""
    c = collections.Counter(); per_tone = collections.Counter(); per_tone_err = collections.Counter()
    weighted, weight_sum = 0.0, 0.0
    for ref, hyp in pairs:
        r, h = tokens(ref, syllabified), tokens(hyp, syllabified)
        if not r:
            continue                                   # empty reference: skipped, as on the leaderboard
        e = _distance(r, h)
        weighted += math.sqrt(len(r)) * e / len(r); weight_sum += math.sqrt(len(r))
        c["utterances_scored"] += 1; c["ref_syllables"] += len(r); c["tonal_distance"] += e
        rbt, hbt = [_base_tone(t) for t in r], [_base_tone(t) for t in h]
        rb, hb = [b for b, _ in rbt], [b for b, _ in hbt]
        c["segmental_distance"] += _distance(rb, hb)
        for i, j in _equal_pairs(rb, hb):
            tr, th = rbt[i][1], hbt[j][1]
            c["matched"] += 1; c["wrong_tone"] += tr != th
            cat = "+".join(TONE_NAMES[x] for x in tr) or "unmarked"
            per_tone[cat] += 1; per_tone_err[cat] += tr != th
    n = max(1, c["ref_syllables"])
    return {"score": weighted / weight_sum if weight_sum else 0.0,
            "TSER": c["tonal_distance"] / n,
            "TER_conditional": c["wrong_tone"] / max(1, c["matched"]),
            "segmental_SER_without_tones": c["segmental_distance"] / n,
            "matched_syllable_rate": c["matched"] / n,
            "TER_per_tone": {k: round(per_tone_err[k] / per_tone[k], 4) for k in sorted(per_tone)},
            "counts": {k: int(v) for k, v in c.items()}}


def read_csv(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def id_and_text(row):
    """(id, text) of a CSV row, in the Zindi format `ID, Target` or in the format `id, transcription`."""
    if "ID" in row:
        return row["ID"], row.get("Target") or ""
    return row["id"], row.get("transcription") or ""


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--reference", required=True)
    ap.add_argument("--submission", required=True)
    ap.add_argument("--usage", choices=["Public", "Private"], help="score only one part of the test")
    ap.add_argument("--output", help="JSON file for the detailed report")
    ap.add_argument("--syllabified", action="store_true",
                    help="both files are already syllabified (e.g. Reference_syllables.csv and your submission file)")
    a = ap.parse_args()
    refs = [r for r in read_csv(a.reference) if not a.usage or r.get("Usage") == a.usage]
    refs = [id_and_text(r) for r in refs]
    sub = dict(id_and_text(r) for r in read_csv(a.submission))
    missing = sum(1 for i, _ in refs if i not in sub)
    report = measure([(text, sub.get(i, "")) for i, text in refs], syllabified=a.syllabified)
    report.update({"utterances": len(refs), "missing_ids_scored_as_empty": missing, "usage": a.usage or "all"})
    print(f"score {report['score']:.6f}  (diagnostics: TSER {report['TSER']:.4f}, conditional TER {report['TER_conditional']:.4f})  "
          f"({len(refs)} utterances, {missing} missing)")
    if a.output:
        Path(a.output).write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
