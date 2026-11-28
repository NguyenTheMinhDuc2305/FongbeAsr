#!/usr/bin/env python
"""Tests of the official score (`score.py`) and of the normalisation (`normalize.py`).

    python test_score.py          (standard library + rapidfuzz, no data needed)

Each test pins a behaviour promised in `conventions.md`: score bounds, role of tones, normalisation shared by
references and submissions, syllable splitting, sqrt(n) weighting of the per-utterance rates, missing ids,
same result on plain and on syllabified text.
A failing test means the script and the published rule disagree: please report it to the organisers.
"""
import sys
import unicodedata as ud
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from normalize import tonal_target  # noqa: E402
from score import id_and_text, measure, syllables  # noqa: E402
from syllabify import syllabify  # noqa: E402


def nfd(t):
    return ud.normalize("NFD", t)


def close(a, b, tol=1e-9):
    return abs(a - b) <= tol


def test_perfect_submission():
    r = measure([("Mǐ ná yi aximɛ.", "Mǐ ná yi aximɛ.")])
    assert r["score"] == 0 and r["TSER"] == 0 and r["TER_conditional"] == 0


def test_empty_submission():
    # everything is deleted: score = 1 (and TSER = 1); no matched syllable: conditional TER = 0
    r = measure([("Mǐ ná yi aximɛ.", ""), ("ná", "")])
    assert close(r["score"], 1) and r["TSER"] == 1 and r["TER_conditional"] == 0


def test_insertions_can_exceed_one():
    # a hypothesis much longer than the reference: the rate is not capped (here 8 edits for 1 syllable)
    r = measure([("a", "kɔ́ kɔ́ kɔ́ kɔ́ kɔ́ kɔ́ kɔ́ kɔ́")])
    assert close(r["score"], 8) and close(r["TSER"], 8) and 0 <= r["TER_conditional"] <= 1


def test_wrong_tone_costs_one_syllable():
    # "ná" read as "na": one syllable out of five wrong, and one matched syllable out of five with a wrong tone
    r = measure([("mǐ ná yi axi", "mǐ na yi axi")])
    assert r["counts"]["ref_syllables"] == 5     # mǐ · ná · yi · a · xi
    assert close(r["score"], 1 / 5) and close(r["TSER"], 1 / 5) and close(r["TER_conditional"], 1 / 5)


def test_tones_removed_only():
    # same text without any tone: no segmental error, all the error is tonal (TSER = conditional TER)
    ref = "Égbé zǎnzǎn ɔ́, Afi yá wǔ gosín tɔkpá sín aximɛ."
    toneless = "".join(c for c in nfd(ref) if c not in "́̀̌̂̄")
    r = measure([(ref, toneless)])
    assert r["segmental_SER_without_tones"] == 0 and r["matched_syllable_rate"] == 1
    assert r["TSER"] > 0 and close(r["TSER"], r["TER_conditional"])


def test_shared_normalisation():
    # case, punctuation, digits, elision apostrophe, NFC/NFD, look-alike characters: none of these count
    ref = "ɖ'é wá 21 kpó, Afi !"
    hyp = "ɖé WÁ kpó afi"
    assert measure([(ref, hyp)])["score"] == 0
    assert measure([(ud.normalize("NFC", "Mɛ̌ ɔ́"), nfd("mɛ̌ ɔ́"))])["score"] == 0
    assert measure([("ɛ́ ɖo", "έ ð" + "o")])["score"] == 0          # Greek epsilon and barred d fixed


def test_word_boundaries_ignored():
    # the syllable is the unit: joining or splitting words does not change the score
    assert measure([("axɔ́sú ɔ́", "axɔ́ sú ɔ́")])["score"] == 0
    assert measure([("kpó ɖé", "kpóɖé")])["score"] == 0


def test_syllable_splitting():
    s = syllables("axɔ́sú")
    assert [b for b, _ in s] == ["a", "xɔ", "su"] and [t for _, t in s] == ["", "́", "́"]
    # syllable-final "n" attached to the vowel; "n" followed by a vowel opens the next syllable
    assert [b for b, _ in syllables("gosín")] == ["go", "sin"]
    assert [b for b, _ in syllables("ana")] == ["a", "na"]
    # syllabic consonant carrying a tone (ń): a syllable of its own
    assert syllables("ń")[0][1] == "́"
    # rising tone (caron) kept distinct from the high tone
    assert syllables("mǐ")[0][1] == "̌" and syllables("mí")[0][1] == "́"


def test_sqrt_weighted_mean_of_utterance_rates():
    # score = sum sqrt(n) * WER_utt / sum sqrt(n): utterance 1 (n = 1) has WER 1, utterance 2 (n = 7) has WER 0
    pairs = [("ná", "na"), ("mǐ ná yi axi kpó ɖé", "mǐ ná yi axi kpó ɖé")]
    r = measure(pairs)
    assert close(r["score"], 1 / (1 + 7 ** 0.5))
    assert close(r["TSER"], 1 / (1 + 7))          # diagnostic only: total edits / total syllables


def test_empty_reference_skipped():
    # an utterance whose reference has no syllable is ignored (no division by zero, no weight)
    assert measure([("?!", "kɔ́"), ("ná", "ná")])["score"] == 0


def test_plain_and_syllabified_agree():
    # the leaderboard compares syllabified files; score.py gives the same value on plain text
    ref = ["Égbé zǎnzǎn ɔ́, Afi yá wǔ gosín tɔkpá sín aximɛ.", "Mǐ ná yi gbɛ̀ ń"]
    hyp = ["egbe zǎnzan ɔ́ afi ya wǔ gosin tɔkpá aximɛ", "mǐ ná yi"]
    plain = measure(list(zip(ref, hyp)))["score"]
    syl = measure([(syllabify(r), syllabify(h)) for r, h in zip(ref, hyp)], syllabified=True)["score"]
    assert plain > 0 and close(plain, syl)


def test_missing_id_counts_as_empty():
    # in score.py, an id missing from the submission is scored with an empty transcription
    assert close(measure([("Mǐ ná yi.", "")])["score"], 1)


def test_submission_formats():
    # the Zindi submission format (ID, Target) and the format of the training files (id, transcription) are read alike
    assert id_and_text({"ID": "test_000001", "Target": "un nɔ1"}) == ("test_000001", "un nɔ1")
    assert id_and_text({"id": "test_000001", "transcription": "un nɔ1"}) == ("test_000001", "un nɔ1")
    assert id_and_text({"ID": "test_000002", "Target": ""}) == ("test_000002", "")


def test_measurement_space():
    # the normalisation used by score.py is the one of normalize.py (tonal_target)
    assert tonal_target("Kɔ́ 2 026 F, « bureau » !") == nfd("kɔ́ f bureau")


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failures = 0
    for name, f in tests:
        try:
            f(); print("ok    ", name)
        except AssertionError as e:
            failures += 1; print("FAILED", name, e)
    print(f"{len(tests) - failures}/{len(tests)} tests passed")
    sys.exit(1 if failures else 0)
