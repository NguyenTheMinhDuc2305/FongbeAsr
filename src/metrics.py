"""Validation metrics.

- syllable_wer: corpus (micro) syllable error rate = total edits / total reference syllables.
- leaderboard_score: the OFFICIAL Zindi metric (organizers' data/score.py, 2026-11-24 revision):
  per-utterance syllable WER_i = E_i / n_i, aggregated as sum(sqrt(n_i) * WER_i) / sum(sqrt(n_i)).
  Utterances with an empty reference are skipped.
"""
from rapidfuzz.distance import Levenshtein
from data.syllabify import syllabify
from data.score import measure


def score_pairs(pairs):
    errors, nref = 0, 0
    for ref, hyp in pairs:
        r, h = syllabify(ref).split(), syllabify(hyp).split()
        errors += Levenshtein.distance(r, h)
        nref += len(r)
    diagnostics = measure(pairs)
    return {"syllable_wer": errors / max(1, nref), "leaderboard_score": diagnostics["score"],
            "errors": errors, "reference_syllables": nref, "utterances": len(pairs),
            "conditional_ter": diagnostics["TER_conditional"],
            "segmental_ser": diagnostics["segmental_SER_without_tones"],
            "official_tser": diagnostics["TSER"]}
