"""Corpus (micro) syllable-WER, not the capped composite in score.py."""
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
    return {"syllable_wer": errors / max(1, nref), "errors": errors,
            "reference_syllables": nref, "utterances": len(pairs),
            "conditional_ter": diagnostics["TER_conditional"],
            "segmental_ser": diagnostics["segmental_SER_without_tones"],
            "official_capped_tser": diagnostics["TSER"],
            "official_composite": diagnostics["score"]}
