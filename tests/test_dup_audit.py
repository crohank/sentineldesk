import numpy as np

from scripts.dup_audit import exact_cross_pairs, jaccard, lsh_candidates, minhash_signatures, normalize, shingles


def test_normalize_collapses_case_and_whitespace():
    assert normalize("  Hello\n\nWorld\tAGAIN ") == "hello world again"


def test_shingles_are_word_5grams():
    assert shingles("a b c d e f") == {"a b c d e", "b c d e f"}
    assert shingles("A  b\nc d E") == shingles("a b c d e")
    assert shingles("too short") == {"too short"}


def test_jaccard():
    assert jaccard(frozenset("ab"), frozenset("ab")) == 1.0
    assert jaccard(frozenset("ab"), frozenset("bc")) == 1 / 3


def _doc_pair(n_words: int, n_changed: int, rng) -> tuple[str, str]:
    words = [f"w{i}" for i in rng.integers(0, 10**6, size=n_words)]
    other = list(words)
    for k in rng.choice(n_words, size=n_changed, replace=False):
        other[k] = f"x{k}"
    return " ".join(words), " ".join(other)


def test_minhash_estimate_tracks_jaccard():
    rng = np.random.default_rng(0)
    errors = []
    for changed in [1, 2, 4, 8, 16]:
        a, b = _doc_pair(200, changed, rng)
        sa, sb = shingles(a), shingles(b)
        sig = minhash_signatures([sa, sb], num_perm=256)
        errors.append(abs((sig[0] == sig[1]).mean() - jaccard(sa, sb)))
    assert max(errors) < 0.1, errors


def test_signatures_are_deterministic():
    sets = [shingles("one two three four five six seven")]
    assert (minhash_signatures(sets) == minhash_signatures(sets)).all()


def test_lsh_finds_near_duplicates_and_skips_unrelated():
    rng = np.random.default_rng(1)
    a, b = _doc_pair(150, 1, rng)
    c, _ = _doc_pair(150, 0, rng)
    sets = [shingles(a), shingles(b), shingles(c)]
    assert jaccard(sets[0], sets[1]) >= 0.85
    cands = lsh_candidates(minhash_signatures(sets))
    assert (0, 1) in cands
    assert (0, 2) not in cands and (1, 2) not in cands


def test_exact_cross_pairs_matches_brute_force():
    rng = np.random.default_rng(2)
    docs = []
    for changed in [0, 1, 3, 10, 40]:
        a, b = _doc_pair(60, changed, rng)
        docs += [a, b]
    sets = [shingles(d) for d in docs]
    left, right = np.arange(0, 10, 2), np.arange(1, 10, 2)
    brute = {(int(i), int(j)) for i in left for j in right if jaccard(sets[i], sets[j]) >= 0.85}
    assert brute
    assert exact_cross_pairs(sets, left, right, chunk=2) == brute
