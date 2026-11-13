from bench.corpus import INDIRECT_SECTION, build_corpus, build_questions, locate_answer


def test_corpus_is_deterministic():
    assert build_corpus(5, seed=1) == build_corpus(5, seed=1)


def test_every_document_has_eight_sections():
    corpus = build_corpus(6)
    assert all(len(sections) == 8 for sections in corpus.values())


def test_hard_questions_point_to_the_indirect_section():
    corpus = build_corpus(10)
    questions = build_questions(corpus, n=300, hard_ratio=0.5)
    hard = [q for q in questions if q.hard]
    assert hard
    assert all(q.answer_section.endswith(f"s{INDIRECT_SECTION:02d}") for q in hard)


def test_hard_ratio_is_respected_roughly():
    corpus = build_corpus(10)
    questions = build_questions(corpus, n=2000, hard_ratio=0.06)
    share = sum(q.hard for q in questions) / len(questions)
    assert 0.04 < share < 0.08


def test_locate_answer_matches_ground_truth():
    corpus = build_corpus(10)
    for q in build_questions(corpus, n=200, hard_ratio=0.3):
        assert locate_answer(corpus, q.text) == q.answer_section
