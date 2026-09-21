import json

import pytest

import eval_grounding as ev


@pytest.fixture(scope="module")
def corpus_index():
    return ev.corpus_surnames_by_year()


def test_parenthetical_citation_is_extracted():
    assert ("Lally", "2010") in ev.extract_citations(
        "One missed day is fine (Lally et al., 2010)."
    )


def test_narrative_citation_is_extracted():
    assert ("Dai", "2014") in ev.extract_citations(
        "Dai, Milkman & Riis (2014) found temporal landmarks boost restarts."
    )


def test_multi_author_parenthetical_is_extracted():
    found = ev.extract_citations("(Wohl, Pychyl & Bennett, 2010)")

    assert ("Wohl", "2010") in found
    assert ("Pychyl", "2010") in found


def test_prose_years_are_not_citations():
    found = ev.extract_citations(
        "Since 2010 you have been reading. That 2014 goal is still open."
    )

    assert found == set()


def test_common_words_are_not_treated_as_surnames():
    found = ev.extract_citations("The median was 66 days (The Study, 2010).")

    assert ("The", "2010") not in found
    assert ("Study", "2010") not in found


def test_a_real_answer_parses_cleanly(corpus_index):
    answer = (
        "The best evidence puts the median at about 66 days, but the range across "
        "individuals was huge - 18 to 254 days (Lally et al., 2010). Temporal "
        "landmarks help you restart (Dai, Milkman & Riis, 2014)."
    )

    citations = ev.extract_citations(answer)

    assert ("Lally", "2010") in citations
    assert ("Dai", "2014") in citations
    assert all(citation in corpus_index for citation in citations)


def test_corpus_index_covers_the_known_sources(corpus_index):
    assert ("Lally", "2010") in corpus_index
    assert ("Gollwitzer", "2006") in corpus_index
    assert ("Fogg", "2019") in corpus_index
    assert ("Clear", "2018") in corpus_index


def test_a_citation_outside_the_corpus_is_flagged(corpus_index):
    result = ev.grade_citations(
        "A Stanford team showed 6am is optimal (Hernandez et al., 2019).",
        retrieved_sources=set(),
        corpus_index=corpus_index,
    )

    assert result["fabricated_sources"] == ["Hernandez 2019"]
    assert result["no_fabricated_sources"] is False
    assert result["citations_grounded"] is False


def test_citing_a_corpus_source_that_was_not_retrieved_is_flagged(corpus_index):
    result = ev.grade_citations(
        "Habits are cue driven (Wood & Neal, 2007).",
        retrieved_sources={"lally-2010-automaticity"},
        corpus_index=corpus_index,
    )

    assert set(result["uncited_retrievals"]) == {"Wood 2007", "Neal 2007"}
    assert result["no_fabricated_sources"] is True
    assert result["citations_grounded"] is False


def test_a_retrieved_citation_passes(corpus_index):
    result = ev.grade_citations(
        "Missing one day did not derail progress (Lally et al., 2010).",
        retrieved_sources={"lally-2010-missed-days"},
        corpus_index=corpus_index,
    )

    assert result["citations_grounded"] is True
    assert result["fabricated_sources"] == []


def test_an_answer_with_no_citations_is_not_flagged(corpus_index):
    result = ev.grade_citations(
        "You are on a 12 day reading streak. Keep going.",
        retrieved_sources=set(),
        corpus_index=corpus_index,
    )

    assert result["citations"] == []
    assert result["citations_grounded"] is True


def test_forbidden_patterns_are_detected():
    result = ev.grade_forbidden("You missed 4 days last month.", [r"you missed \d+"])

    assert result["no_forbidden_claims"] is False
    assert result["forbidden_hits"] == [r"you missed \d+"]


def test_forbidden_patterns_are_case_insensitive():
    assert ev.grade_forbidden("It takes 21 DAYS.", [r"\b21[ -]days?\b"])["no_forbidden_claims"] is False


def test_clean_answer_passes_forbidden_check():
    result = ev.grade_forbidden(
        "I cannot see your completion history, so I cannot say how many days you missed.",
        [r"you missed \d+"],
    )

    assert result["no_forbidden_claims"] is True


def test_every_case_is_well_formed():
    payload = json.loads(ev.CASES_FILE.read_text(encoding="utf-8"))
    cases = payload["cases"]
    fixtures = payload["habit_fixtures"]

    assert 15 <= len(cases) <= 100
    ids = [case["id"] for case in cases]
    assert len(ids) == len(set(ids))

    for case in cases:
        assert set(case) >= {
            "id", "tags", "question", "fixture",
            "expect_research", "expect_citations", "forbid_patterns",
        }
        assert case["fixture"] in fixtures
        assert case["question"].strip()
        assert case["tags"]


def test_every_forbidden_pattern_compiles():
    payload = json.loads(ev.CASES_FILE.read_text(encoding="utf-8"))
    import re

    for case in payload["cases"]:
        for pattern in case["forbid_patterns"]:
            re.compile(pattern)


def test_the_case_set_covers_every_failure_mode():
    payload = json.loads(ev.CASES_FILE.read_text(encoding="utf-8"))
    tags = {tag for case in payload["cases"] for tag in case["tags"]}

    assert {"grounded", "personal", "unknowable", "trap", "out-of-scope"} <= tags


def test_grade_marks_an_errored_run_without_crashing(corpus_index):
    record = {"error": "timeout", "answer": None, "tool_calls": [], "retrieved_ids": []}
    case = {"expect_research": True, "expect_citations": True, "forbid_patterns": []}

    assert ev.grade(record, case, corpus_index)["status"] == "error"


def test_all_passed_is_false_when_a_metric_fails():
    rows = [{"status": "ok", "research_when_expected": False}]

    assert ev.all_passed(rows) is False


def test_all_passed_ignores_missing_optional_metrics():
    rows = [{"status": "ok", "research_when_expected": True}]

    assert ev.all_passed(rows) is True


def test_a_grouped_parenthetical_pairs_each_author_with_its_own_year():
    found = ev.extract_citations("repetition beats volume (Fogg, 2019; Clear, 2018).")

    assert ("Fogg", "2019") in found
    assert ("Clear", "2018") in found
    assert ("Clear", "2019") not in found
    assert ("Fogg", "2018") not in found


def test_a_grouped_parenthetical_is_not_flagged_as_fabricated(corpus_index):
    result = ev.grade_citations(
        "Keep it small (Fogg, 2019; Clear, 2018).",
        retrieved_sources={"fogg-tiny-habits", "clear-atomic-habits"},
        corpus_index=corpus_index,
    )

    assert result["fabricated_sources"] == []
    assert result["citations_grounded"] is True
