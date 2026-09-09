"""Article boundaries and split leakage checks independent of network access."""

import pytest

from src.core.wikitext_data import articles_from_rows, deduplicate


def test_sections_stay_in_articles_and_every_character_survives():
    rows = [
        "",
        " = First = \n",
        "",
        "Paragraph.\n",
        " = = Section = = \n",
        "More text.\n",
        " = Second = \n",
        "Final paragraph.\n",
    ]
    result = articles_from_rows(rows)
    assert len(result) == 2
    assert "Section" in result[0] and "More text" in result[0]
    assert "".join(result) == "".join(rows)
    assert result[1].startswith(" = Second = ")


@pytest.mark.parametrize("rows", [["unheaded text"], ["", "\n"], [None]])
def test_malformed_sources_are_rejected(rows):
    with pytest.raises(ValueError):
        articles_from_rows(rows)


def test_validation_priority_removes_normalized_duplicates_preserving_text():
    raw = {
        "valid": [" = A = \nabc", "= A = abc"],
        "train": ["= A = abc", " = B = \nxyz", "= B = xyz"],
    }
    data, removed = deduplicate(raw)
    assert data == {"valid": [raw["valid"][0]], "train": [raw["train"][1]]}
    assert [r["article_index"] for r in removed["train"]] == [0, 2]
    assert [r["article_index"] for r in removed["valid"]] == [1]
