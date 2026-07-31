"""SKILL.md must match the competition template structurally.

A format error makes the submission invalid, and no amount of quantitative work
survives that. The competition rules single out a missing risk notice as an
explicit invalidity condition, so that gets its own assertions.
"""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SKILL = ROOT / "SKILL.md"

EXPECTED_SECTIONS = [
    "Skill Name",
    "Strategy Type",
    "Applicable Market",
    "Core Logic",
    None,  # section 5 is the template's "Why <this signal>?" slot -- free title
    "Agent Execution Flow",
    "Core Parameters",
    "Standard Output Format",
    "Risk Notice",
    "Submission Checklist",
    "Public GitHub Link",
    "Disclaimer",
]


@pytest.fixture(scope="module")
def text():
    assert SKILL.exists(), "SKILL.md is the submission; it must exist"
    return SKILL.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def headings(text):
    return re.findall(r"^## (\d+)\. (.+)$", text, flags=re.MULTILINE)


def flat(s: str) -> str:
    """Collapse whitespace so prose assertions survive line wrapping."""
    return re.sub(r"\s+", " ", s).lower()


# --------------------------------------------------------------------------- #
# Frontmatter
# --------------------------------------------------------------------------- #


def test_has_yaml_frontmatter_with_exactly_the_template_fields(text):
    assert text.startswith("---\n")
    end = text.index("\n---\n", 4)
    front = text[4:end]
    keys = re.findall(r"^([a-z_]+):", front, flags=re.MULTILINE)
    assert keys == ["name", "description"], f"unexpected frontmatter keys: {keys}"


def test_name_is_kebab_case_and_matches_the_repo(text):
    name = re.search(r"^name:\s*(\S+)$", text, flags=re.MULTILINE).group(1)
    assert re.fullmatch(r"[a-z0-9]+(-[a-z0-9]+)*", name), f"{name} is not kebab-case"
    assert name == "fee-floored-adaptive-grid"


def test_description_is_a_folded_block_scalar(text):
    assert re.search(r"^description:\s*>\s*$", text, flags=re.MULTILINE), (
        "the template uses a folded block scalar (`description: >`)"
    )


def test_description_mentions_the_challenge(text):
    end = text.index("\n---\n", 4)
    assert "CWC AI Trading Skill Challenge" in text[4:end]


# --------------------------------------------------------------------------- #
# Structure
# --------------------------------------------------------------------------- #


def test_has_exactly_twelve_numbered_sections_in_order(headings):
    assert len(headings) == 12, f"expected 12 sections, found {len(headings)}"
    assert [int(n) for n, _ in headings] == list(range(1, 13))


def test_section_titles_match_the_template(headings):
    for (_num, title), expected in zip(headings, EXPECTED_SECTIONS):
        if expected is not None:
            assert title.strip() == expected, f"expected '{expected}', found '{title}'"


def test_has_a_single_h1_followed_by_a_blockquote(text):
    body = text[text.index("\n---\n", 4) + 5 :]
    h1 = re.findall(r"^# (.+)$", body, flags=re.MULTILINE)
    assert len(h1) == 1, f"expected exactly one H1, found {h1}"
    after = body[body.index(f"# {h1[0]}") + len(h1[0]) + 2 :].lstrip("\n")
    assert after.startswith(">"), "the template puts a blockquote directly under the H1"


# --------------------------------------------------------------------------- #
# Required blocks
# --------------------------------------------------------------------------- #


def test_core_logic_agent_flow_and_output_are_text_fences(text):
    """The template uses ```text fences, never a programming language."""
    for section in ("## 4. Core Logic", "## 6. Agent Execution Flow", "## 8. Standard Output Format"):
        start = text.index(section)
        end = text.index("\n## ", start + 1)
        assert "```text" in text[start:end], f"{section} must contain a ```text block"


def test_core_logic_contains_numbered_rules_and_a_risk_overlay(text):
    start = text.index("## 4. Core Logic")
    block = text[start : text.index("\n## ", start + 1)]
    for token in ("Rule 1", "Rule 2", "Rule 3", "Risk Overlay"):
        assert token in block, f"Core Logic is missing '{token}'"
    assert "->" in block or "→" in block


def test_risk_overlay_states_hard_numeric_caps(text):
    start = text.index("Risk Overlay")
    block = text[start : start + 1600]
    assert len(re.findall(r"\d+(\.\d+)?%", block)) >= 4, "the overlay needs numeric caps"


def test_applicable_market_states_both_best_and_least_suited(text):
    start = text.index("## 3. Applicable Market")
    block = text[start : text.index("\n## ", start + 1)]
    assert "Best suited for" in block
    assert "Less suited for" in block


def test_parameter_table_is_a_four_column_markdown_table(text):
    start = text.index("## 7. Core Parameters")
    block = text[start : text.index("\n## ", start + 1)]
    rows = [ln for ln in block.splitlines() if ln.strip().startswith("|")]
    assert len(rows) >= 8, "parameter table is too thin"
    assert rows[0].count("|") == 5, "expected 4 columns"
    assert "`" in block, "parameter names should be in backticks"


# --------------------------------------------------------------------------- #
# Risk notice -- an explicit invalidity condition in the competition rules
# --------------------------------------------------------------------------- #


def test_risk_notice_exists_and_is_substantial(text):
    start = text.index("## 9. Risk Notice")
    block = text[start : text.index("\n## ", start + 1)]
    assert len(block.split()) > 300, "risk notice is too thin to be meaningful"


def test_risk_notice_covers_all_three_required_areas(text):
    start = text.index("## 9. Risk Notice")
    block = flat(text[start : text.index("\n## ", start + 1)])
    assert "applicable conditions" in block
    assert "invalidation conditions" in block
    assert "maximum risk assumptions" in block


def test_risk_notice_quantifies_the_worst_case(text):
    start = text.index("## 9. Risk Notice")
    block = text[start : text.index("\n## ", start + 1)]
    assert re.search(r"\d+\.\d+%", block), "max risk must be quantified, not described"
    assert "total loss" in flat(block)


def test_risk_notice_states_the_educational_disclaimer(text):
    start = text.index("## 9. Risk Notice")
    block = flat(text[start : text.index("\n## ", start + 1)])
    assert "constitute investment advice" in block
    assert "educational" in block


def test_risk_notice_states_the_data_quality_halt(text):
    start = text.index("## 9. Risk Notice")
    block = flat(text[start : text.index("\n## ", start + 1)])
    assert "delayed" in block and "stop generating trading actions" in block


# --------------------------------------------------------------------------- #
# Honesty guards -- claims we committed to never making
# --------------------------------------------------------------------------- #


FORBIDDEN = [
    r"\brisk[- ]free\b",
    r"\bguaranteed\b",
    r"\bno loss\b",
    r"\bmarket[- ]neutral\b(?!,)",
    r"\bexpected (annual|monthly) return\b",
    r"\bwill (earn|return|profit)\b",
    r"\bpassive income\b",
]


@pytest.mark.parametrize("pattern", FORBIDDEN)
def test_no_forbidden_marketing_claims(text, pattern):
    for match in re.finditer(pattern, text, flags=re.IGNORECASE):
        context = text[max(0, match.start() - 90) : match.end() + 90].lower()
        # Allowed only when explicitly disclaimed, e.g. "not risk-free".
        assert re.search(r"\b(not|never|no|nor)\b[^.]{0,40}$", context[: 90 + 4]), (
            f"unqualified forbidden claim {match.group(0)!r} near: {context.strip()}"
        )


def test_states_it_is_not_a_live_trading_record(text):
    assert "never been run against a real" in flat(text)


def test_does_not_claim_a_cwc_listing(text):
    lowered = flat(text)
    for phrase in ("cwc will list", "cwc is listed", "cwc listing date", "when cwc launches"):
        assert phrase not in lowered


def test_readme_carries_the_same_honesty_guards():
    readme = flat((ROOT / "README.md").read_text(encoding="utf-8"))
    assert "never been run against a real" in readme
    assert "constitute investment advice" in readme
    assert "if you believe btc is going up, hold btc" in readme


def test_github_link_is_present_and_public_looking(text):
    start = text.index("## 11. Public GitHub Link")
    block = text[start : text.index("\n## ", start + 1)]
    match = re.search(r"https://github\.com/([\w.-]+)/([\w.-]+)", block)
    assert match, "a concrete public GitHub URL is required"
    assert "<your-account>" not in block, "template placeholder was left in"
    assert match.group(2) == "fee-floored-adaptive-grid"


def test_reported_numbers_agree_with_generated_results(text):
    """Headline figures cited in SKILL.md must match results/RESULTS.md after rounding.

    Substring matching is not enough here: SKILL.md rounds to one decimal
    ("41.9%") while the generated report carries two ("41.86%"), so the check has
    to parse both and compare numerically.
    """
    results = ROOT / "results" / "RESULTS.md"
    if not results.exists():
        pytest.skip("results/RESULTS.md not generated yet — run `python run_backtest.py --all`")
    generated = results.read_text(encoding="utf-8")

    row = re.search(r"\|\s*`full-2019-2026`\s*\|(.+)$", generated, flags=re.MULTILINE)
    assert row, "full-run row missing from RESULTS.md"
    cells = [c.strip().strip("*` ") for c in row.group(1).split("|")]
    nums = [float(c.rstrip("%").replace("+", "").replace(",", "")) for c in cells[:5]]
    full_return, full_dd, bh_return = nums[0], nums[1], nums[2]

    skill = flat(text)
    for label, value in (("return", full_return), ("drawdown", full_dd), ("buy-and-hold", bh_return)):
        rounded = f"{abs(value):,.1f}"
        assert rounded in skill.replace("−", "-"), (
            f"SKILL.md does not cite the generated full-run {label} ({rounded})"
        )


def test_readme_headline_table_matches_generated_results():
    """The README's results table is transcribed by hand; pin it to the artifact."""
    results = ROOT / "results" / "RESULTS.md"
    if not results.exists():
        pytest.skip("results/RESULTS.md not generated yet")
    generated = results.read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    for window in ("bear-2022", "range-2023", "bull-2020Q4", "flat-2024-26",
                   "year-2025", "chop-2019H2", "full-2019-2026"):
        pattern = rf"\|\s*\**`{re.escape(window)}`\**\s*\|(.+)$"
        gen_row = re.search(pattern, generated, flags=re.MULTILINE)
        rd_row = re.search(pattern, readme, flags=re.MULTILINE)
        assert gen_row and rd_row, f"{window} row missing"

        def first_two(match):
            cells = [c.strip().strip("*` ") for c in match.group(1).split("|")][:2]
            return [round(float(c.rstrip("%").replace("+", "").replace("−", "-")), 2) for c in cells]

        assert first_two(gen_row) == first_two(rd_row), (
            f"README row for {window} disagrees with results/RESULTS.md"
        )
