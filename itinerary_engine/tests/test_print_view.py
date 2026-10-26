"""The printed itinerary is a separate document, not the app with bits hidden.

These are structural checks on the page itself. The document is built in the browser, so
what can be verified here is that the pieces it needs are present and that the app chrome
cannot leak into it -- the failure mode being guarded against is someone adding a panel to
the screen and finding it on the customer's printout a month later.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

PAGE = Path(__file__).resolve().parents[1] / "app" / "static" / "index.html"


@pytest.fixture(scope="module")
def page() -> str:
    return PAGE.read_text(encoding="utf-8")


def test_the_print_document_has_its_own_container(page):
    assert 'id="print-doc"' in page
    assert "function renderPrint(plan)" in page


def test_printing_shows_only_that_document(page):
    """Opt-out, not opt-in: everything on screen is hidden and the document is shown."""
    print_block = page[page.index("@media print"):]
    print_block = print_block[:print_block.index("}\n  }") + 4]
    assert ".wrap > *:not(#print-doc)" in print_block, "the app chrome must be hidden wholesale"
    assert "#print-doc { display: block" in print_block


def test_the_page_is_set_up_for_a4(page):
    assert re.search(r"@page\s*{[^}]*size:\s*A4", page), "no A4 page size"
    assert re.search(r"@page\s*{[^}]*margin:", page), "no print margins"


def test_days_do_not_split_across_pages(page):
    assert ".pr-day" in page and "break-inside: avoid" in page
    assert "page-break-inside: avoid" in page, "older engines need the legacy property too"


def test_the_document_carries_what_a_customer_needs(page):
    for piece in ["pr-title", "pr-day-head", "pr-time", "pr-transfer", "pr-terms", "printTerms"]:
        assert piece in page, f"missing {piece}"
    assert ">Included<" in page and ">Not included<" in page


def test_nothing_the_engine_cannot_promise_is_listed_as_included(page):
    """The engine models no hotels, meals or prices, so it must not print them as included."""
    body = page[page.index("function printTerms"):]
    body = body[:body.index("function renderPrint")]
    included = body[body.index("const included"):body.index("const excluded")]
    for promise in ["Accommodation", "Hotel", "hotel", "Meals", "Entry ticket", "Guide"]:
        assert promise not in included, f"'{promise}' is promised but not modelled"
    excluded = body[body.index("const excluded"):]
    for honest in ["Accommodation", "Meals", "Entry tickets"]:
        assert honest in excluded, f"'{honest}' should be stated as not included"


def test_printed_times_are_rounded_consistently(page):
    """Rounding starts down and ends up made consecutive rows overlap on paper."""
    assert "function roundClock" in page
    helper = page[page.index("function roundClock"):]
    helper = helper[:helper.index("function printTime")]
    assert "Math.round" in helper
    assert "Math.ceil" not in helper and "Math.floor((" not in helper


def test_the_print_button_builds_the_document_first(page):
    handler = page[page.index('$("print").addEventListener'):]
    handler = handler[:handler.index("});") + 3]
    assert "renderPrint(lastPlan)" in handler
    assert "window.print()" in handler
    assert "Generate an itinerary first" in handler, "printing nothing should say so"


def test_the_document_is_rebuilt_whenever_the_plan_changes(page):
    """Every customisation re-renders the plan; the printout has to follow it."""
    assert re.search(r"lastPlan = plan;\s*\n\s*renderPrint\(plan\);", page)
