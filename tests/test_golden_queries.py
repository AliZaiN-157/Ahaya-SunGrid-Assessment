import json
from pathlib import Path


def test_golden_queries_cover_fixed_documents_and_required_outcomes():
    fixture_path = Path(__file__).parent / "fixtures" / "golden_queries.json"
    cases = json.loads(fixture_path.read_text(encoding="utf-8"))
    by_id = {case["id"]: case for case in cases}
    covered = {name for case in cases for name in case["expected_documents"]}

    assert len(by_id) == len(cases)
    assert {f"{path.stem}.md" for path in Path("docs").glob("*.md")} <= covered
    assert by_id["doc_06_rebate"]["expected_category"] == "incentive_rebate"
    assert by_id["doc_06_billing"]["expected_category"] == "billing_account"
    assert {
        by_id[key]["expected_outcome"]
        for key in by_id
        if key.startswith("eligibility_")
    } == {
        "eligible",
        "ineligible",
        "needs_more_input",
    }
    assert by_id["cross_battery"]["expected_documents"] == [
        "09_battery_storage_incentive.md",
        "07_installer_certification.md",
    ]
    assert by_id["cross_trueup"]["expected_documents"] == [
        "03_billing_faqs.md",
        "11_net_metering_trueup.md",
    ]
