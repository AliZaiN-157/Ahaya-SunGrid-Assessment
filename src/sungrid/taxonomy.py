from typing import Literal

from pydantic import BaseModel, Field


CATEGORIES = (
    "program_policies",
    "incentive_rebate",
    "billing_account",
    "technical_installation",
    "company_updates",
)
Category = Literal[
    "program_policies",
    "incentive_rebate",
    "billing_account",
    "technical_installation",
    "company_updates",
    "non_relevant",
]

DOCUMENT_CATEGORIES: dict[str, tuple[str, ...]] = {
    "01_program_policies": ("program_policies",),
    "02_incentive_rebate_programs": ("incentive_rebate",),
    "03_billing_faqs": ("billing_account",),
    "04_technical_installation_guidance": ("technical_installation",),
    "05_company_updates": ("company_updates",),
    "06_ambiguous_rebate_billing_adjustments": ("incentive_rebate", "billing_account"),
    "07_installer_certification": ("technical_installation",),
    "08_governance_voting": ("program_policies",),
    "09_battery_storage_incentive": ("incentive_rebate",),
    "10_low_income_bill_credit": ("incentive_rebate",),
    "11_net_metering_trueup": ("billing_account",),
    "12_autopay_failure_policy": ("billing_account",),
    "13_battery_storage_installation": ("technical_installation",),
    "14_annual_impact_report": ("company_updates",),
}


class Classification(BaseModel):
    primary_category: Category
    confidence: float = Field(ge=0, le=1)
    eligibility_intent: bool
