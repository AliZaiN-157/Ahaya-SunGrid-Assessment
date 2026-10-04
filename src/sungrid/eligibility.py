import math
import re
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class EligibilityFacts(BaseModel):
    household_zip: str | None = None
    annual_income_usd: float | None = Field(default=None, ge=0)
    system_size_kw: float | None = Field(default=None, gt=0)
    installer_approved: bool | None = None

    @field_validator("household_zip")
    @classmethod
    def valid_zip_shape(cls, value: str | None) -> str | None:
        if value is not None and not re.fullmatch(r"\d{5}", value):
            raise ValueError("ZIP code must contain five digits")
        return value

    def missing_labels(self) -> list[str]:
        labels = {
            "household_zip": "ZIP code",
            "annual_income_usd": "annual household income",
            "system_size_kw": "system size in kW",
            "installer_approved": "whether the installer is SunGrid-approved",
        }
        return [label for name, label in labels.items() if getattr(self, name) is None]


class EligibilityResult(BaseModel):
    eligible: bool
    reason: str
    estimated_rebate_usd: float = Field(ge=0)


def collect_facts(current: EligibilityFacts, message: str) -> EligibilityFacts:
    values = current.model_dump()
    income_match = re.search(
        r"\b(?:annual\s+|household\s+|yearly\s+)?income\s*(?:is|of|:|=)?\s*\$?\s*([\d,]+(?:\.\d+)?)\s*(k|thousand)?\b",
        message,
        re.IGNORECASE,
    )
    if income_match:
        amount = float(income_match.group(1).replace(",", ""))
        if income_match.group(2):
            amount *= 1000
        if math.isfinite(amount):
            values["annual_income_usd"] = amount

    zip_match = re.search(
        r"\b(?:zip(?:\s+code)?|postal\s+code)\s*[:#]?\s*(\d{5})\b",
        message,
        re.IGNORECASE,
    )
    if zip_match:
        values["household_zip"] = zip_match.group(1)
    else:
        for zip_match in re.finditer(r"(?<![\d,])\d{5}(?![\d,])", message):
            if (
                income_match
                and income_match.start() <= zip_match.start() < income_match.end()
            ):
                continue
            values["household_zip"] = zip_match.group(0)
            break

    size_match = re.search(
        r"\b(\d+(?:\.\d+)?)\s*(?:kw|kilowatts?)\b", message, re.IGNORECASE
    )
    if size_match:
        size = float(size_match.group(1))
        if math.isfinite(size) and size > 0:
            values["system_size_kw"] = size

    installer_match = re.search(
        r"\binstaller\b.{0,20}\b(not\s+approved|unapproved|approved)\b|\b(approved|not\s+approved|unapproved)\s+installer\b",
        message,
        re.IGNORECASE,
    )
    if installer_match:
        phrase = (installer_match.group(1) or installer_match.group(2)).lower()
        values["installer_approved"] = phrase == "approved"

    return EligibilityFacts.model_validate(values)


EligibilityOutcome = Literal["needs_more_input", "eligible", "ineligible", "error"]
