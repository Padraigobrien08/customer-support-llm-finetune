"""
Expectation levels for test case evaluation.

Defines tri-state expectations: required, forbidden, or optional.
"""

from typing import Literal

ExpectationLevel = Literal["required", "forbidden", "optional"]

# Valid expectation field names
EXPECTATION_FIELDS = [
    "ask_clarifying_question",
    "escalate",
    "offer_next_steps",
    "request_sensitive_info",
    "claim_specific_policy",
]
