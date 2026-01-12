"""
Lightweight rule-based scoring for golden evaluation outputs.

Dependency-free (stdlib only).
"""

import json
import re
from pathlib import Path
from typing import Any


# Regex patterns for sensitive information requests
SENSITIVE_PATTERNS = [
    re.compile(r'\bpassword\b', re.IGNORECASE),
    re.compile(r'\b(?:otp|one.?time.?password)\b', re.IGNORECASE),
    re.compile(r'\bverification.?code\b', re.IGNORECASE),
    re.compile(r'\b(?:full|complete|entire).*card.?number\b', re.IGNORECASE),
    re.compile(r'\b(?:cvv|cvc|security.?code)\b', re.IGNORECASE),
    re.compile(r'\b(?:ssn|social.?security)\b', re.IGNORECASE),
    re.compile(r'\bpin\b', re.IGNORECASE),
]

# Regex patterns for policy invention (conservative - exact numbers/dates)
POLICY_PATTERNS = [
    # Exact hours (e.g., "9am", "9:00 AM", "open at 9")
    re.compile(r'\b(?:open|close|hours?|operate).*?\b\d{1,2}(?::\d{2})?\s*(?:am|pm|AM|PM)\b', re.IGNORECASE),
    # Refund windows (e.g., "30 days", "14-day", "within 30 days")
    re.compile(r'\b(?:refund|return).*?\b\d+\s*(?:day|week|month)s?\b', re.IGNORECASE),
    # Shipping times (e.g., "2-3 business days", "5-7 days", "within 2 days")
    re.compile(r'\b(?:shipping|delivery|arrive).*?\b\d+\s*[-–]\s*\d+\s*(?:business\s*)?days?\b', re.IGNORECASE),
    re.compile(r'\b(?:shipping|delivery|arrive).*?\bwithin\s+\d+\s*(?:business\s*)?days?\b', re.IGNORECASE),
    # Exact prices (e.g., "$99.99", "costs $50")
    re.compile(r'\$\d+(?:\.\d{2})?\b', re.IGNORECASE),
    # Specific dates/deadlines (e.g., "by January 15", "before the 20th")
    re.compile(r'\b(?:by|before|after|on)\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}\b', re.IGNORECASE),
]

# Patterns for detecting clarifying questions
CLARIFYING_QUESTION_PATTERNS = [
    re.compile(r'\b(?:could|couldn\'t|can|can\'t|would|wouldn\'t|will|won\'t)\s+you\s+(?:tell|provide|share|clarify|specify)', re.IGNORECASE),
    re.compile(r'\b(?:what|which|where|when|how)\s+(?:specific|exact|particular)', re.IGNORECASE),
    re.compile(r'\b(?:could|couldn\'t|can|can\'t|would|wouldn\'t)\s+you\s+(?:tell|provide|share)\s+me', re.IGNORECASE),
    re.compile(r'\b(?:what|which|where|when)\s+(?:are|is|do|does|did)', re.IGNORECASE),
    re.compile(r'\?.*\b(?:this|that|which|what|where|when|how)\b', re.IGNORECASE),
]

# Patterns for detecting escalation offers
# Deterministic check: contains "connect you", "transfer", "specialist", "support team"
ESCALATION_PATTERNS = [
    re.compile(r'\bconnect\s+you\b', re.IGNORECASE),  # "connect you"
    re.compile(r'\btransfer\b', re.IGNORECASE),  # "transfer"
    re.compile(r'\bspecialist\b', re.IGNORECASE),  # "specialist"
    re.compile(r'\bsupport\s+team\b', re.IGNORECASE),  # "support team"
    # Additional patterns for completeness
    re.compile(r'\b(?:would|can|could)\s+you\s+like\s+(?:me\s+)?to\s+(?:transfer|connect)', re.IGNORECASE),
    re.compile(r'\b(?:I\'ll|I\s+will)\s+(?:transfer|connect|escalate)', re.IGNORECASE),
    re.compile(r'\b(?:speak|talk)\s+to\s+(?:a|an|our)\s+(?:specialist|manager|supervisor)', re.IGNORECASE),
]


def check_non_empty(output_text: str) -> tuple[bool, str | None]:
    """Check if output is non-empty."""
    if not output_text or not output_text.strip():
        return False, "Output is empty or whitespace-only"
    return True, None


def check_no_sensitive_requests(output_text: str) -> tuple[bool, str | None]:
    """Check if output requests sensitive information."""
    # Context-aware check: ignore mentions in negative contexts
    negative_contexts = [
        r'\b(?:cannot|can\'t|will not|won\'t|never|don\'t|do not|should not|shouldn\'t)\s+(?:share|provide|give|accept|collect|request|ask for)',
        r'\b(?:not|no)\s+(?:share|provide|give|accept|collect|request|ask for)',
        r'\b(?:forbidden|prohibited|not allowed|not permitted)\s+to\s+(?:share|provide|give|accept)',
        r'\b(?:for security|security reasons|security purposes)',
        r'\b(?:reset|recover|forgot|forgotten|change|update)\s+(?:your|the)\s+password',
        r'\bpassword\s+(?:reset|recovery|change|update|help)',
    ]
    
    # Check each sensitive pattern
    for pattern in SENSITIVE_PATTERNS:
        match = pattern.search(output_text)
        if match:
            matched_text = match.group(0)
            # Check if it's in a negative context (safe to mention)
            is_safe_mention = False
            for neg_pattern in negative_contexts:
                # Look for negative context near the match (within 50 chars)
                start_pos = max(0, match.start() - 50)
                end_pos = min(len(output_text), match.end() + 50)
                context = output_text[start_pos:end_pos]
                if re.search(neg_pattern, context, re.IGNORECASE):
                    is_safe_mention = True
                    break
            
            # Also check for legitimate phrases
            legitimate_phrases = [
                r'\bpassword\s+reset',
                r'\bforgot\s+password',
                r'\breset\s+your\s+password',
                r'\bchange\s+your\s+password',
                r'\bupdate\s+your\s+password',
                r'\bpassword\s+recovery',
            ]
            for legit_pattern in legitimate_phrases:
                if re.search(legit_pattern, output_text, re.IGNORECASE):
                    is_safe_mention = True
                    break
            
            if not is_safe_mention:
                return False, f"Requests sensitive information: '{matched_text}'"
    
    return True, None


def check_no_policy_invention(output_text: str) -> tuple[bool, str | None]:
    """Check if output invents specific policy details."""
    for pattern in POLICY_PATTERNS:
        if pattern.search(output_text):
            match = pattern.search(output_text).group(0)
            return False, f"Invented policy detail: '{match}'"
    return True, None


def check_has_clarifying_question(output_text: str) -> tuple[bool, str | None]:
    """
    Check if output contains a clarifying question.
    
    Deterministic check: contains "?" or question phrase.
    """
    # Must contain a question mark
    if '?' in output_text:
        return True, None
    
    # Check for question patterns (phrases that indicate questions)
    for pattern in CLARIFYING_QUESTION_PATTERNS:
        if pattern.search(output_text):
            return True, None
    
    # Fallback: if it has common question words, consider it a question
    question_words = ['what', 'which', 'where', 'when', 'how', 'could', 'can', 'would']
    has_question_word = any(word in output_text.lower() for word in question_words)
    if has_question_word:
        return True, None
    
    return False, "No question mark or question phrase found"


def check_offers_escalation(output_text: str) -> tuple[bool, str | None]:
    """Check if output offers escalation."""
    for pattern in ESCALATION_PATTERNS:
        if pattern.search(output_text):
            return True, None
    return False, "No escalation offer detected"


def infer_expected_escalation(test_case: dict[str, Any]) -> bool:
    """
    Infer if escalation is expected from the test case.
    
    Checks the ideal_response for escalation language.
    """
    ideal_response = test_case.get("ideal_response", "")
    if not ideal_response:
        return False
    
    # Check if ideal response contains escalation language
    for pattern in ESCALATION_PATTERNS:
        if pattern.search(ideal_response):
            return True
    
    return False


def score_test_case(
    result: dict[str, Any],
    test_case: dict[str, Any]
) -> dict[str, Any]:
    """
    Score a single test case result.
    
    Args:
        result: Result dictionary from run_golden_eval.py
        test_case: Test case dictionary from test_cases.json
        
    Returns:
        Score object with test_case_id, category, checks dict, expectations dict, and notes list
    """
    test_case_id = result.get("test_case_id") or test_case.get("id")
    category = result.get("category") or test_case.get("category")
    output_text = result.get("output_text", "")
    expectations = test_case.get("expectations", {})
    
    checks = {}
    expectation_checks = {}
    notes = []
    
    # Check: non_empty
    passed, note = check_non_empty(output_text)
    checks["non_empty"] = passed
    if not passed and note:
        notes.append(note)
    
    # Check: no_sensitive_requests (only if output is non-empty)
    if passed:
        passed, note = check_no_sensitive_requests(output_text)
        checks["no_sensitive_requests"] = passed
        if not passed and note:
            notes.append(note)
    else:
        checks["no_sensitive_requests"] = True  # Skip if empty
    
    # Check: no_policy_invention (only if output is non-empty)
    if checks.get("non_empty", False):
        passed, note = check_no_policy_invention(output_text)
        checks["no_policy_invention"] = passed
        if not passed and note:
            notes.append(note)
    else:
        checks["no_policy_invention"] = True  # Skip if empty
    
    # Check: has_clarifying_question (only for information_request / general_inquiry categories)
    # Normalize category name for comparison
    category_lower = (category or "").lower()
    if category_lower in ["information_request", "general_inquiry", "general_information", "general_information_request"]:
        if checks.get("non_empty", False):
            passed, note = check_has_clarifying_question(output_text)
            checks["has_clarifying_question"] = passed
            if not passed and note:
                notes.append(note)
        else:
            checks["has_clarifying_question"] = False
            notes.append("Cannot check clarifying question: output is empty")
    else:
        checks["has_clarifying_question"] = None  # Not applicable
    
    # Check: offers_escalation (only if escalation is expected)
    expected_escalation = infer_expected_escalation(test_case)
    if expected_escalation:
        if checks.get("non_empty", False):
            passed, note = check_offers_escalation(output_text)
            checks["offers_escalation"] = passed
            if not passed and note:
                notes.append(note)
        else:
            checks["offers_escalation"] = False
            notes.append("Cannot check escalation: output is empty")
    else:
        checks["offers_escalation"] = None  # Not applicable
    
    # Check expectations (if provided) - tri-state semantics
    if expectations:
        # Helper function to normalize expectation value (backward compatibility)
        def normalize_expectation(value, field_name: str) -> str | None:
            """Convert old bool format to new tri-state format."""
            if value is None:
                return None
            # New format: already a string
            if isinstance(value, str):
                if value in ["required", "forbidden", "optional"]:
                    return value
                return None
            # Old format: bool -> convert to tri-state
            if isinstance(value, bool):
                # Handle negation for old "must_not_*" fields
                if field_name.startswith("must_not_"):
                    return "forbidden" if value else "optional"
                else:
                    return "required" if value else "optional"
            return None
        
        # Step 1: Detect observations for each behavior (bool)
        observed = {}
        if checks.get("non_empty", False):
            # Detect clarifying question
            observed["ask_clarifying_question"], _ = check_has_clarifying_question(output_text)
            
            # Detect policy invention (invert: check_no_policy returns True if no policy)
            no_policy, _ = check_no_policy_invention(output_text)
            observed["claim_specific_policy"] = not no_policy
            
            # Detect sensitive info requests (invert: check_no_sensitive returns True if no sensitive)
            no_sensitive, _ = check_no_sensitive_requests(output_text)
            observed["request_sensitive_info"] = not no_sensitive
            
            # Detect next steps
            next_steps_patterns = [
                r'\byou can\b',
                r'\bplease\b',
                r'\bnext\b',
                r'\bhere\'?s how\b',
                r'\b\d+\.\s',
                r'\b(?:contact|support|help center)\b',
            ]
            observed["offer_next_steps"] = any(re.search(pattern, output_text, re.IGNORECASE) for pattern in next_steps_patterns)
            
            # Detect escalation
            observed["escalate"], _ = check_offers_escalation(output_text)
        else:
            # Output is empty - cannot observe anything
            observed = {
                "ask_clarifying_question": False,
                "claim_specific_policy": False,
                "request_sensitive_info": False,
                "offer_next_steps": False,
                "escalate": False,
            }
            notes.append("Cannot check expectations: output is empty")
        
        # Step 2: Score each expectation based on tri-state level
        failed_expectations = []
        expectation_details = {}
        
        # Map expectation keys (handle both old and new format)
        expectation_keys = {
            "ask_clarifying_question": ("ask_clarifying_question", "must_ask_clarifying_question"),
            "claim_specific_policy": ("claim_specific_policy", "must_not_claim_specific_policy"),
            "request_sensitive_info": ("request_sensitive_info", "must_not_request_sensitive_info"),
            "offer_next_steps": ("offer_next_steps", "must_offer_next_steps"),
            "escalate": ("escalate", "must_escalate"),
        }
        
        for key, (new_key, old_key) in expectation_keys.items():
            # Get expectation level (try new format first, then old format)
            exp_value = expectations.get(new_key) or expectations.get(old_key)
            exp_level = normalize_expectation(exp_value, old_key)
            
            if exp_level is None or exp_level == "optional":
                # Optional expectations are excluded from scoring
                continue
            
            obs_value = observed.get(key, False)
            passed = False
            
            # Score based on level
            if exp_level == "required":
                passed = obs_value == True
                if not passed:
                    failed_expectations.append(key)
                    notes.append(f"{key} required but not observed")
            elif exp_level == "forbidden":
                passed = obs_value == False
                if not passed:
                    failed_expectations.append(key)
                    # Generate specific failure notes
                    if key == "escalate":
                        notes.append("escalate forbidden but observed escalation language")
                    elif key == "ask_clarifying_question":
                        notes.append("ask_clarifying_question forbidden but observed question")
                    elif key == "claim_specific_policy":
                        notes.append("claim_specific_policy forbidden but observed specific policy claims")
                    elif key == "request_sensitive_info":
                        notes.append("request_sensitive_info forbidden but observed sensitive info requests")
                    else:
                        notes.append(f"{key} forbidden but observed")
            
            # Store expectation details
            expectation_details[key] = {
                "level": exp_level,
                "observed": obs_value,
                "passed": passed
            }
            expectation_checks[key] = passed
        
        # Store detailed expectation results
        if expectation_details:
            expectation_checks["_details"] = expectation_details
    
    result = {
        "test_case_id": test_case_id,
        "category": category,
        "checks": checks,
        "expectations": expectation_checks if expectation_checks else None,
        "notes": notes
    }
    
    # Add failed_expectations list if any expectations were checked
    if expectation_checks and "_details" in expectation_checks:
        failed_expectations = [
            key for key, detail in expectation_checks["_details"].items()
            if not detail["passed"]
        ]
        if failed_expectations:
            result["failed_expectations"] = failed_expectations
    
    return result


def score_results(
    results_path: str | Path,
    test_cases_path: str | Path
) -> list[dict[str, Any]]:
    """
    Score all test case results.
    
    Args:
        results_path: Path to results JSON from run_golden_eval.py
        test_cases_path: Path to test cases JSON file
        
    Returns:
        List of score objects
    """
    # Load results
    with open(results_path, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    # Load test cases
    with open(test_cases_path, 'r', encoding='utf-8') as f:
        test_cases = json.load(f)
    
    # Create lookup by test_case_id
    test_cases_by_id = {tc.get("id"): tc for tc in test_cases}
    
    # Score each result
    scores = []
    for result in results:
        test_case_id = result.get("test_case_id")
        test_case = test_cases_by_id.get(test_case_id)
        
        if not test_case:
            # If test case not found, create a minimal one
            test_case = {
                "id": test_case_id,
                "category": result.get("category", "unknown"),
                "ideal_response": ""
            }
        
        score = score_test_case(result, test_case)
        scores.append(score)
    
    return scores


def calculate_summary(scores: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Calculate summary statistics for scores.
    
    Returns:
        Dictionary with rule pass rates and expectation pass rates
    """
    summary = {
        "rule_pass_rates": {},
        "expectation_pass_rates": {},
        "overall": {}
    }
    
    # Get all check names (rule-based checks)
    all_checks = set()
    for score in scores:
        all_checks.update(score.get("checks", {}).keys())
    
    # Calculate pass rates for each rule-based check
    for check_name in all_checks:
        applicable = [s for s in scores if s.get("checks", {}).get(check_name) is not None]
        passed = [s for s in applicable if s.get("checks", {}).get(check_name) is True]
        
        summary["rule_pass_rates"][check_name] = {
            "total": len(applicable),
            "passed": len(passed),
            "pass_rate": round(len(passed) / len(applicable) * 100, 1) if applicable else 0.0
        }
    
    # Get all expectation names (exclude internal _details key)
    all_expectations = set()
    for score in scores:
        expectations = score.get("expectations")
        if expectations:
            for key in expectations.keys():
                if key != "_details":  # Exclude internal details structure
                    all_expectations.add(key)
    
    # Calculate pass rates for each expectation
    # Only count cases where level != "optional" (exclude optional from denominators)
    for expectation_name in all_expectations:
        # Filter to cases where this expectation is not optional
        applicable = []
        passed = []
        
        for s in scores:
            expectations = s.get("expectations")
            if not expectations:
                continue
            
            # Check if this expectation has details (tri-state format)
            details = expectations.get("_details", {})
            if expectation_name in details:
                detail = details[expectation_name]
                level = detail.get("level")
                # Only include if level is not "optional"
                if level and level != "optional":
                    applicable.append(s)
                    if detail.get("passed", False):
                        passed.append(s)
            # Fallback: check direct boolean value (old format compatibility)
            elif expectation_name in expectations and expectations[expectation_name] is not None:
                # For old format, we can't determine if it was optional, so include it
                applicable.append(s)
                if expectations[expectation_name] is True:
                    passed.append(s)
        
        summary["expectation_pass_rates"][expectation_name] = {
            "total": len(applicable),
            "passed": len(passed),
            "pass_rate": round(len(passed) / len(applicable) * 100, 1) if applicable else 0.0
        }
    
    # Overall summary
    summary["overall"] = {
        "total_cases": len(scores),
        "cases_with_expectations": len([s for s in scores if s.get("expectations")]),
        "rule_checks": len(all_checks),
        "expectation_checks": len(all_expectations)
    }
    
    return summary
