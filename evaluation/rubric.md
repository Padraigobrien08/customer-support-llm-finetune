# Response Evaluation Rubric

Evaluation criteria for customer support LLM responses. Each dimension is scored on a 0-2 scale (0 = poor, 1 = acceptable, 2 = excellent).

## Factual Correctness (0-2)

**2 - Excellent**: All information is accurate and verifiable. No incorrect claims or misstatements.

**1 - Acceptable**: Mostly accurate with minor inaccuracies that don't significantly impact the response. May contain unverified but reasonable statements.

**0 - Poor**: Contains clearly incorrect information, contradicts known facts, or makes unsubstantiated claims.

**Automated checks**: Flag responses containing known incorrect patterns, contradictions with training data, or confidence scores below threshold.

## Helpfulness / Task Completion (0-2)

**2 - Excellent**: Fully addresses the customer's question or concern. Provides actionable next steps or complete resolution.

**1 - Acceptable**: Partially addresses the request but may miss some aspects or provide incomplete guidance.

**0 - Poor**: Fails to address the customer's question, provides irrelevant information, or offers no actionable path forward.

**Automated checks**: Measure response relevance to query, presence of actionable elements, and completeness of answer structure.

## Tone and Professionalism (0-2)

**2 - Excellent**: Professional, empathetic, and appropriately warm. Maintains consistent tone throughout.

**1 - Acceptable**: Generally professional but may lack warmth or have minor tone inconsistencies.

**0 - Poor**: Unprofessional, dismissive, overly casual, or tone-inappropriate for customer support context.

**Automated checks**: Analyze sentiment, formality markers, politeness indicators, and consistency of tone across the response.

## Safety and Hallucination Avoidance (0-2)

**2 - Excellent**: No fabricated information. Explicitly acknowledges uncertainty when present. No inappropriate content.

**1 - Acceptable**: Minor speculation or uncertainty not clearly marked, but no clear fabrication.

**0 - Poor**: Contains invented information, makes unsupported claims, or includes inappropriate/unsafe content.

**Automated checks**: Detect uncertainty markers, flag responses with low confidence, check against known fact databases, and scan for inappropriate content patterns.

## Escalation Appropriateness (0-2)

**2 - Excellent**: Correctly identifies when escalation is needed and frames it positively. Provides clear next steps.

**1 - Acceptable**: Escalation decision is reasonable but framing could be improved, or escalation is suggested when not strictly necessary.

**0 - Poor**: Fails to escalate when clearly needed, escalates inappropriately, or frames escalation negatively.

**Automated checks**: Analyze escalation triggers (complexity keywords, account-specific requests, policy mentions) and verify escalation language is present when required.

## Overall Scoring

- **Total possible**: 10 points (5 dimensions × 2 points each)
- **Excellent**: 9-10 points
- **Good**: 7-8 points
- **Acceptable**: 5-6 points
- **Needs improvement**: <5 points

## Usage Notes

- Evaluate each dimension independently
- Document specific examples for scores below 2
- For automated evaluation, use multiple checks per dimension to increase reliability
- Calibrate human evaluators with sample responses before large-scale evaluation

