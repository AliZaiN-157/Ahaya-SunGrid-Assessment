# SunGrid Cooperative Copilot

Language for the member-support system that answers questions from SunGrid's
knowledge base and checks rooftop-rebate eligibility.

## Language

**Primary category**:
The single SunGrid knowledge area that best matches a member question and limits
which knowledge-base material is searched. A question may also have related
categories when it needs evidence from more than one area.
_Avoid_: Route, intent

**Related category**:
An additional searchable knowledge area needed to answer another part of the same
question. It expands the retrieval filter without changing the primary category.
_Avoid_: Secondary route, unrelated category

**Eligibility intent**:
A member's request to determine whether a household qualifies for the Rooftop
Rebate Program.
_Avoid_: Incentive category, rebate topic

**Non-relevant question**:
A question that is unrelated to SunGrid or cannot be answered from the available
SunGrid knowledge base.
_Avoid_: Unknown category, unsupported category

**Clarification**:
Additional information requested from a member when their question cannot be
classified confidently.
_Avoid_: Guess, fallback answer

**Retrieval filter**:
A limit to the primary category and any related categories selected for a question.
_Avoid_: Route, agent branch

**Filter widening**:
One repeat search across all knowledge-base categories after a category-filtered
search finds no useful material.
_Avoid_: Retry loop, category change

**Searchable category**:
A SunGrid knowledge area under which a document can be found. One document can
have more than one searchable category while retaining one primary category.
_Avoid_: Secondary route, duplicate document

**Needs more input**:
An eligibility-check outcome stating that one or more required household facts are
missing. It makes no eligibility decision and provides no rebate estimate.
_Avoid_: Partial eligibility, provisional estimate

**Eligibility hard gate**:
The rule that an ineligible household receives no rebate estimate or alternative
eligibility answer, regardless of other favorable facts.
_Avoid_: Recommendation, warning
