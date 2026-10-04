# Part C — Design Write-Up

## Evaluation before shipping

I would evaluate the system in layers so a good final answer cannot hide a weak
classifier, retrieval step, or eligibility check. I would use a fixed, human-labeled
set of questions covering every document, paraphrases, questions that need more than
one document, both interpretations of the ambiguous rebate-adjustment document,
unsupported questions, unclear questions, and eligibility cases with complete,
missing, and failing facts. The repository already has a golden-query fixture for
these scenarios; its current test checks that the fixture covers the documents and
outcomes, but does not measure live model quality.

Before release, I would run that set through the real classifier, embedding model,
Qdrant index, and answer model. I would report classifier category accuracy and
confusion by category, retrieval recall at the four returned chunks, and whether
each answer is supported by its cited sections. I would manually review answers
against the source text because automated similarity alone cannot establish that a
policy answer is correct. I would separately verify that missing facts produce no
eligibility claim, every failed eligibility condition produces an ineligible
result, and an ineligible result never reaches answer composition. My release gates
would be zero eligibility hard-gate failures, correct outcomes for every fixed
eligibility case, and agreed minimums for classification and retrieval before
shipping. I would record the model ids and thresholds with the evaluation so a
model or prompt change can be compared against the same baseline.

After release, I would monitor the privacy-safe request logs already emitted by the
service: outcomes, model ids, category and confidence, selected path, retrieval
scores and chunk ids, filter widening, retries, token use or reported cost, and
latency. I would watch for rising errors, retries, latency, cost, low retrieval
scores, or frequent widened searches, then rerun the offline set after any model,
prompt, corpus, or threshold change. The logs intentionally omit raw questions and
eligibility facts, so reviewing real answer quality would require a separately
approved, privacy-safe sampling process rather than logging member content by
default.

## What changes at higher volume

At ten times the document volume, I expect indexing to become the first pressure
point. A corpus change currently causes all sections to be embedded locally and
the Qdrant collection to be rebuilt. I would measure startup and reindex duration,
then batch the embedding work and index only changed documents. I would build a new
collection and switch to it only after it is complete, so a failed rebuild does not
leave the service without its prior index. I would also review section sizes and
retrieval recall before changing the simple one-section-per-chunk rule. Ten times
the current corpus may still be small for Qdrant; I would benchmark before adding
more infrastructure.

At one hundred times the query volume, synchronous model and embedding calls are
more likely to limit throughput and increase cost. I would load-test first, then
set provider timeouts and concurrency limits, reuse model and database connections,
and add API workers behind a load balancer as measurements require. More workers
would also expose a state issue: pending eligibility sessions currently live in
one process. To keep the prototype's session data in memory, I would first route an
opaque session consistently to the worker holding it and set a short expiry. If a
shared store became necessary, I would decide its privacy and retention rules before
introducing it. I would add rate limits and cost monitoring before raising
concurrency substantially.

## Adding a document category

I would add the category to the shared taxonomy and the classifier's validated
category schema, then map each new document to one or more searchable categories.
The document loader and Qdrant filter already use category metadata, so their
overall flow would stay the same. I would add labeled questions for the new category
to the golden set, test filtering and cross-category retrieval, and check whether
the confidence threshold still gives acceptable results. The eligibility hard gate,
chat-session behavior, and API contract would not need to change unless the new
category introduces a new workflow.

## Non-obvious trade-off

I kept each Markdown section as one chunk with no overlap. This preserves the local
context around policy statements, lists, and tables, and makes each citation easy
to trace back to a heading. The trade-off is that sections can vary in length: a
large section may be harder to retrieve precisely, while splitting every section
into fixed token windows could separate a rule from its conditions. I chose the
simpler section boundary for this fixed corpus and would change it only if the
offline retrieval evaluation shows that long sections are hurting results.
