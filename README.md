# SlotGround

Core implementation of **SlotGround: Evidence-Guided Argument State Inference for Memory-Augmented LLM Agents**.

SlotGround infers the arguments of a given tool from the current request, retrieved conversation history, an initial Reader prediction, and the tool schema. It constructs source-aware candidates and selects whether to retain, replace, or omit each argument relative to a default-completed Reader reference.

## Method

The algorithm has three stages:

1. **Parameter-Conditioned Evidence Access.** Construct parameter-specific queries, fuse lexical and dense rankings within each query, and fuse the resulting rankings across parameters. Select a shared evidence context under a candidate cap and token budget.
2. **Source-Aware Candidate State Construction.** Check extracted observations against their cited sources and schema constraints. Combine accepted observations and literal fields with the Reader reference and field omission.
3. **Applicability-Aware Reference-Regularized Inference.** Score candidate records, aggregate records representing the same state by their maximum score, and revise the reference only when the score gain reaches the revision margin.

Source verification establishes textual support for admitted observations. It does not establish that a historical value applies to the current request. The Reader and schema-default candidates are retained as separate candidate sources.

## Repository Contents

| Path | Contents |
| --- | --- |
| `slotground/retrieval.py` | Parameter queries, hierarchical reciprocal rank fusion, and evidence packing |
| `slotground/grounding/evidence_state.py` | Observation contract, source verification, and candidate construction |
| `slotground/grounding/field_candidates.py` | Literal field extraction and normalization |
| `slotground/grounding/compile_state.py` | Schema type and enumeration checks |
| `slotground/inference.py` | Feature extraction, logistic scoring, and reference-relative state selection |
| `slotground/types.py` | Task, evidence, and parameter types |
| `prompts/baselines/memory_fill.v1.jinja` | Reader prompt |
| `prompts/state/record_candidates.v2.jinja` | Observation Extractor prompt |

This repository contains the algorithm core and prompts. It does not contain benchmark files, fitted scorer weights, hosted model clients, external baseline implementations, or experiment logs.

## Requirements

Python 3.11 or later is required. The development environment is named `llm-agent`.

```bash
conda activate llm-agent
python -m pip install numpy pydantic scikit-learn jinja2
```

Run Python from the repository root to import `slotground`.

## Interfaces

`parameter_queries(task, slots)` returns the parameter-specific retrieval queries. `select_evidence(evidence, lexical_rankings, dense_rankings)` accepts one lexical ranking and one dense ranking per query. Rankings contain evidence identifiers; the retrieval backend is supplied by the caller. The defaults are a candidate cap of 10, an estimated evidence budget of 600 tokens, and an RRF constant of 60. Evidence size is estimated as one token per four characters, rounded up for each item.

`ObservationOutput.model_validate(payload)` parses the Extractor response. Each observation identifies a declared field, a JSON-encoded value, its source identifier, and a verbatim quotation. The contract permits at most 48 observations. The prompts use `query`, `tool_schema`, `evidence`, and optional `error_feedback` as template variables.

`construct_candidates(task, evidence, reader, extracted)` returns the verified pool, field candidates, and default-completed reference. `score_candidates(...)` computes candidate correctness scores. These scores are not calibrated probabilities.

`fit_model(xs, ys, weights, mask=(8, 10, 11))` fits a standardized L2-regularized logistic scorer and returns its parameters. `xs` contains candidate feature rows, `ys` contains binary correctness labels, and `weights` contains sample weights. Training data and validation splits are supplied by the caller. The stored feature layout has 20 entries; masking the three relation entries leaves the 17 active features used by the reported selector. `ACTIVE_FEATURES` lists those features.

`infer_arguments(task, evidence, reader, extracted, model, margin=0.3)` runs candidate construction, scoring, and field-wise selection. `model` is the parameter dictionary returned by `fit_model`. The inference function does not read gold labels. Given the candidate scores, selection scans the field candidates without enumerating their Cartesian product.

## Datasets

Raw datasets are not redistributed. Obtain them from the original repositories and follow their respective licenses and usage terms.

| Dataset | Source | Revision used in the study |
| --- | --- | --- |
| Schema-Guided Dialogue (SGD) | [Official repository](https://github.com/google-research-datasets/dstc8-schema-guided-dialogue) | `e852981ae34990f4358979625854259302feaa78` |
| Mem2ActBench | [Official repository](https://github.com/Cantaloupe-M/Mem2ActBench) | `c935c65be9aa19c930b60657c0c4b2b7ea13a34a` |

### SGD-derived tasks

The study constructs tool-argument tasks from annotated SGD service calls. The immediately preceding user utterance is the current request, and earlier turns provide the accessible history. Eligible calls contain all required gold fields, contain no undeclared fields, and have recoverable nondefault values. At least one required field must depend on history and be absent from the current request.

The formal 200-task confirmation cohort is held out from the **official development split**, with no original-dialogue overlap with scorer fitting. It is a filtered history-dependence stress test, not an evaluation on the complete SGD dataset or the official SGD test split.

### Mem2ActBench

The study uses the existing 400-task evaluation split. The 384 tasks with accessible history form the primary memory cohort. The upstream data include conversation histories and question-answer tasks; obtain and prepare them using the instructions in the official repository.

### Fitting and evaluation scope

Scorer fitting uses 207 development tasks: 107 from Mem2ActBench and 100 from SGD. Development and evaluation tasks have no overlapping task identifiers or original conversation sources. Both datasets contribute fitting data, so the cross-dataset results do not establish training on one dataset and zero-shot evaluation on another.

The strongest evidence for learned selection over the tested deterministic rules comes from Mem2ActBench. On the DeepSeek/SGD stress test, Rule-S has higher F1 and Fill-S has slightly higher exact match than SlotGround. The results do not establish consistent superiority over rules or equal-compute superiority over Reader refinement.

## Scope and Limitations

The target tool and its schema are given. The algorithm does not perform tool selection, multi-tool planning, or execution. Requiredness is a scoring feature rather than a hard constraint; the output may omit required fields. Independent field selection also does not enforce cross-field constraints. Deployment therefore requires separate schema validation and an appropriate repair or clarification policy.

The Extractor adds a generation stage and can dominate the online token cost. Source-verified observations provide inspectable candidate records, but the feature set does not fully resolve entity relations, temporal updates, or conflicting histories. This core release is not a complete reproduction package for the external-baseline experiments.
