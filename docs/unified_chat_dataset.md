# Unified chat corpus v3

This immutable base is preserved. See the audited [v4 tool/reasoning expansion](chat_dataset_v4.md)
for42,210additional dialogues and the current fresh-training command.

This is the single new corpus for the fixed-size encoder-memory baseline and the
future matched database variant. The consumer remains 23,209,216 parameters;
the frozen encoder is an additional 9,426,432. No new research training is launched
by these scripts. The older v2 run is preserved, paused at update 2,168.

## Completed build (2026-10-12)

- 442,401 training / 4,496 development / 4,388 test dialogues.
- Training: 287,065,313 visible tokens, 153,589,862 assistant targets,
  691,938 segments; 86,496 multi-turn dialogues.
- Two epochs, batch32: 43,248 updates; the final batch of each epoch has2segments.
- Across all splits: 433,592 reasoning-off and 17,693 reasoning-on dialogues.
- Target-token shares: general chat/persona/instructions about49.8%, stories9.2%,
  reasoning13.7%, educational continuation15.7%, science7.1%, tables0.7%,
  tools3.1%, memory0.8%. Monitor memory validation separately; this mix is not
  yet evidence that fresh training preserves the earlier retrieval ability.
- Full audit passed: file hashes, all masks/EOS/segment coverage, split separation,
  memory boundaries/record labels, mode consistency and922retokenized dialogues.
- Two independently prepared1,500-example smoke corpora have identical binary,
  dialogue and segment-index bytes. Seventeen targeted CPU tests passed.
- Manifest SHA256: `562bdb936f7c7f11a266416d40c3711c5824a1805d49b3266b095442915c5531`.

The tool subset retains40,000xLAM examples and1,588Hermes examples. The full
corpus contains533observed Tool turns; not every tool example includes an
execution result. The strict adapter excludes6,795entries with unsupported
schema wrappers that mix surrounding instructions with XML. These are not
necessarily bad data. This bounded release covers canonical call formats, not
every published tool annotation. Other rejections and actual source totals are
recorded in the manifest. No claim of universal execution-ready tool use follows.

## Sources and purpose

The recipe uses bounded public subsets, not the complete datasets. The pinned
revision/file list is in `tools/download_unified_sources.py` and
`tools/download_chat_sources.py`. Download receipts contain SHA256 hashes.

| Source | Accepted example ceiling | Purpose |
|---|---:|---|
| Smol-SmolTalk | 110,000 | Short instruction following, conversation, rewriting, summarization |
| Everyday Conversations | 3,000 | Complete everyday exchanges; actual source is smaller |
| TinyStories | 50,000 | Simple coherent language; reduced so stories do not dominate |
| SystemChats via SmolTalk2 | 25,000 | Personas, roleplay instructions, multi-turn consistency |
| Tulu persona instructions via SmolTalk2 | 25,000 | Following varied instructions |
| OpenHermes 2.5 via SmolTalk2 | 75,000 | Broad topics, code and instruction examples |
| Mixture of Thoughts science, no-think via SmolTalk2 | 25,000 | Science questions and direct answers |
| Table-GPT, no-think via SmolTalk2 | 10,000 | Reading structured information |
| Hermes function calls via SmolTalk2 | 10,000 | Tool schemas, calls, observed results and follow-up answers |
| xLAM traces via SmolTalk2 | 40,000 | Selecting tools and arguments |
| Qwen3 SystemChats reasoning via SmolTalk2 | 12,000 | Complete short reasoning traces and answers |
| Qwen3 everyday reasoning via SmolTalk2 | 2,500 | Reasoning in conversation |
| Table-GPT reasoning via SmolTalk2 | 4,000 | Reasoning over tables |
| FineWeb-Edu | 30,000 | Real passage continuation, not fabricated question-answer pairs |
| Local memory episodes | 40,000 attempted | Facts beyond the raw-token window, distractors, missing facts |

Actual retained counts and **assistant-target token shares** are authoritative in
`dump/data/unified-chat-v3/manifest.json`. Caps are applied after filtering; a
source can retain fewer examples. These are free pre-generated teacher datasets;
no paid distillation API or local teacher generation is required. Their answers
are not guaranteed correct merely because a larger model generated them.

Public source cards:
- [SmolTalk2 composition, teachers, and component licenses](https://huggingface.co/datasets/HuggingFaceTB/smoltalk2)
- [Smol-SmolTalk](https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk)
- [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories)
- [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)

FineWeb-Edu is ODC-BY; TinyStories is CDLA-Sharing-1.0. SmolTalk2's newly generated
sets are Apache-2.0; existing components retain their individual upstream licenses.
Retain provenance and cards rather than relabel the entire mixture Apache-2.0.
PIPPA was considered, but its loosely curated character conversations are not
included in this first general-purpose recipe. SystemChats supplies persona and
roleplay examples instead. Long OpenThoughts traces are not added wholesale:
most do not fit this model's current training history budget.

## Reproduction

Use the existing CUDA Python for training. Preprocessing only needs the normal
project data dependencies (PyArrow, NumPy, tokenizers); no tool servers or simulator
environment is required. Dependency versions are recorded. From the repository root:

```powershell
python tools/download_chat_sources.py
python tools/download_unified_sources.py
python tools/prepare_unified_chat.py
python tools/audit_chat_corpus.py --data dump/data/unified-chat-v3
```

The second downloader also fetches and hash-checks the unchanged 4,096-token
tokenizer from the pinned encoder repository. Prepared data is rebuilt directly
from pinned public shards and the deterministic local memory generator. It does
not depend on prior experiment corpora, checkpoints, or machine-local exclusion
lists. Downloaders reuse verified caches; preparation requires a fresh output
directory. A new environment must install the project's data dependencies first;
the scripts do not modify the environment themselves.

`--limit-per-source 100 --output dump/data/my-smoke` is a small pipeline test,
not the full recipe. The manifest records the limit. Train/dev/test partitions
are deterministic; source prompt groups and long identical answers cannot cross
splits. Think/no-think variants of a prompt share the same partition. There is
exact normalized deduplication, but no claim of exhaustive semantic deduplication.

## Template, reasoning and tools

Every example has an explicit `Reasoning: off.` or `Reasoning: on.` system
instruction. Ordinary examples use off. On examples retain supplied complete
`<think>...</think>` text **and the final answer**; empty or incomplete reasoning
does not qualify. We do not invent a trace for examples that lack one.
`src/chat_data/unified_format.py` exposes the same `thinking=True/False` control
for inference prompt construction. It changes an input instruction, not model
parameters, and does not guarantee that an untrained model obeys it.
After training v3, `tools/chat_unified.py --run <run> --thinking on` or `off`
uses that exact serializer. The CLI displays any tool call without executing it.

Roles are System, User, Assistant and Tool. System instructions and complete tool
schemas are preserved. Python-style data literals in published call annotations
are safely parsed and serialized to canonical JSON; no source code is executed.
Calls are checked against supplied names and required argument fields. Tool
observations are visible to attention but **never CE targets**. Assistant calls,
reasoning, final replies and EOS are CE targets. Actual execution correctness,
argument type compliance and successful tool use still require a held-out runtime
evaluation; training on calls does not implement a tool runner or a database.

The 512 active token window and up to 32 historical vectors are unchanged. Whole
dialogues longer than 2,560 tokens are rejected, not silently truncated. Legacy
short-chat sources retain their 384-token per-reply filter; other sources allow
longer complete answers within the total budget. FineWeb documents are converted
to prefix-continuation requests using a real paragraph boundary. This teaches
language in the same training stage; it does not manufacture conversational facts.

## Comparison and claims

Both future arms must use the same manifest, tokenizer, loss masks, segment order,
effective batch, epoch coverage and evaluation splits. Two epochs means every
prepared training segment twice, including the final partial batch. Keep the
original checkpoints and v2 corpus for historical comparisons; v2 CE and v3 CE
are on different data and cannot be compared directly.

SmolTalk2 reports upstream benchmark decontamination, but this mixture is **not
independently certified benchmark-clean**. Before any claim against a 1B model,
freeze named held-out evaluations and check their questions/near duplicates
against every training source. Measure conversation quality, grounded retrieval,
reasoning accuracy, executed tool calls, missing-information behavior and runtime
separately. Aggregate CE can fall because easy stories or memory templates improve
while conversation remains weak; always report per-source validation results.
Do not compare token CE directly to a 1B model using a different tokenizer.
Use identical held-out questions, source information and scoring rules. An
external-memory advantage must not be confused with giving only one model the
facts needed to answer. A focused win is evidence for that task, not every domain.

More high-quality, suitably balanced data may help. It cannot guarantee a 23M
model beats 1B in every use case. Training from scratch in chat format still has
to learn language and general skills; external storage removes neither requirement.
