# Chat-first corpus for the fixed-size encoder-memory models

Prepared-data work only, 2026-10-12. No training, DB implementation or model-size
increase is part of this change. Consumer cap: **23,209,216 learned parameters**,
including future DB heads. Frozen encoder: 9,426,432 additional parameters.

## Prepared result (2026-10-12)

Dataset: `dump/data/chat-memory-v2`.

| Source | Train | Development | Test |
|---|---:|---:|---:|
| Smol-SmolTalk, filtered | 106,706 | 1,120 | 1,003 |
| Everyday conversations | 1,962 | 20 | 22 |
| TinyStories | 98,036 | 1,002 | 962 |
| Generated memory episodes | 39,270 | 348 | 382 |
| **Total** | **245,974** | **2,490** | **2,369** |

Training has 134,649,410 visible tokens, 68,254,786 scored assistant tokens,
356,211 supervised segments and 80,030 multi-turn examples (40,760 public
conversations plus 39,270 generated memory episodes). At effective batch32,
two full epochs require **22,264 updates**, including a 19-segment final batch
in each epoch. They present 136,509,572 scored assistant tokens in total.

The independent audit passed all file hashes, token/mask and EOS checks,
exact split/long-answer separation, segment coverage and memory record/chunk
labels. It re-tokenized 532 dialogues. Twelve CPU tests passed, including
shift/masking/epoch coverage and raw-model lifecycle/resume. A fresh small
preparation run also passed after correcting the synthetic missing-fact shortcut.
These validate preparation, not future model quality or universal decontamination.

Public chat includes 38,630 concise Magpie dialogues, 24,263 rewriting examples,
19,598 summarization examples, 26,235 constrained-instruction examples and103
additional everyday examples. Story-writing examples are about41% of scored
training tokens; conversational tasks about57%, generated memory about2%.
Uniform segment batches expose memory often, but have few answer tokens; the
future trainer must state any task weighting explicitly and use it in both arms.

The exact preprocessing sources, one-off correction, and source cards are copied
into the dataset. The normal preparation command now includes those corrections;
the one-off repair script was removed from active tools. No new research training,
encoder pass, DB operation, checkpoint deletion or parameter increase occurred.
Original, retrieval-specialist, mixed-chat and encoder hashes remain unchanged.

## Sources and conversion

| Source | Purpose | Treatment |
|---|---|---|
| [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) | Simple coherent English and narrative structure | Bounded 100,000-story subset from one pinned original train shard; four generic story-writing prompts, story as assistant reply |
| [Smol-SmolTalk](https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk) | Instruction following, conversation, rewriting, summarization | Scan all four train shards; retain whole dialogues with concise replies; exclude coding, OpenHermes, LongAlign and unrelated components in this recipe |
| [Everyday conversations](https://huggingface.co/datasets/HuggingFaceTB/everyday-conversations-llama3.1-2k) | Greetings and multi-turn everyday interaction | Preserve complete dialogues; process before overlapping copies in Smol-SmolTalk |
| Local memory episodes | Grounded answers, missing facts, future architectural read/write labels | 40,000 generated multi-turn examples with notebook-location facts, unrelated intervening context and delayed questions |

TinyStories is existing GPT-3.5/GPT-4-generated data, not new paid distillation.
SmolTalk includes existing synthetic instruction/conversation data; its selected
Magpie component was generated with Llama-3.1-405B-Instruct. Everyday conversations
is an existing Llama-3.1 dataset. Local memory examples are programmatic templates,
not model-distilled natural conversations. No teacher model or paid API is run.
These are bounded curated subsets, not all examples in the upstream datasets.
No separate specialist roleplay corpus is added in this first recipe; story writing
and conversational instructions should not be described as broad roleplay coverage.

Plain prose can become a valid story-writing task. It cannot acquire factual QA,
summarization or multi-turn supervision merely by adding role labels. We therefore
do not turn arbitrary FineWeb passages into invented answers to unrelated questions.
The old FineWeb corpus remains intact for independent language-regression tests.

Source revisions, download URLs and SHA256 values live in the dataset manifest.
TinyStories uses CDLA-Sharing-1.0; selected newly generated SmolTalk components and
Everyday conversations use Apache-2.0. Preserve source-specific notices; do not
relabel the whole mixture as uniformly Apache-licensed. Dataset cards are retained
under `dump/data/chat-corpus-sources`.

## Format and preprocessing

`tools/download_chat_sources.py` downloads/verifies pinned files, reusing the
already downloaded first chat shard via a hard link. `tools/prepare_chat_corpus.py`
prepares data without loading a model or running an encoder. Existing prepared
directories are immutable to that command. Download and preprocessing are separate
from training and are not repeated automatically at each training start.

The format is `chat-memory-v2`, with one shared tokenizer (unchanged 4096 vocabulary),
textual System/User/Assistant headers and EOS2 after every assistant reply. It does
not add vocabulary entries or embeddings. Tokenization is explicitly per message
header and body, so the future inference path must share that serializer.

Each example stores all accepted turns, provenance, token spans and optional
memory-event metadata. Overlength dialogues are rejected whole, never silently
cut mid-answer. Public chat replies are at most 384 tokens each; a whole dialogue
fits within 2560 tokens. TinyStories can use a longer answer within that total.
Tokens are uint16 and loss masks uint8 in memory-mapped files. JSONL indexes retain
complete text for inspection. Independent dialogues are never packed together.

Only assistant bodies and EOS are CE targets. User/system/header tokens remain
visible in attention but have no direct CE target. Padding validity must therefore
be separate from the loss mask. `src/chat_data/dataset.py` implements that distinction
and the causal one-token shift with the same 512-token rollover as the existing
model. Completed history uses the unchanged frozen encoder on demand. Up to 2048
old tokens become 32 vectors; targets use at most 512 active positions.

Whole dialogues are grouped into train/dev/test, approximately 98/1/1. Grouping
uses the first substantive user message; generic greetings are not treated as the
identity of every dialogue beginning with "Hi". Stories split by content, not the
generic story prompt. Exact normalized dialogue duplicates and cross-split identical
long answers are excluded. Longer question identities from the previous chat
experiment are excluded. There is no guarantee against semantic near-duplicates,
all previous short-prompt exposure, or upstream model contamination.

## Memory data and fair comparison

Each synthetic episode stores three variable notebook locations. In 20% of cases
the requested person's location is absent. The same name pool supplies present
and absent queries: name alone cannot determine whether a record exists. Dev/test
use different names and question templates. Answers are phrases rather than one
of the original sixteen words. This is still a narrow generated task.

Intervening context puts the facts outside the active 512-token block before the
answer. Metadata identifies write records, source token spans, independently encoded
64-token chunk IDs, positive read records/chunks, no-match cases and no-read events.
These labels are **not textual tool calls or model inputs**. Target answers are
never encoded as history for their own prediction.

A session-reset marker specifies a future DB test, but the current data loader does
not execute a reset or DB operation. Both future arms must see the same source facts
and score the same answer tokens. For DB episodes, the controller must learn which
facts to retain/retrieve; the no-DB arm can receive the equivalent history vectors
as an information-access control. Also evaluate autonomous DB retrieval against
oracle retrieval, and separate persistence capability from ordinary CE comparisons.

## Two-epoch contract and next work

Two epochs means **every prepared training segment exactly twice**, reshuffling
without replacement per epoch. Include the smaller final batch; normalize CE by
the actual scored tokens. The manifest reports examples, visible tokens, scored
assistant tokens, segments and `2 * ceil(training_segments / 32)` optimizer updates.
Epochs over paragraphs, conversations and segments are not interchangeable.

Use the same segment order, target masks, batch size, tokenizer and evaluation sets
for both encoder-memory arms. Microbatch regrouping may differ. Any DB heads must
fit inside the fixed parameter cap. Account for retrieval/encoding time and the
frozen encoder separately. Source balancing must be specified by scored tokens as
well as examples; a larger dataset and two epochs do not guarantee usable chat.

**The legacy training commands do not consume this format.** The new
`src/models/token_memory/training.py` entry point now trains the unchanged
encoder-memory model from scratch on these files, with same-folder resume. See
[commands and resume contract](../src/models/token_memory/CHAT_TRAINING.md). The
budgeted DB controller remains a later variant; no long training run was launched.
Training in chat format from scratch still requires learning language; it removes
the separate format stage, not the computational need to learn.

`tools/audit_chat_corpus.py` verifies file hashes, split identity, all target masks,
EOS boundaries, segment coverage, memory causality/labels, and sampled retokenization.
The completed dataset's `manifest.json`, `audit.json` and `samples.json` are the
authoritative counts and preprocessing evidence.
