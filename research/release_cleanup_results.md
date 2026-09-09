# H087 - Retired rational branch; compact release verification

Native rational BlockShuffle is retired after repeated memory-gate failures.
The active code now has two model folders, five variants and eight recipes.
Full/narrow dense controls and plain BlockShuffle remain comparison tools.
No newly proposed architecture is promoted by this cleanup.

Before retirement,72 source/config/document files were archived and independently
read back. The [source manifest](archive/h086_manifest.json) records every byte
hash. The rational model, its dedicated test suite, recipe and memory driver
remain readable in [the retired tree](archive/h087_retired/).

The active factory rejects the retired variant. Diagnostic collection no longer
imports it; general shape metadata and activation gradient norms remain available
for future candidates. Scalar slope assumptions were removed from the general
diagnostic path so a future coupled activation is not misreported as elementwise.
Packed/fused execution still rejects unsupported custom FFNs before partial
conversion. Generic allocator-accounting tests are retained.

Validation after the code changes: **97 active tests pass in50.42 seconds**,
including CUDA checks; Ruff passes for maintained source, tests, scripts and the
entry point. The reduced count reflects retired candidate tests and registry
entries, not skipped failures. H086's separate backend failure remains recorded.

Git excludes local datasets, checkpoints, raw tensor observations, compiler
caches, allocator pickles, repeated run source ZIPs and dense sampled-curve grids.
Reports, endpoint metrics, protocols, final receipts, reproducible source,
figures and essential source snapshots are retained. All local raw evidence
remains on disk. The [artifact policy](ARTIFACTS.md) explains what a small clone
can and cannot reproduce and links the full local content-hash inventory.

The attributes file disables automatic newline conversion because older frozen
manifests hash exact source bytes. Historical workers can legitimately reject
today's changed registry: use the recorded source snapshot instead of weakening
their assertions. This release does not recertify old measurements against new
source or establish a research breakthrough.

The next distinct mechanism is [coupled feature-pair geometry](research_direction_2026_09_10.md).
Its stability calculation is a hypothesis to qualify; learning performance is
unmeasured. The original scientific goal remains active and unmet.

The local inventory covers10,227 files totaling45,724,534,435 bytes and took
204.30 seconds. Hashing copies no tensors into Git. Initial index selection is
3,074 files totaling48,324,579 bytes, below the48MiB public-tree review limit.
A first whole-index whitespace check treated preserved historical CRLF endings
as trailing whitespace. The attributes now identify CR as the line terminator;
maintained-code whitespace is checked separately from frozen historical text.

A clean export of staged files passes all13 retained-workflow tests in14.25 seconds,
using isolated Python imports from the export and no ignored data/checkpoints.
The [release receipt](evidence/release_v1.json) pins maintained source hashes and
validation. The known corrupt H079 result payload stays local; its failure report,
content hash and valid protocol remain available.
