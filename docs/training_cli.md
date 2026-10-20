# Training commands

The SwiGLU experiment was retired on 2026-10-12. Its source folder and saved
weights were removed; result logs and source archives remain for reference.
Use the existing raw baseline or original token_memory entry points below.

The raw-token baseline uses `src/models/raw_transformer/training.py` with the
same name/resume/steps/microbatch arguments. The hierarchical experiment is
retired; its entry point and weights have been removed. The
[three-model comparison](three_model_comparison.md) is a historical record.

## Step 3: token prediction with encoder memory

Active implementation: `src/models/token_memory`. Previous latent MSE, parallel
decoder and vector-output modes were removed. Exact prior sources/docs/tests are
archived in `dump/superseded-vector-output-source.zip`. Run evidence, weights and
downloaded data remain intact. Old checkpoints cannot resume into this model.
Stop any old training process before starting another worker.

First repack downloaded documents once (no download or vector encoding):

```powershell
& D:\Git\latent-text-compressor\.venv\Scripts\python.exe D:\Git\complex_fnn\src\models\token_memory\prepare_data.py
```

After `Complete`, start fresh:

```powershell
& D:\Git\latent-text-compressor\.venv\Scripts\python.exe D:\Git\complex_fnn\src\models\token_memory\training.py --name token-memory-512-fresh --steps 20000
```

Defaults:23,209,216trainable parameters,8layers,width512,8heads,512live token
positions plus older vectors;batch32,microbatch2,16accumulation passes. Encoder
frozen, loaded without decoder. Direct token CE only. No loss/decoder/sampling
mode flags. Memory grows by8vectors per completed512tokens; no memory cap. Full
history is recomputed each inference step; KV caching is a later optimization.

Each2112-token document gives5segments (four512 and one64). Segments are shuffled
without replacement;320kdocuments yield1.6Msegments.20kupdates at batch32 sample
640ksegments, not two full document passes. Each segment trains every token prefix
causally. Validation uses100evenly spaced development documents (500segments)
every1000updates; use `--valid-documents 1000` for the entire development split.

Progress prints every25updates; checkpoints every250, retained milestones every1000.
Ctrl+C finishes the update, saves and evaluates. Resume into a new name:

```powershell
& D:\Git\latent-text-compressor\.venv\Scripts\python.exe D:\Git\complex_fnn\src\models\token_memory\training.py --name token-memory-512-resume --resume D:\Git\complex_fnn\dump\token-memory-512-fresh
```

`--steps` means total updates; extend a finished checkpoint with a larger total.
Extension continues at the old ending LR. Data/effective-batch changes are refused on resume.
Microbatch may change: total batch32 with microbatch8 uses4 accumulation passes
instead of16. Same examples, token-weighted CE, optimizer and LR schedule are
preserved; floating-point rounding can differ. Stop with Ctrl+C once and wait for
`PAUSED` and the shell prompt before resuming. Do not launch a second worker.

```powershell
& D:\Git\latent-text-compressor\.venv\Scripts\python.exe D:\Git\complex_fnn\src\models\token_memory\training.py --name token-memory-512-micro8 --resume D:\Git\complex_fnn\dump\token-memory-512-fresh --microbatch 8
```

This continues to the saved total20000, not20000 additional updates. Micro8 is
a conservative candidate for the12GB GPU, not a measured speed or fit guarantee.
The original trainer hash has an explicit compatibility migration; all other
source hashes must match. A resume.json records the regrouping and restored step.
`--inspect` prints settings without data preparation or training. Training checks
binary sizes, tokenizer and manifest identity, not full binary hashes each launch.
GPU memory fit has not been benchmarked. No research training started during setup.

## Steps 1 and 2

Their `training.py` entry points retain the shared CE trainer (`--help`). The
shared trainer no longer contains Step3 modes. Historical resume requires the
matching frozen source archive when source hashes have changed.
