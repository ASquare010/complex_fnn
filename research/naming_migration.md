# Architecture folder names

The numbered placeholder folders were renamed at the user's request:

| Previous folder | Current folder |
|---|---|
| model_name_1 | src/dense_ffn |
| model_name_2 | src/bezier_ffn |
| model_name_3 | src/grouped_ffn |
| model_name_4 | src/shared_ffn |
| model_name_5 | src/affine_shared_ffn |
| model_name_6 | src/cross_group_ffn |
| model_name_7 | src/adaptive_cross_group_ffn |

Active imports, tests, documentation and links use these names. Historical run
metadata and source archives retain the paths that actually existed during the
experiment. They are evidence, so their provenance hashes were not rewritten.
Checkpoints store state dictionaries rather than pickled model classes and do
not require old import aliases. All 15 existing variants retained exactly the
same deterministic parameter and output hashes through the rename. The check
is recorded in results/verification/naming_migration.json.
