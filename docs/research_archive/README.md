# Collaborator research snapshots

These snapshots preserve PR #2 and #3 without changing the live engine.
See [the review and adoption boundaries](../collaborator_pr_review_2026-10-02.md).

- [PR #2 KV compression source and report](pr2_ca8b68c_snapshot.zip)
- [PR #3 two-pass attention / G4 source and reports](pr3_13b57b1_snapshot.zip)
- [Provenance and file hashes](manifest.json)

Each ZIP contains the PR-changed files at their original paths, not a complete
checkout. Original reports are preserved unedited, including historical dates,
plans and acceptance wording. They are historical evidence, not current policy.

Do not extract these ZIPs into the active repository: each contains an old
`specdec/engine/model_self.py` and `docs/data_ledger.md`. They are not installed
packages or current runnable tests. Use a separate checkout of the recorded
source commit for historical reproduction; do not add these paths to PYTHONPATH
or copy their engine over main. GPU results have not been rerun for this import.
