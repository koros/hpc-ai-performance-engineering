# Historical source provenance

On 28 October 2026 the maintainer authorized a local history rewrite before public
release. The original repository and evidence are retained privately. The public
rewrite keeps commit dates, messages (except identified private strings), author
names, merge structure, and empty commits. Personal author/committer email metadata
was replaced with a GitHub noreply address. Archived operational notes and local-only
folders were removed from every historical tree. Internal node labels, one personal
source-document path, and internal installation paths were generalized.

## Resolving recorded source commits

[commit-map.txt](commit-map.txt) contains 565 original-to-sanitized commit pairs,
including the source commits on both main and the feature branch and the cleanup
commit. The header is `old new`; each later line contains two full Git commit IDs.
No mapped original commit was dropped. [refs-map.json](refs-map.json) records branch
tips at the rewrite boundary, before this provenance documentation was added.

Recorded `git_commit` values in experiment evidence were deliberately retained:
they describe the original execution, not a rerun. Resolve a full recorded ID with:

```bash
awk -v old=ORIGINAL_COMMIT '$1 == old { print $2 }' docs/provenance/commit-map.txt
git show SANITIZED_COMMIT
```

Abbreviated historical IDs mentioned in runbooks or commit messages can be matched
against the beginning of the `old` column; confirm the prefix is unique. Create
report links using the mapped full commit ID after the sanitized refs are published.
The provenance-documentation commit itself has no original counterpart.

## Fingerprints and evidence checksums

Executable/configuration trees were compared for every mapped commit: source,
scripts, configs, Slurm execution files, platform profiles, schemas, tests, Makefile,
and pyproject.toml are byte-identical to the originals (excluding Markdown and
placeholder files). Source-file and workload/runtime fingerprints retain their original inputs.
Resolved-config identities affected by later portable-path normalization are mapped
in [config-hash-map.json](config-hash-map.json); see the
[transformation record](../reproducibility.md#portable-paths-and-resolved-configuration-identities). A rewritten Git commit ID is a separate identifier.

Historical result files and dashboard/documentation metadata were sanitized.
The 127 affected historical checksum-manifest blob versions were updated to hash
the published result bytes. All 101,250 distinct historical manifest/evidence checks
passed. These manifests verify the sanitized files; they do not assert byte identity
with privately archived originals. Numeric measurements and scientific cohort IDs
were not changed. See [the release transformation](../reproducibility.md#public-metadata-transformation).

## Pre-existing source-reference gaps

Of 120 distinct source commit IDs in current canonical evidence, 116 resolve through
the mapping. Four were already missing from the local object database and reachable
history. Direct fetch attempts from GitHub returned `not our ref` for each. This is
a pre-existing limitation, not a consequence of rewriting history. Do not infer an
exact source snapshot from a neighbouring commit or replace a recorded hash.

| Original source commit | Current evidence records |
| --- | --- |
| `024f881d645ffadead3f5a8a5d4f2b98936e1e51` | 8 |
| `2bfc258caec961bfebc054cf71a3bb4fa1e4dacc` | 75 |
| `c9a35bb883756e34e46030410a5480ca1d71a6e0` | 6 |
| `e79d665f682bfc072611d8da990e5763e6f3a64e` | 1 |

[source-reference-gaps.json](source-reference-gaps.json) lists affected records and
their recorded dirty-worktree flags. If original source copies become available,
archive and audit them before extending provenance. A `git_dirty: true` execution
also means its recorded commit alone cannot establish all uncommitted source changes,
even when the commit is available; retain runtime/config fingerprints and limitations.

[validation.json](validation.json) records the rewrite-boundary audit counts, source-input
comparison, and topology checks.

## Publication boundary

The rewrite is local. Original GitHub refs still point to the original history until
a separately reviewed remote replacement. Publishing new branch tips alone does not
remove old refs, pull-request refs, cached pages, forks, or other clones. Use explicit
reviewed branch/tag refspecs; do not push local application/checkpoint refs or mirror
all refs. Keep the remote private until that replacement and follow-up audit finish.
Collaborators must use a fresh clone or carefully reset to sanitized refs before
contributing; merging old history would reintroduce removed content.

The local `origin` remote is deliberately disconnected after the rewrite so that
an ordinary fetch cannot restore unsanitized tracking refs. The original remote
configuration is preserved in the private backup. Reconnect only as part of the
reviewed publication step. Local main is advanced to the validated cleanup commit.
