---
name: publish-package-on-npm
description: Release a new version of this package (umap-gpu) to npm. Use when asked to publish, release, ship, or cut a new npm version. Publishing happens via a GitHub Release that triggers the Actions OIDC pipeline — NOT via local `npm publish`.
---

# Publish umap-gpu to npm

This package is published **only** through `.github/workflows/publish.yml`, which
triggers on a published **GitHub Release** and uses npm **OIDC trusted publishing**
(no token; `id-token: write`). The release **tag name** drives the version
(`npm version <tag> --no-git-tag-version` then `npm publish --access public`).

Local `npm publish` will NOT work: local npm auth returns 401 and OIDC is CI-only.
Tags are `v`-prefixed (e.g. `v0.3.0`). Releases are cut from `main`.

## Steps

1. **Pick the version.** `npm view umap-gpu version` for the current latest.
   Choose the next semver: **minor** bump for behavior/output changes, **patch**
   for fixes (pre-1.0, so breaking-ish changes still go in minor).

2. **Get the changes onto `main`.**
   - Commit the work.
   - `git push origin main` (merge the feature branch first if needed).

3. **⚠️ Do NOT bump `package.json` to the release version.** It is intentionally
   kept stale — the tag bumps it in CI. If `package.json` already equals the tag,
   the workflow step `npm version <tag>` fails with **"Version not changed"** and
   the publish aborts. Leave `package.json` behind.

4. **Verify green locally** (mirrors the pipeline gate):
   - `npm test` (or `npx vitest run`) → all pass
   - `npm run build` → clean
   - Optional: `npm publish --dry-run` to inspect tarball contents (ignore the
     "Cannot apply latest tag" error — it's just comparing the stale
     `package.json` version to the registry).

5. **Cut the release** (this is what triggers publishing):
   ```bash
   gh release create vX.Y.Z --target main --title "vX.Y.Z" --notes "…release notes…"
   ```

6. **Watch the workflow and confirm it published:**
   ```bash
   gh run list --workflow=publish.yml --limit 1          # get the run id
   gh run watch <run-id> --exit-status                   # wait for success
   npm view umap-gpu dist-tags                           # latest == X.Y.Z
   ```

## If the run fails before the "Publish" step
Nothing was published, so it's safe to redo. Delete the failed release + tag,
fix the cause, push, and recreate:
```bash
gh release delete vX.Y.Z --yes --cleanup-tag
git push origin main                 # after committing the fix
gh release create vX.Y.Z --target main --title "vX.Y.Z" --notes "…"
```

## Notes
- `prepublishOnly` runs `bun test && bun run build`. Bun's native test runner
  skips the WebGPU tests (no Dawn under the vitest setupFile) — that's expected;
  **0 failures** is the gate.
- The workflow currently uses `actions/checkout@v4` / `setup-node@v4` (Node 20),
  which GitHub forces to Node 24 starting **2026-06-16**.
