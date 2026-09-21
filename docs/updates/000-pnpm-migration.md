# npm → pnpm migration

The repository no longer tracks `package-lock.json`. The project metadata declares pnpm 10 as the package manager.

## One-time setup

From the repository root:

```bash
corepack enable
corepack prepare pnpm@10 --activate
pnpm install
```

That install step creates the canonical `pnpm-lock.yaml`. Commit that lockfile.

## Verify

```bash
pnpm install --frozen-lockfile
pnpm lint
pnpm build
```

Do not use `npm install` for normal project dependency changes. If dependencies change, use `pnpm add`, `pnpm add -D`, `pnpm remove`, or `pnpm update` so `pnpm-lock.yaml` remains authoritative.
