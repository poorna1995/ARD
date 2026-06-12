# Archive — superseded code (move only, never delete)

When refactoring, move replaced files here with a manifest entry.

## Layout

```
archive/
├── README.md                 ← this file
└── refactor_2026/
    ├── MANIFEST.md           ← old_path → new_path, date, restore steps
    └── …                     ← copies of monoliths / deprecated modules
```

## Rules

1. **Never delete** — git history + archive are both safety nets.
2. Each batch updates `refactor_YYYY/MANIFEST.md`.
3. Active tree may re-export from new locations until callers migrate.
4. **`archive/` is git-tracked** (unlike gitignored `old/` local legacy).

## Restore

```bash
# Example: restore monolith router before a split
cp archive/refactor_2026/routing/router.py routing/router.py
uv run python -m routing verify
```

See `docs/REFACTORING_PLAYBOOK.md` for phase gates.
