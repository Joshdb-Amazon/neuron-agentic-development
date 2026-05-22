# Environment Setup Flow

## Script

**Location:** `<skill-install-dir>/scripts/setup_autoport.sh`

Run with bash directly. Do not redirect or suppress output — always surface logs to the user. Run `--help` for the full interface.

## Rules

1. **Never install without explicit user consent.** If you cannot interact with the user, stop and report that a valid venv is required.
2. **If no venv path was provided, ask where to create it.** Default is `<cwd>/_venv` but let the user provide a custom path.
3. Surface all script output to the user.
4. On failure after install (exit 3), offer alternatives — don't loop.

## Sequence

1. If `pathToVenv` is provided: run `--validate-only --venv <path>`
   - Exit 0 → extract RESOLVED paths, proceed.
   - Exit 2 → go to step 2.
2. Run `--dry-run` (with `--venv <path>` if provided)
   - Exit 0 → extract RESOLVED paths, proceed.
   - Exit 4 → show output to user. Ask consent to install. If no `pathToVenv` was provided, also ask where to create the venv (default: `<cwd>/_venv`, or a reusable global path). Wait for reply.
   - Exit 5 → runtime error. Surface the error to the user. Do NOT offer to reinstall.
   - If user says no → stop.
3. **GATE: Only after explicit yes.** Run with `--venv <user-chosen-path>` (no safety flags).

## Exit codes

- Exit 0 → success. Parse `RESOLVED:NXDI_SRC=`, `RESOLVED:NXD_SRC=`, `RESOLVED:TRANSFORMERS_SRC=` from output.
- Exit 2 → hard failure. Surface error and stop.
- Exit 3 → still failing after install. Offer alternatives, don't loop.
- Exit 4 → (dry-run only) install required. Do NOT proceed without consent.
- Exit 5 → runtime error. Packages are present but imports fail due to system-level issues (GLIBC, drivers). Reinstalling will NOT help. Surface the error to the user.
