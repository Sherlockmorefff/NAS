# Repository working rules

## Environment roles

- This repository is edited and reviewed on a local macOS machine.
- The local Mac does not have an NVIDIA GPU.
- Do not attempt to run CUDA or full GNN training experiments locally.
- GPU integration experiments are run separately on a remote Linux server.
- Local checks should focus on syntax, static checks, CPU-safe unit tests, and code review.

## Implementation requirements

- Inspect the existing implementation and call relationships before editing.
- Implement complete executable code, not pseudocode.
- Do not leave TODO, pass, placeholder functions, or NotImplementedError.
- Preserve unrelated behavior.
- Keep backward compatibility unless the task explicitly requires breaking it.
- Add clear errors instead of silently falling back.
- Use explicit random seeds.
- Keep experiment outputs under results/ and logs/.

## Validation

After code modifications:

- Run git diff --check.
- Run Python syntax compilation on modified Python files.
- Run CPU-safe tests that do not require Cora downloads or a GPU.
- Report exactly which tests ran and which could not run.
- Never claim a test passed unless it was actually executed.

## Git safety

- Do not commit or push unless explicitly instructed.
- Do not modify files under results/ or logs/.
- Do not delete existing experiment artifacts.

## Execution priority and anti-overdefense

- The primary goal is to complete the deliverable explicitly requested by the user. Do not turn optional audits, compatibility layers, migration frameworks, or extra configuration into the main task.
- Complete one primary deliverable at a time using the minimum sufficient implementation. Do not add feature flags, wrappers, registries, state machines, or fallback workflows unless a demonstrated requirement exists.
- Run validation only when you can identify a concrete failure risk and the result would change the next action. Do not repeat validation for unchanged content that has already passed.
- Handle only edge cases that are currently reachable and reproducible. Do not write defensive code for purely theoretical risks.
- When fixing a local issue, prefer a local fix followed by a direct retest. Do not refactor the entire workflow because of a small problem.
- Stop after the deliverable is complete and passes the minimum relevant validation. Mention optional improvements in at most one or two sentences; do not implement them automatically.
- Preserve the following mechanisms as necessary research rigor, but use them only when they materially affect experimental conclusions or run attribution: candidate fingerprints; dataset, split, checkpoint, and source identity; fixed training and evaluation seeds; history, provenance, and resume metadata; formal resource-admission decisions; and user-requested data migration, deletion records, Git validation, or security validation. Do not expand these mechanisms into endless hashes, audits, version directories, or acceptance chains.
