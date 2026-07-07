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