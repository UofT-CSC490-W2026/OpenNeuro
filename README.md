# OpenNeuro
OpenNeuro is an open-source, locally running AI companion.

## Implementation
The main codebase lives in the repos/OpenNeuro submodule (This is pointing to a commit before the deadline which we submit for our project).

### Prerequisites

- [Bun](https://bun.sh/)
- [uv](https://docs.astral.sh/uv/)
- [Rust](https://rustup.rs/) (for desktop app)

### Development

```sh
cd repos/OpenNeuro
bun install
```

**NVIDIA (default):**
```sh
bun dev
```

**macOS (Apple Silicon):**
```sh
cd backend && uv run --no-group cuda12 python -m src.main &
cd frontend && bun run dev
```

**AMD (ROCm):**
```sh
cd backend && uv sync --no-group cuda12 --group rocm && uv run python -m src.main &
cd frontend && bun run dev
```

### Desktop app (Tauri)

```sh
bun tauri dev
```

### Build

```sh
bun tauri build
```

## Assignments
Assignment PDFs are in their corresponding `./a{1..5}` directories.

## Infrastructure
IaC scripts can be found in the `./infra` directory. These scripts are used for hosting cloud models that serve as an alternative option to running all models locally.
