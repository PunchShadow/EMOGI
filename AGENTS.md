# Repository Guidelines

## Project Structure & Module Organization
Root-level CUDA sources implement each algorithm: `bfs.cu`, `cc.cu`, `sssp.cu`, `sssp_float.cu`, `pagerank.cu`, plus 32-bit variants (`*_32.cu`). Shared helpers live in `helper_emogi.h`, `helper_cuda.h`, `helper_string.h`. The build outputs binaries in the repository root.

## Build, Test, and Development Commands
- `make` or `make all`: build all binaries with `nvcc`.
- `make bfs` (or any target name): build a single binary.
- `make clean`: remove built binaries.
- Update `NVCCFLAGS` in `Makefile` to match your GPU architecture (e.g., `compute_70`/`sm_70`).
- `./bfs -h`: show runtime flags. Example: `./bfs -f com-Friendster.bel -r 0 -t 2 -m 2`.

## Coding Style & Naming Conventions
Use 4-space indentation and K&R-style braces, matching existing `.cu` files. Keep filenames aligned with algorithm variants (`pagerank_32.cu` -> `pagerank_32` binary). Prefer fixed-width integer types (`uint32_t`, `uint64_t`) and keep macro names uppercase.

## Testing Guidelines
There is no automated test suite. Validate changes by running the affected binaries on known datasets and checking outputs or performance against prior runs. Place dataset files anywhere and pass the `.bel` prefix via `-f`.

## Data Format & Configuration
Datasets are CSR in custom binary format: `.bel.col` (row offsets), `.bel.dst` (column indices), `.bel.val` (edge weights for SSSP). Files use 8-byte headers and 8-byte indices; `sssp` consumes `.bel.val`.

## Commit & Pull Request Guidelines
Recent commits use short, lowercase messages like "minor fixes" and "fix readme"; follow that style. PRs should include a concise summary, commands run, GPU architecture changes, and dataset names used for validation.
