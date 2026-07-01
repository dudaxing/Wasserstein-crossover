# AGENTS.md

## Cursor Cloud specific instructions

This is a pure-Python scientific-computing research project (Wasserstein-crossover
EA for L-bracket topology optimization). There are **no servers, databases, or
network services** — everything runs as standalone Python batch scripts under
`experiments/` that import the library in `src/` and write figures/`.npz` to
`results/`. See `README.md` ("Layout & quick start") for the canonical commands.

- **Python env:** dependencies are installed into a project-local venv at `.venv`
  (the VM has Python 3.12; the README targets 3.14, but the core deps
  `numpy`/`scipy`/`matplotlib` + optional CPU `jax` install and run fine on 3.12).
  The update script keeps `.venv` in sync with `requirements.txt`. Run everything
  with `.venv/bin/python ...`.
- **`results/` must exist before running the main EA.** It is gitignored, so it is
  absent on a fresh checkout, and `experiments/run_lbracket.py` writes a partial
  cache into it without creating it first (raises `FileNotFoundError` otherwise).
  Run `mkdir -p results` before the first `run_lbracket.py` invocation.
- **Tests** are plain scripts (no pytest): `.venv/bin/python experiments/test_operator.py`,
  `test_fem.py`, `test_bodyfitted.py`. Each prints `ALL ... TESTS PASSED` and exits 0.
- **Lint:** the repo defines no linter/formatter config. Use
  `.venv/bin/python -m py_compile src/*.py experiments/*.py` as a syntax check.
- **Main run (dev):** `.venv/bin/python experiments/run_lbracket.py` runs the full
  LF-seeding → body-fitted-HF → Wasserstein-crossover EA. Defaults (npop=48,
  tmax=40) take a while; for a quick smoke test use small flags, e.g.
  `--npop 8 --nxo 8 --tmax 3 --n-s1 2 --n-s2 2 --hf-seeds 1 --tag smoke`.
  It caches the LF population and checkpoints in `results/` keyed by a config
  hash; change `--tag` (or delete the cache files) to force a clean rerun.
  Note: small/fast runs show flat improvement by design — meaningful gains only
  appear at paper scale (`--n-s1 4 --n-s2 25 --tmax 100 --npop 100`, much slower).
- **Optional TDA extra** (`requirements-tda.txt`, `experiments/test_topo_selection.py`,
  `src/topo_selection.py`) needs a **separate** Python 3.13/3.12 venv (`torch` +
  `POT` + `gudhi` + `torch-topological`); it is not part of the default env and is
  not needed for the core L-bracket pipeline. See the header of `requirements-tda.txt`.
