# ETIA-JAX state — 2026-11-23

## Doing
Increment 8 (predictive models and AFS) is next. Increments 1–7 are done and committed. Plan and status: `docs/increments.md`; deck: `docs/deck/` (plateview).

## Done (this session: increment 7, license, public repository)
- Remote: public GitHub repository https://github.com/kostastsa/ETIA-JAX (`origin`). License GNU GPL v3 or later (`LICENSE`, README section; `93a7b8c`). The old commits still contain home paths in logs (the user chose to keep the history).
- Inc 7 files: `etia_jax/search/{notears,sam}.py`; `tests/test_optim.py`; `benchmarks/opt_timing.py`; `demo/increment7/` (plate `014ea6c8c2af`); fixtures `reference/fixtures/{notears_ref,sam_data,sam_ref}.npz` (`reference/make_search_data.py sam`, `make_fixtures.py notears|sam`).
- Results inc 7: NOTEARS same graph as ETIA 40/41 (cont20_s3: another local optimum, xfail); SAM AUROC 0.856 (10 seeds) vs 0.877 (ETIA, 1 run each); NOTEARS CPU 3–210×, SAM GPU 76–306× faster; 167 tests in the suite.

## Next (increment 8)
1. Read ETIA's predictive models (random forest) and AFS (FBED, SES) and their parameter grids.
2. Implement an MLP and a linear model in JAX, and FBED and SES with the CI tests of increment 2.
3. Compare the selected sets with ETIA on the same data, and the run time.
4. Per increment: tests, GPU/CPU timing vs ETIA, plate with STE "How to read" texts, README section, update `docs/increments.md` and the deck, then ask the user to commit and push.

## Open
- Workflow rules: exact Tetrad/tigramite match in `tetrad_compat`; NOTEARS/SAM use `etia_compat`; textbook default checked by oracles; ETIA venv `~/PROJECTS/ETIA/.venv` (JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64) for references; commits use `-c user.name="Kostas Tsampourakis" -c user.email=33462414+kostastsa@users.noreply.github.com`; commit and push only when the user asks.
- Run ETIA's cdt SAM in a new process: cdt forks workers and hangs if the parent already used CUDA.
- Increment 9 must handle: the k-fold protocol of ETIA stops on NOTEARS and SAM (2 return values, not 3); NOTEARS ignores the fold rows.
- Known limits: FGES/SVAR-GFCI slower than Tetrad (Python search); GPDC/CMIsymb p-values not reproducible; NOTEARS is slower on the GPU (use the CPU); SAM on the CPU is about as slow as cdt on the GPU from 20 variables.
