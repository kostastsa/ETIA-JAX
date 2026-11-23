# ETIA-JAX state — 2026-11-23

## Doing
Increment 9 (OCT and CausalLearner) is next. Increments 1–8 are done and committed. Plan and status: `docs/increments.md`; deck: `docs/deck/` (plateview).

## Done (this session: increments 7 and 8, license, public repository)
- Remote: public GitHub repository https://github.com/kostastsa/ETIA-JAX (`origin`). License GNU GPL v3 or later (`93a7b8c`). The old commits still contain home paths in logs (the user chose to keep the history).
- Inc 7 (`ba4da9b`): NOTEARS and SAM with optax; plate `014ea6c8c2af`.
- Inc 8 files: `etia_jax/afs/{fs,models,afs}.py`; `tests/test_afs.py`; `benchmarks/afs_timing.py`; `demo/increment8/` (plate `89493558771c`); fixtures `reference/fixtures/{fs_ref,afs_ref}.npz` (`make_fixtures.py fs|afs`); `reference/install_mxm.R`.
- Results inc 8: FBED/SES 456/456 calls same as MXM (58 are R errors of ETIA, copied in `etia_compat`); AFS 8/8 runs same winner and columns; FS call ~1000×, AFS run 9–190× faster; 632 tests in the suite.

## Next (increment 9)
1. Read ETIA's OCT (`CausalLearning/CDHPO/OCT/`), the validation protocols (`model_validation_protocols/`), `CausalLearner.py`, and the regressors used by OCT (`CausalLearning/regressors/`).
2. Port OCT and the CausalLearner on top of increments 3–8; decide the exactness (OCT uses random forests and random folds).
3. Known ETIA problem: the k-fold protocol stops on NOTEARS and SAM (2 return values, not 3); NOTEARS ignores the fold rows.
4. Per increment: tests, GPU/CPU timing vs ETIA, plate with STE "How to read" texts, README section, update `docs/increments.md` and the deck, then ask the user to commit and push.

## Open
- Workflow rules: exact Tetrad/tigramite/MXM match in the compat modes (`tetrad_compat`, `etia_compat`); textbook default checked by oracles; ETIA venv `~/PROJECTS/ETIA/.venv` (JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64) for references; R references need `R_LIBS=~/R/library` (MXM 1.5.8); commits use `-c user.name="Kostas Tsampourakis" -c user.email=33462414+kostastsa@users.noreply.github.com`; commit and push only when the user asks.
- Do not install R packages myself (the permission check blocks it): ask the user.
- Run ETIA's cdt SAM in a new process: cdt forks workers and hangs if the parent already used CUDA.
- Known limits: FGES/SVAR-GFCI slower than Tetrad (Python search); GPDC/CMIsymb p-values not reproducible; NOTEARS is slower on the GPU (use the CPU); SAM on the CPU is about as slow as cdt on the GPU from 20 variables; the AFS has no StratifiedKFold yet (categorical targets); the random forest of ETIA is an MLP in ETIA-JAX.
