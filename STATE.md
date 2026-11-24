# ETIA-JAX state — 2026-11-24

## Doing
Increment 9 (OCT and CausalLearner) is done and NOT committed yet: ask the user to commit and push. Increment 10 (CRV) is next. Plan and status: `docs/increments.md`; deck: `docs/deck/` (plateview).

## Done (increment 9)
- Files: `etia_jax/oct/{configs,algorithms,oct,learner}.py`, `etia_jax/graphs/markov.py`, `etia_jax/search/tetrad_members.py`, `etia_jax/learn/forest.py` (JAX random forest, also in the AFS instead of the MLP); `tests/test_oct.py`; `benchmarks/oct_timing.py`; `demo/increment9/` (plate `817e9478b8d8`); fixtures `reference/fixtures/{oct_ref.npz,members_ref.npz,oct_configs_ref.json}` (`make_fixtures.py oct_configs | oct | oct1 | members`).
- User decisions: port Tetrad's random MAG/DAG (seeded); `etia_n_jobs` flag, default copies ETIA's joblib workers (lost state); random forest in JAX.
- Results: ETIA's OCT is not reproducible run to run (Java random MAG), so tests compare each step: 120/120 fold PAGs, 8813/8813 step outputs, 24/24 choices; ported random graphs agree with Tetrad's draws. OCT 6–129× faster (GPU, 10 nodes), 1–14× on larger/mixed, about 1× on discrete (Python FCI). AFS full grid with the forest: 4/4 same as ETIA, 0.25–0.7 s. Suite: 718 passed + 1 xfailed.
- Findings: random MAG with tail-tail edges dropped by ETIA's markov_boundary (also FGES); joblib workers lose mb_size/p_values; final graph on last fold rows; two CausalLearner methods crash; `default_conf` mutated globally; Tetrad FCI not thread-safe with chisquare/gsquare (ETIA OCT on disc10_s0 crashed 4/4); FisherZ divides by zero on a 1-processor JVM.

## Next (increment 10: CRV)
1. Read ETIA's `CRV/` (queries, adjustment sets, bootstrap confidence, visualization).
2. Per increment: tests, GPU/CPU timing vs ETIA, plate with STE "How to read" texts, README section, `docs/increments.md`, deck, then ask the user to commit and push.

## Open
- Workflow rules: exact Tetrad/tigramite/MXM match in the compat modes (`tetrad_compat`, `etia_compat`); textbook default checked by oracles; ETIA venv `~/PROJECTS/ETIA/.venv` (JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64); R references need `R_LIBS=~/R/library` (MXM 1.5.8); commits use `-c user.name="Kostas Tsampourakis" -c user.email=33462414+kostastsa@users.noreply.github.com`; commit and push only when the user asks.
- Do not install R packages myself (the permission check blocks it): ask the user.
- Run ETIA's cdt SAM in a new process (cdt forks and hangs after CUDA). The ETIA OCT fixtures need a JVM with 2 processors (FisherZ), and the discrete case a lock around each test (`oct1`).
- Known limits: FGES/SVAR-GFCI and the FCI family on discrete data are Python searches (slow); GPDC/CMIsymb p-values not reproducible; NOTEARS faster on CPU; the forest needs the GPU (CPU 17–117 s per OCT run); OCT has no time-series algorithms; AFS has no StratifiedKFold.
