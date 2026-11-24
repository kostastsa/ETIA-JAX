# ETIA-JAX state — 2026-11-24

## Doing
All 10 increments are done, committed and pushed (increment 10: `e67b08d`). Plan and status: `docs/increments.md`; deck: `docs/deck/` (plate `c7db7e95d21f`).

## Done (increment 10)
- Files: `etia_jax/crv/{queries,dagitty,adjustment,confidence,visualization}.py`; `tests/test_crv.py`; `benchmarks/crv_timing.py`; `demo/increment10/` (plate `c99ab2d0d3f6`); fixtures `reference/fixtures/{adjset_ref.json,adjset_time.json,crv_ref.npz}` (`make_fixtures.py adjset | adjset_time | crv`); `reference/adjset_dagitty.R` (dagitty in R with an inline pcalg2dagitty; pcalg is not installed).
- User decisions: exact port of dagitty 0.3.4 (libdagitty) in etia_compat; bootstrap confidence: real bootstrap samples, ETIA's working functions copied exactly.
- Results: 400/400 adjustment queries as dagitty; 4800/4800 path queries, 400/400 edge tables, 150/150 bootstrap FCI graphs and all weights as ETIA; default oracle 304/304 sets give the true effect (ETIA: 309/328, wrong ones on CPDAGs). Adjustment ~490× faster, bootstrap 5–10× (CPU), path metrics 18–38×. Suite: 738 passed + 1 xfailed.
- Findings: find_adjset cannot run on POSIX (shell=True list) and drops CPDAG undirected edges; pcalg2dagitty drops --o; potentially directed query misses --o/--- (87/392); calculate_confidence cannot run; path consistency skips later edges; consistency table stricter in 29/100 mark pairs; Cytoscape confidence methods read missing columns.

## Earlier
- Increments 1–9 committed and pushed (increment 9: `e6d64db`).

## Open
- Workflow rules: exact Tetrad/tigramite/MXM match in the compat modes (`tetrad_compat`, `etia_compat`); textbook default checked by oracles; ETIA venv `~/PROJECTS/ETIA/.venv` (JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64); R references need `R_LIBS=~/R/library` (MXM 1.5.8); commits use `-c user.name="Kostas Tsampourakis" -c user.email=33462414+kostastsa@users.noreply.github.com`; commit and push only when the user asks.
- Do not install R packages myself (the permission check blocks it): ask the user.
- Run ETIA's cdt SAM in a new process (cdt forks and hangs after CUDA). The ETIA OCT fixtures need a JVM with 2 processors (FisherZ), and the discrete case a lock around each test (`oct1`).
- Known limits: FGES/SVAR-GFCI and the FCI family on discrete data are Python searches (slow); GPDC/CMIsymb p-values not reproducible; NOTEARS faster on CPU; the forest needs the GPU (CPU 17–117 s per OCT run); OCT has no time-series algorithms; AFS has no StratifiedKFold.
