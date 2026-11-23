# ETIA-JAX state — 2026-11-23

## Doing
Increment 7 (NOTEARS and SAM) is next. Increments 1–6 are done and committed (`f08958a` = increment 6). Plan and status: `docs/increments.md`; deck: `docs/deck/` (plateview).

## Done (this session: increments 4–6)
- Inc 4 FCI family (`6d6264c`), inc 5 scores/FGES/GFCI (`e219726`), deck+docs (`88ef50b`), inc 6 time series (`f08958a`).
- Inc 6 files: `etia_jax/data/svar.py`, `etia_jax/search/{knowledge,svar,pcmci,lpcmci}.py`, knowledge added in `pcalg.py`, `fci.py`, `fges.py`, `gfci.py`; `tests/test_timeseries.py`; `benchmarks/ts_timing.py`; `demo/increment6/` (plate `603633780311`); fixtures `reference/fixtures/ts_*.npz` (`reference/make_ts_data.py`, `make_fixtures.py ts|ts_tigramite`).
- Results inc 6: Tetrad algorithms with tiers 448/448; tigramite 252/255 (3 misses = GPDC/CMIsymb random p-values); oracle 400/400; 139 tests pass (`pytest -m ""`, 63 min).

## Next (increment 7)
1. Read ETIA's NOTEARS (`~/PROJECTS/ETIA/ETIA/CausalLearning/algorithms/causalnex_algorithm/NoTears.py`, causalnex) and SAM (`cdt_algorithms/SAMAlgorithm.py`, cdt) and their parameter grids (`CausalLearning/configurations/causal_configs.py`: notears max_iter/h_tol/threshold; sam lr/dlr/lambda1/lambda2/nh/dnh/train_epochs…).
2. Implement in JAX + optax: NOTEARS (augmented Lagrangian, h(W)=tr(exp(W∘W))−d) and SAM (adversarial generators/discriminator).
3. Random algorithms: compare accuracy (SHD vs true graph) and run time with ETIA on the same data (fixtures from the ETIA venv), not exact outputs.
4. Per increment: tests, GPU/CPU timing vs ETIA, plate with STE "How to read" texts, README section, update `docs/increments.md` and the deck, then ask the user to commit.

## Open
- Workflow rules: exact Tetrad/tigramite match in `tetrad_compat`; textbook default checked by oracles; ETIA venv `~/PROJECTS/ETIA/.venv` (JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64) for references; commits use `-c user.name="Kostas Tsampourakis" -c user.email=33462414+kostastsa@users.noreply.github.com`; commit only when the user asks.
- Known limits: FGES/SVAR-GFCI slower than Tetrad (Python search); GPDC/CMIsymb p-values not reproducible; `lpcmci.py` is GPL-3.0 (derived from tigramite).
- No git remote: nothing is pushed.
