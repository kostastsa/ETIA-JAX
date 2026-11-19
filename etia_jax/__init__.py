"""ETIA-JAX: a pure-JAX port of ETIA (automated causal discovery)."""
import os

import jax

# float64 everywhere: p-values and log-determinants must agree with the
# references (Tetrad, tigramite) to many digits.
jax.config.update("jax_enable_x64", True)

# Keep compiled code on disk, so a new process does not compile again.
# ETIA_JAX_CACHE_DIR sets the folder; an empty value turns the cache off.
_cache = os.environ.get("ETIA_JAX_CACHE_DIR", os.path.expanduser("~/.cache/etia_jax/jax"))
if _cache:
    jax.config.update("jax_compilation_cache_dir", _cache)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)
