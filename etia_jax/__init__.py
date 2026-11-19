"""ETIA-JAX: a pure-JAX port of ETIA (automated causal discovery)."""
import jax

# float64 everywhere: p-values and log-determinants must agree with the
# references (Tetrad, tigramite) to many digits.
jax.config.update("jax_enable_x64", True)
