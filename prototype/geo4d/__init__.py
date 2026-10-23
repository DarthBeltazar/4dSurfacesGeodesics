"""4D-гиперповерхности и геодезические: прототип этапа 1 (Python + JAX, float64)."""
import jax

# Должно выполниться до создания любых массивов, иначе JAX молча останется во float32.
jax.config.update("jax_enable_x64", True)
