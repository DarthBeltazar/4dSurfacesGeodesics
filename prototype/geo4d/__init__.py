"""4D-гиперповерхности и геодезические: прототип (Python + JAX).

Точность: float64 по умолчанию; GEO4D_X64=0 — float32 (эталон «прототип-f32» для сравнения с GPU, DD-004 п. 3).
"""
import os

import jax

# Должно выполниться до создания любых массивов, иначе JAX молча останется во float32.
X64 = os.environ.get("GEO4D_X64", "1") != "0"
jax.config.update("jax_enable_x64", X64)
