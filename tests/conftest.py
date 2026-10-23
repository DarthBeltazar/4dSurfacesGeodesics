import jax

jax.config.update("jax_enable_x64", True)  # до любых массивов (см. geo4d/__init__.py)

import geo4d  # noqa: E402,F401
