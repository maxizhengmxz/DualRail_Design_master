"""Runtime setup helpers for numerical analysis scripts."""

import os

from threadpoolctl import threadpool_limits

DEFAULT_NUM_THREADS = "1"
THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def configure_single_threaded_blas(num_threads: str = DEFAULT_NUM_THREADS) -> None:
    """Set process-wide limits for loaded and subsequently imported backends.

    Environment variables alone cannot reconfigure an initialized BLAS library.
    In particular, importing this module executes the package's ``__init__``,
    which already imports NumPy and SciPy. Apply a runtime limit as well; leave
    it active for subsequent computations and inherited worker processes.
    """
    for env_var in THREAD_ENV_VARS:
        os.environ[env_var] = num_threads
    threadpool_limits(limits=int(num_threads), user_api="blas")


__all__ = [
    "DEFAULT_NUM_THREADS",
    "THREAD_ENV_VARS",
    "configure_single_threaded_blas",
]
