import os
import unittest
from unittest.mock import patch

import numpy as np
import scipy.linalg
from pathos.pools import ProcessPool
from threadpoolctl import threadpool_info, threadpool_limits

from core.runtime import configure_single_threaded_blas


def _worker_blas_threads(_):
    return [
        info["num_threads"]
        for info in threadpool_info()
        if info["user_api"] == "blas"
    ]


class RuntimeTests(unittest.TestCase):
    def test_helper_limits_already_loaded_numpy_and_scipy_backends(self):
        # Reproduce notebook startup: numerical libraries load before setup.
        np.linalg.eigh(np.eye(3))
        scipy.linalg.eigh(np.eye(3))
        with patch.dict(os.environ), threadpool_limits(limits=2, user_api="blas"):
            for requested in ("1", "2", "1"):
                configure_single_threaded_blas(requested)
                backends = [
                    info for info in threadpool_info() if info["user_api"] == "blas"
                ]
                self.assertTrue(backends, "Expected an active BLAS backend")
                for info in backends:
                    with self.subTest(requested=requested, library=info["prefix"]):
                        self.assertEqual(info["num_threads"], int(requested))

    def test_process_pool_workers_use_configured_limit(self):
        with patch.dict(os.environ), threadpool_limits(limits=2, user_api="blas"):
            configure_single_threaded_blas()
            pool = ProcessPool(nodes=2)
            try:
                results = pool.map(_worker_blas_threads, range(2))
            finally:
                pool.close()
                pool.join()
                pool.clear()
            for counts in results:
                self.assertTrue(counts, "Expected an active BLAS backend in worker")
                self.assertTrue(all(count == 1 for count in counts), counts)


if __name__ == "__main__":
    unittest.main()
