"""Checkpoint identity, policy guards, map-only scope and real Windows locks."""

import ctypes
import json
import os
import sqlite3
import tempfile
import threading
import unittest
from pathlib import Path

from readout_workflow.storage import Ledger, atomic_json


class LedgerTests(unittest.TestCase):
    def test_immutable_events_resume_and_reject_robustness(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder)
            ledger = Ledger(p, {"same": "equations"})
            ledger.save("feasibility_normal", (90, 75), {"status": "no_witness_found"})
            with self.assertRaises(RuntimeError):
                ledger.save("feasibility_normal", (90, 75), {"status": "changed"})
            with self.assertRaises(ValueError):
                ledger.save("robustness", (1, 0), {})
            ledger.close()
            resumed = Ledger(p, {"same": "equations"})
            self.assertTrue(resumed.done("feasibility_normal", (90, 75)))
            self.assertEqual(len(resumed.rows("feasibility_normal")), 1)
            resumed.close()
            with self.assertRaises(RuntimeError):
                Ledger(p, {"different": "equations"})

    def test_original_checkpoint_namespace_is_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder)
            db = sqlite3.connect(p / "samples.sqlite")
            db.execute("CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT)")
            old = "a" * 64
            db.execute("INSERT INTO metadata VALUES('policy_hash',?)", (old,))
            db.commit()
            db.close()
            ledger = Ledger(p, {"consolidated": "same physical rules"})
            self.assertEqual(ledger.policy_hash, old)
            self.assertEqual(
                ledger.sample_id("diagnostic", (90, 75)), "a" * 16 + ":diagnostic:90:75"
            )
            ledger.close()


@unittest.skipUnless(os.name == "nt", "Windows sharing regression")
class WindowsLockTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.root = Path(self.folder.name)
        self.kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        self.kernel.CreateFileW.argtypes = [
            ctypes.c_wchar_p,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_void_p,
        ]
        self.kernel.CreateFileW.restype = ctypes.c_void_p
        self.kernel.CloseHandle.argtypes = [ctypes.c_void_p]

    def tearDown(self):
        self.folder.cleanup()

    def lock(self, path):
        handle = self.kernel.CreateFileW(str(path), 0x80000000, 0, None, 3, 0x80, None)
        self.assertNotEqual(handle, ctypes.c_void_p(-1).value)
        return handle

    def test_transient_lock_retries(self):
        target = self.root / "snapshot.json"
        target.write_text("{}")
        handle = self.lock(target)
        timer = threading.Timer(0.25, self.kernel.CloseHandle, args=(handle,))
        timer.start()
        try:
            self.assertTrue(atomic_json(target, {"new": True}))
        finally:
            timer.join()
        self.assertEqual(json.loads(target.read_text()), {"new": True})

    def test_required_snapshot_failure_is_not_silenced(self):
        target = self.root / "snapshot.json"
        target.write_text("{}")
        handle = self.lock(target)
        try:
            with self.assertRaises(PermissionError):
                atomic_json(target, {"new": True}, attempts=1)
        finally:
            self.kernel.CloseHandle(handle)
        self.assertEqual(json.loads(target.read_text()), {})

    def test_locked_status_cannot_abort_a_committed_event(self):
        ledger = Ledger(self.root, {"test": "file lock"})
        p = self.root / "progress.json"
        p.write_text("{}")
        handle = self.lock(p)
        try:
            ledger.save("feasibility_normal", (90, 75), {"status": "test"})
        finally:
            self.kernel.CloseHandle(handle)
            ledger.close()
        db = sqlite3.connect(self.root / "samples.sqlite")
        try:
            self.assertEqual(db.execute("SELECT count(*) FROM events").fetchone()[0], 1)
            self.assertEqual(db.execute("PRAGMA integrity_check").fetchone()[0], "ok")
        finally:
            db.close()


if __name__ == "__main__":
    unittest.main()
