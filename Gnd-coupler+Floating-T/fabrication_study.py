"""Compatibility imports. New optional runs use study.py robustness."""

from studies.policy import campaign_source_signature, failed_reasons, retry_task

if __name__ == "__main__":
    raise SystemExit("Use: python Gnd-coupler+Floating-T/study.py robustness --help")
