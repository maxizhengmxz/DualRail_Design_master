"""Compatibility exports; shared policy now lives in core.cz_channel_policy.

No simulation or search is launched. Keep class/function identity for legacy
imports and serialized historical references. New code imports core directly.
"""
from core.cz_channel_policy import *  # noqa: F401,F403
