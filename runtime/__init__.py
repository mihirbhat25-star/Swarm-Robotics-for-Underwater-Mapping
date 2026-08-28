"""Execution support for GNCA experiments.

Original experiment scripts remain the public entry points. Runtime modules
contain local/cloud data movement and compute details only.
"""

LOCAL_BACKEND = "local"
CLOUD_BACKEND = "cloud"
SUPPORTED_BACKENDS = (LOCAL_BACKEND, CLOUD_BACKEND)
