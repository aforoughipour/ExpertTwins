"""Environment: TLS trust and console encoding.

TLS. On networks that terminate TLS at an inspecting proxy and re-sign with a
private CA held in the operating-system trust store, `requests` can fail with
`CERTIFICATE_VERIFY_FAILED` even though the host is reachable. The difference is
trust-anchor selection: `requests` uses `certifi`, a fixed bundle of public
roots, while the administrator-managed private CA lives in the OS trust store.

`verify=False` would hide the error by disabling certificate verification for
every request this process makes. `truststore` instead teaches Python's SSL
module to use the operating-system trust store. Verification stays on; the trust
anchors become the ones the machine is configured to trust.

CONSOLE. The Windows console may use cp1252, while author names and titles can
contain Unicode characters outside that encoding. Printing diagnostics must not
be able to fail a run.

Both settings degrade explicitly: if `truststore` is missing this says so and
returns False, rather than continuing to a failure that looks like an API outage.
"""
from __future__ import annotations

import sys

_INJECTED = False


def use_system_trust_store(*, quiet: bool = False) -> bool:
    """Route TLS verification through the OS trust store. Idempotent."""
    global _INJECTED
    if _INJECTED:
        return True
    try:
        import truststore
    except ImportError:
        if not quiet:
            print("note: `truststore` is not installed. On a network that "
                  "inspects TLS, arXiv and other https APIs will fail with "
                  "CERTIFICATE_VERIFY_FAILED, which looks exactly like the "
                  "service being down.\n      pip install truststore")
        return False
    truststore.inject_into_ssl()
    _INJECTED = True
    return True


def console_utf8() -> None:
    """Make stdout/stderr survive a name with a non-cp1252 character in it."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:                                         # noqa: BLE001
            pass
