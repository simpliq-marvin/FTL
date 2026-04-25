"""
Shared pytest fixtures for FTL API integration tests.
"""

import socket
import sys
import os
import time

import dns.resolver
import pytest
import requests

# Add test/api to the path so libs/ can be imported
sys.path.insert(0, os.path.dirname(__file__))

FTL_URL = "http://127.0.0.1"

# ---------------------------------------------------------------------------
# Upstream DNSSEC detection helpers
# ---------------------------------------------------------------------------

def _wait_for_pdns(host="127.0.0.1", port=5555, attempts=10, delay=2.0):
    """Block until host:port accepts connections, or attempts are exhausted.

    Returns True if the port opened within the allotted retries, False
    otherwise.
    """
    for attempt in range(attempts):
        try:
            with socket.create_connection((host, port), timeout=2.0):
                return True
        except OSError:
            if attempt < attempts - 1:
                time.sleep(delay)
    return False


def _has_ds_record(resolver, domain):
    """Return True if *domain* currently has a DS record."""
    try:
        resolver.resolve(domain, "DS")
        return True
    except (dns.resolver.NoAnswer, dns.resolver.NXDOMAIN,
            dns.resolver.NoNameservers, dns.exception.Timeout):
        return False


@pytest.fixture(scope="session", autouse=True)
def detect_upstream_dnssec():
    """Detect upstream DNSSEC state and populate query-counter constants.

    Waits for the local pdns_recursor (port 5555) to become available
    with retry/backoff before probing, avoiding false-negatives from a
    slow daemon startup.  Probes DS records on icloud.com and
    apple-dns.net — the two zones crossed by the mask.icloud.com CNAME
    chain — to determine whether dnsmasq will fire extra DNSKEY
    validation queries.

    The module-level expected-counter constants in test_api are updated
    in-place so all assertions there reflect the current upstream
    DNSSEC posture without requiring changes to the individual tests.
    """
    import test_api  # imported here to ensure test_api is fully loaded before modifying its globals

    resolver = dns.resolver.Resolver(configure=False)
    resolver.nameservers = ["127.0.0.1"]
    resolver.port = 5555
    resolver.lifetime = 5

    # Wait for pdns_recursor to accept connections before probing
    if not _wait_for_pdns():
        import warnings
        warnings.warn(
            "pdns_recursor on 127.0.0.1:5555 did not become available; "
            "DNSSEC detection skipped — using non-DNSSEC counter defaults.",
            RuntimeWarning,
            stacklevel=2,
        )
        return

    upstream_dnssec = (
        _has_ds_record(resolver, "icloud.com") and
        _has_ds_record(resolver, "apple-dns.net")
    )

    test_api.TOTAL      = 137 if upstream_dnssec else 135
    test_api.FORWARDED  = 47  if upstream_dnssec else 45
    test_api.DNSKEY     = 9   if upstream_dnssec else 7
    test_api.TOP_DOMAIN = "." if upstream_dnssec else "localhost"


@pytest.fixture(scope="session")
def ftl_url():
    """Base URL for the FTL API."""
    return FTL_URL


@pytest.fixture(scope="session")
def api_session():
    """Shared authenticated requests.Session for the entire test run.

    If no password is set, the session works without authentication.
    If a password is set, logs in with "ABC" and stores the SID in
    the session headers so all subsequent requests are authenticated.
    """
    session = requests.Session()
    session.headers["Accept"] = "application/json"
    try:
        r = session.get(f"{FTL_URL}/api/auth", timeout=5)
        if r.status_code not in (200, 401):
            r.raise_for_status()
    except requests.ConnectionError:
        pytest.skip("FTL is not running at " + FTL_URL)

    data = r.json()
    if not data.get("session", {}).get("valid"):
        # Password is set — login with "ABC"
        r = session.post(f"{FTL_URL}/api/auth",
                         json={"password": "ABC"}, timeout=10)
        sid = r.json().get("session", {}).get("sid")
        if sid:
            session.headers["X-FTL-SID"] = sid
    return session


@pytest.fixture(scope="session")
def openapi():
    """Parsed OpenAPI specifications (session-scoped, parsed once)."""
    from libs.openAPI import openApi
    specs = openApi(base_path="src/api/docs/content/specs/", api_root="/api")
    assert specs.parse("main.yaml"), "Failed to parse OpenAPI specs"
    return specs


@pytest.fixture(scope="session")
def ftl():
    """FTLAPI client with endpoints loaded (session-scoped).

    Authenticates with password "ABC" if a password is set, otherwise
    connects without authentication.
    """
    from libs.FTLAPI import FTLAPI
    # Check if authentication is required
    r = requests.get(f"{FTL_URL}/api/auth", timeout=5)
    data = r.json()
    if data.get("session", {}).get("valid"):
        client = FTLAPI(FTL_URL)
    else:
        client = FTLAPI(FTL_URL, "ABC")
    client.get_endpoints()
    return client
