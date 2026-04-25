"""
Pi-hole FTL API stress tests -- thread-safety of the auth subsystem.

Targets two race conditions fixed by PR #2835:

1. pi_hole_extra_headers: a global char[1024] written by every authenticated
   request handler (check_client_auth, send_api_auth_status) and read by
   civetweb's send_additional_header(). Without _Thread_local, concurrent
   requests overwrite each other's Set-Cookie SID.

2. auth_data[]: the shared session array is read/written by all civetweb
   worker threads without synchronisation. get_all_sessions() references
   strings directly via JSON_REF_STR_IN_OBJECT; a concurrent
   delete_session() memsets the slot to zero mid-serialisation. Likewise,
   check_client_auth() stores api->session = &auth_data[user_id] which is
   read later by send_api_auth_status() / get_session_object() without
   rechecking whether the slot was deleted.

Tests create sessions sequentially (respecting the 3 req/s login rate
limit), then hammer the read/delete paths with barrier-synchronised
concurrent bursts.

The "s" prefix sorts this module between the mutation tests and the
auth-workflow tests (test_z_auth.py).

Usage:
    pytest test/api/test_s_auth_stress.py -v
"""

import concurrent.futures
import re
import threading
import time

import requests

FTL_URL = "http://127.0.0.1"
PASSWORD = "stress-test-pw"
NUM_SESSIONS = 14          # close to max_sessions default (16)
BURST_ROUNDS = 50          # barrier bursts per test
WORKERS = 30               # concurrent threads per burst


# ── helpers ───────────────────────────────────────────────────────────────

def _set_password(pw, sid=None):
    headers = {"X-FTL-SID": sid} if sid else {}
    r = requests.patch(
        f"{FTL_URL}/api/config/webserver/api/password",
        json={"config": {"webserver": {"api": {"password": pw}}}},
        headers=headers, timeout=20,
    )
    assert r.status_code == 200, f"set password: {r.status_code} {r.text}"


def _login(pw, timeout=10):
    return requests.post(
        f"{FTL_URL}/api/auth", json={"password": pw}, timeout=timeout)


def _login_rate_limited(pw, timeout=10):
    for _ in range(30):
        r = _login(pw, timeout=timeout)
        if r.status_code != 429:
            return r
        time.sleep(1)
    return r


def _check(sid, timeout=5):
    return requests.get(
        f"{FTL_URL}/api/auth",
        headers={"X-FTL-SID": sid}, timeout=timeout)


def _logout(sid, timeout=5):
    return requests.delete(
        f"{FTL_URL}/api/auth",
        headers={"X-FTL-SID": sid}, timeout=timeout)


def _sessions(sid, timeout=5):
    """GET /api/auth/sessions -- lists every session slot."""
    return requests.get(
        f"{FTL_URL}/api/auth/sessions",
        headers={"X-FTL-SID": sid}, timeout=timeout)


def _delete_slot(sid, slot_id, timeout=5):
    """DELETE /api/auth/session/{id} -- delete session by slot index."""
    return requests.delete(
        f"{FTL_URL}/api/auth/session/{slot_id}",
        headers={"X-FTL-SID": sid}, timeout=timeout)


def _logout_all(sids):
    for s in sids:
        try:
            _logout(s)
        except Exception:
            pass


def _create_sessions(n):
    sids = []
    for i in range(n):
        r = _login_rate_limited(PASSWORD)
        assert r.status_code == 200, f"login {i}: {r.status_code} {r.text}"
        sid = r.json()["session"]["sid"]
        assert sid, f"login {i}: empty SID"
        sids.append(sid)
    return sids


def _cookie_sid(resp):
    """Extract SID from ``Set-Cookie: sid=<value>;...``."""
    m = re.search(r"sid=([^;]+)", resp.headers.get("Set-Cookie", ""))
    return m.group(1) if m else None


def _assert_alive(ctx=""):
    """Verify FTL has not crashed."""
    try:
        r = requests.get(f"{FTL_URL}/api/auth", timeout=5)
        assert r.status_code in (200, 401), f"FTL unhealthy: {r.status_code}"
    except requests.ConnectionError:
        msg = "FTL crashed (connection refused)"
        if ctx:
            msg += f" -- {ctx}"
        raise AssertionError(msg)


# ── module fixtures ───────────────────────────────────────────────────────

def setup_module(_mod):
    _set_password(PASSWORD)


def teardown_module(_mod):
    time.sleep(2)
    try:
        r = _login_rate_limited(PASSWORD)
        sid = r.json().get("session", {}).get("sid")
        _set_password("", sid=sid)
    except Exception:
        pass  # FTL may have crashed during the tests


# ── tests ─────────────────────────────────────────────────────────────────

class TestSetCookieRace:
    """pi_hole_extra_headers is a global char[1024] written by
    check_client_auth() (auth.c:247) and send_api_auth_status() (auth.c:382)
    on every authenticated request, then consumed by civetweb's
    send_additional_header().

    Without _Thread_local, thread A's Set-Cookie SID is overwritten by
    thread B before civetweb sends thread A's response headers.

    Detection: each response's Set-Cookie SID must match the SID that was
    sent in the request.  A mismatch proves another thread's SID leaked in.
    """

    def test_cookie_echoes_correct_sid(self):
        """Concurrent validates must each get their own SID back in
        Set-Cookie, not another session's SID."""
        sids = _create_sessions(NUM_SESSIONS)
        try:
            mismatches = []

            for rnd in range(BURST_ROUNDS):
                barrier = threading.Barrier(len(sids), timeout=10)

                def fire(sid, _barrier=barrier):
                    try:
                        _barrier.wait()
                    except threading.BrokenBarrierError:
                        return None
                    r = _check(sid)
                    if r.status_code != 200:
                        return None
                    got = _cookie_sid(r)
                    if got and got != "deleted" and got != sid:
                        return (sid, got)
                    return None

                with concurrent.futures.ThreadPoolExecutor(len(sids)) as pool:
                    futs = [pool.submit(fire, s) for s in sids]
                    for f in concurrent.futures.as_completed(futs):
                        try:
                            pair = f.result()
                        except requests.ConnectionError:
                            _assert_alive("Set-Cookie burst")
                            continue
                        if pair:
                            mismatches.append(
                                f"round {rnd}: sent {pair[0][:12]}... "
                                f"but cookie has {pair[1][:12]}..."
                            )
                if mismatches:
                    break

            _assert_alive("Set-Cookie race test")
            assert not mismatches, (
                f"Set-Cookie SID mismatch ({len(mismatches)}x) -- "
                f"pi_hole_extra_headers race: {mismatches[0]}"
            )
        finally:
            _logout_all(sids)


class TestSessionListRace:
    """get_all_sessions() (auth.c:280) iterates every auth_data[] slot and
    builds JSON with JSON_REF_STR_IN_OBJECT -- raw pointers into the shared
    session array.  A concurrent delete_session() (auth.c:355) memsets a
    slot to zero, so the iteration reads partially-zeroed data.

    Detection: a session listed as *valid* must have a non-empty
    remote_addr (always set to the client IP at login, auth.c:602).  An
    empty remote_addr on a valid session means delete_session() zeroed the
    struct while get_all_sessions() was mid-iteration.

    Fixed by a pthread_mutex_t protecting auth_data and switching to
    JSON_COPY_STR_TO_OBJECT.
    """

    def test_listed_sessions_not_corrupted_during_deletes(self):
        """Flood GET /api/auth/sessions while deleting sessions by slot ID.
        Valid sessions must never have empty fields."""
        sids = _create_sessions(NUM_SESSIONS)
        admin_sid = sids[0]

        try:
            # Discover internal slot IDs
            r = _sessions(admin_sid)
            assert r.status_code == 200, f"initial listing: {r.status_code}"
            victim_slots = [
                e["id"] for e in r.json()["sessions"]
                if not e.get("current_session")
            ]

            corruptions = []

            for slot in victim_slots:
                barrier = threading.Barrier(WORKERS, timeout=10)

                def lister(_b=barrier):
                    try:
                        _b.wait()
                    except threading.BrokenBarrierError:
                        return None
                    try:
                        resp = _sessions(admin_sid)
                    except requests.ConnectionError:
                        return "CRASH"
                    if resp.status_code != 200:
                        return None
                    for e in resp.json().get("sessions", []):
                        # A valid session must have a non-empty remote_addr
                        if e.get("valid") and not e.get("remote_addr"):
                            return (
                                f"slot {e['id']}: valid but empty remote_addr"
                            )
                    return None

                def deleter(_b=barrier, _slot=slot):
                    try:
                        _b.wait()
                    except threading.BrokenBarrierError:
                        return None
                    try:
                        _delete_slot(admin_sid, _slot)
                    except requests.ConnectionError:
                        return "CRASH"
                    return None

                with concurrent.futures.ThreadPoolExecutor(WORKERS) as pool:
                    futs = [pool.submit(deleter)]
                    futs += [pool.submit(lister) for _ in range(WORKERS - 1)]
                    for f in concurrent.futures.as_completed(futs):
                        err = f.result()
                        if err == "CRASH":
                            _assert_alive("session-list race burst")
                        elif err:
                            corruptions.append(
                                f"while deleting slot {slot}: {err}"
                            )

                if corruptions:
                    break

            _assert_alive("session-list race test")
            assert not corruptions, (
                f"Corrupted session data in listing ({len(corruptions)}x) -- "
                f"auth_data race in get_all_sessions(): {corruptions[0]}"
            )
        finally:
            _logout_all(sids)


class TestValidateDeleteRace:
    """check_client_auth() (auth.c:274) stores
    api->session = &auth_data[user_id], then returns.
    send_api_auth_status() (auth.c:382) reads auth_data[user_id].sid to
    build the Set-Cookie header, and calls get_session_object() (auth.c:337)
    which uses JSON_REF_STR_IN_OBJECT on auth_data[user_id].sid and .csrf.

    A concurrent delete_session() can memset the slot between these steps:
    - Wrong SID in the JSON body (slot reused by another session)
    - Wrong SID in Set-Cookie (pi_hole_extra_headers overwritten)
    - SIGSEGV on weakly-ordered architectures (ARM)

    Fixed by copying the session to a local struct under a mutex.
    """

    def test_validate_while_deleting_same_session(self):
        """For every session, fire a validate and a delete simultaneously.
        Repeat several rounds to increase the chance of hitting the race
        window between check_client_auth and get_session_object."""
        all_errors = []

        for rnd in range(3):
            sids = _create_sessions(NUM_SESSIONS)
            n = len(sids)
            barrier = threading.Barrier(n * 2, timeout=10)
            round_errors = []

            def validate(sid, _b=barrier):
                try:
                    _b.wait()
                except threading.BrokenBarrierError:
                    return None
                try:
                    r = _check(sid)
                except requests.ConnectionError:
                    return "CRASH"
                if r.status_code != 200:
                    return None
                try:
                    s = r.json().get("session", {})
                except ValueError:
                    return f"{sid[:12]}: corrupted JSON"
                # If the delete won the race, get_session_object sees
                # used=false and returns valid=false.  That is a benign
                # TOCTOU (check_client_auth and get_session_object each
                # acquire the lock independently), not a data-corruption
                # bug, so skip the remaining checks.
                if not s.get("valid"):
                    return None
                # Wrong SID in response body — auth_data slot was
                # overwritten by another session between the two locks
                ret = s.get("sid")
                if ret and ret != sid:
                    return f"{sid[:12]}: body SID={ret[:12]}"
                # Wrong SID in Set-Cookie (pi_hole_extra_headers race)
                csid = _cookie_sid(r)
                if csid and csid != "deleted" and csid != sid:
                    return f"{sid[:12]}: cookie SID={csid[:12]}"
                return None

            def delete(sid, _b=barrier):
                try:
                    _b.wait()
                except threading.BrokenBarrierError:
                    return None
                try:
                    _logout(sid)
                except requests.ConnectionError:
                    return "CRASH"
                return None

            with concurrent.futures.ThreadPoolExecutor(n * 2) as pool:
                futs = []
                for sid in sids:
                    futs.append(pool.submit(validate, sid))
                    futs.append(pool.submit(delete, sid))
                for f in concurrent.futures.as_completed(futs):
                    err = f.result()
                    if err == "CRASH":
                        _assert_alive("validate+delete burst")
                    elif err:
                        round_errors.append(f"round {rnd}: {err}")

            all_errors.extend(round_errors)
            if all_errors:
                break

        _assert_alive("validate+delete race test")
        assert not all_errors, (
            f"Validate+delete race ({len(all_errors)}x): {all_errors[0]}"
        )

    def test_validate_survivors_during_concurrent_deletes(self):
        """Delete half the sessions while validating the other half.
        Surviving sessions must still be valid with the correct SID --
        concurrent deletes of *other* slots must not corrupt them."""
        sids = _create_sessions(NUM_SESSIONS)
        keepers = sids[: NUM_SESSIONS // 2]
        victims = sids[NUM_SESSIONS // 2:]

        try:
            errors = []
            barrier = threading.Barrier(len(sids), timeout=10)

            def validate(sid, _b=barrier):
                try:
                    _b.wait()
                except threading.BrokenBarrierError:
                    return None
                try:
                    r = _check(sid)
                except requests.ConnectionError:
                    _assert_alive("survivor validation")
                    return None
                if r.status_code != 200:
                    return None
                s = r.json().get("session", {})
                if not s.get("valid"):
                    return f"{sid[:12]}: valid=false (not deleted)"
                ret = s.get("sid")
                if ret and ret != sid:
                    return f"{sid[:12]}: SID={ret[:12]}"
                csid = _cookie_sid(r)
                if csid and csid != "deleted" and csid != sid:
                    return f"{sid[:12]}: cookie={csid[:12]}"
                return None

            def delete(sid, _b=barrier):
                try:
                    _b.wait()
                except threading.BrokenBarrierError:
                    return None
                try:
                    _logout(sid)
                except requests.ConnectionError:
                    pass
                return None

            with concurrent.futures.ThreadPoolExecutor(len(sids)) as pool:
                futs = [pool.submit(validate, s) for s in keepers]
                futs += [pool.submit(delete, s) for s in victims]
                for f in concurrent.futures.as_completed(futs):
                    err = f.result()
                    if err:
                        errors.append(err)

            _assert_alive("survivor validation test")
            assert not errors, f"Survivor validation failures: {errors}"
        finally:
            _logout_all(keepers)
