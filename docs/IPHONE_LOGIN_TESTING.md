# iPhone login and session testing

**This document was rewritten this session (2026-11-13) because it had gone
stale**: it previously described a demo pairing-key system
(`scripts/serve_demo.py`) that commit `3ce2f3a3` ("Simplify login to a
Mac-local operator console, remove demo pairing-key system") deleted
entirely, without the doc being updated. The content below describes the
system as it actually exists now.

## How sign-in actually works

There is no SMS provider configured in this project (per
`docs/DEVELOPMENT_CHECKPOINT_2026-11-12.md`: "Delivery is mock-only"). A
verification code is read from a **Mac-local operator console** instead --
a separate, tiny FastAPI app (`src/accident_risk/api/operator.py`), started
alongside the main backend by `scripts/serve.py`, bound to `127.0.0.1`
only. It is never reachable through ngrok: that is a real network-isolation
boundary (there is no route from the public internet to that port at all),
not a client-side check. It reuses the SAME account/`auth_codes` database
as the main app -- generating a code there is exactly the same operation
the main app's own `/app/api/auth/request-code` would perform, just with
the code revealed in the response instead of (never-configured) SMS.

### Start everything

```sh
.venv/bin/python scripts/serve.py
```

This prints both URLs:

```text
RoadSense backend:      http://0.0.0.0:8000  (point ngrok at this)
Operator console (Mac): http://127.0.0.1:8787  (never exposed via ngrok)
```

(`--port`/`--operator-port` override the defaults; see `scripts/serve.py
--help`.)

### Reach it from an iPhone

```sh
ngrok http 8000
```

Use the printed HTTPS URL. A Mac's `localhost` address cannot reach that
Mac from a phone; on the same trusted LAN, the Mac's LAN IP also works if
RoadSense is granted local-network access in iOS Settings. The backend URL
is editable from **Server Settings** on both the login and signed-in
screens -- changing it signs out the old endpoint's account.

### Sign in

1. Open `http://127.0.0.1:8787` in a browser **on the same Mac** running
   `serve.py` (never through the ngrok tunnel -- it can't reach this port
   anyway).
2. Enter the same phone number the iPhone app will use, tap **Generate
   Verification Code**.
3. On the iPhone, enter that phone number, request a code, then enter the
   code shown in the console. Single-use, expires per `CODE_TTL_SECONDS`
   (`src/accident_risk/api/auth.py`).

An account with the `driver` role must already exist for that phone number
-- an account manager (or the auto-created bootstrap admin, printed at
server startup as `[SYSTEM] Initialized Admin user: ...`) adds one via
`POST /app/api/drivers` (manager-only), which the dashboard's own
driver-management UI calls; there is no self-service sign-up.

## Repeatable software checks

```sh
.venv/bin/python -m pytest -q
cd ios && swift test
xcodebuild -project ios/RoadSense.xcodeproj -scheme RoadSense \
  -configuration Debug -sdk iphonesimulator \
  -destination 'generic/platform=iOS Simulator' \
  -derivedDataPath /private/tmp/roadsense-build CODE_SIGNING_ALLOWED=NO build
```

`cd ios && swift test` runs directly against the real app's own source
tree (`ios/Package.swift`'s `RoadSenseAuth` target) -- no file copying, no
temporary package. (An earlier version of this workflow,
`scripts/test_ios_auth.sh`, copied `AuthManager.swift`/`RoadSenseAPI.swift`
into a throwaway package because the real `ios/Package.swift` could not be
built or tested at all at the time; that script was removed this session
once the real package was fixed -- see
`docs/DEVELOPMENT_PROGRESS.md`'s Phase 1 entry for why it was broken and
what a direct fix looked like. Using the copy-based script now would also
silently miss every pure-logic test file added after it was written.)

`AuthTests.swift` covers unreachable servers, expired-tunnel HTML,
malformed JSON/MIME, valid JSON, endpoint changes, token expiry, and
failed session starts -- all via a stubbed `URLProtocol`, never a real
server. Python tests cover account creation, operator-console code
generation/verification, session ownership, and (per this session's
checkpoint fixes) incident-lifecycle enforcement and dedup. **A successful
build and these tests do not prove physical-device reachability** -- no
physical iPhone was available in this environment this session; see
`docs/DEVELOPMENT_PROGRESS.md`'s Phase 18 notes for what that would still
require.

## Diagnosing failures

- **HTML instead of JSON:** stale/expired tunnel, tunnel interstitial, or
  wrong URL. Check tunnel upstream and backend version. Unknown API paths
  return JSON 404.
- **Cannot reach server:** ensure the Mac and backend are awake and
  running, ngrok is connected, and the phone has internet/local-network
  permission.
- **No account for this phone number:** the operator console only
  generates a code for an EXISTING account; an account manager must add
  the driver first.
- **Session expired:** sign in again. Invalid HTML, missing JSON fields,
  redirects, and failed HTTP status codes never count as a successful
  login/session creation.
