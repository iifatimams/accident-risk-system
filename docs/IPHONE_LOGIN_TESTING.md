# iPhone login and session testing

The backend URL is now editable from **Server Settings** on both the login and
signed-in screens. Changing it signs out the old endpoint's account. Stop a drive
before changing servers. No tunnel URL is built into the app.

For a phone on a different network, use the current **HTTPS** tunnel URL. A Mac's
`localhost` URL cannot reach that Mac from an iPhone. On the same trusted LAN,
use the Mac's LAN address and permit RoadSense local-network access in iOS Settings.

## Restricted university demo

Start a separate demo backend from the repository root:

```sh
.venv/bin/python scripts/serve_demo.py --port 8001
```

This creates a new temporary directory, a separate SQLite database, a random
pairing key in a private mode-0600 file, and one allowed fictional phone number:
`+12025550101`. It does not open `data/app.sqlite3`, seed real administrators, load
models, or send SMS. The terminal prints the key file's path, never the key itself.
Directories are retained on exit. Phone capture may subsequently record real sensor
or location readings in that separate database; keep it out of Git.

Forward the tunnel to that port (stop a conflicting old tunnel first):

```sh
ngrok http 8001 --inspect=false
```

Disabling request inspection avoids storing login response bodies or pairing-key
headers in the local tunnel inspector. Use the printed HTTPS endpoint; do not add
`/app` or another path to the saved address.

On the iPhone:

1. Open **Server Settings**. Enter the HTTPS URL.
2. Enable **Restricted demo login**. Copy the key from the private file into the
   pairing-key field. This key authenticates the test device; never put it in a
   URL, source file, shared screenshot or chat. The app keeps it in memory only.
3. Tap **Save & Test Connection**. The phone must report **Connected to RoadSense**
   and **Restricted demo server**. A browser page or a Mac-only probe is insufficient.
4. Close Settings, enter `+12025550101`, and request a code. The screen should show
   **DEMO ONLY — Code: …** for both a newly created and an existing test account.
5. Enter the six digits and sign in. Tap **Start Drive Session** while parked.
   Capture starts only after a valid, owned session response from the backend.
6. Stop the session. Repeat once using mobile data/different Wi-Fi to establish that
   the HTTPS endpoint, rather than a local-only address, is reachable from the phone.

Demo mode requires an explicit isolated database at application creation, a pairing
key at least 32 characters long, and a phone allowlist. The launcher supplies these.
A missing/wrong key, an unlisted number, or a spoofed private/forwarded IP address
cannot reveal a verification code or create a demo account. The new mode is never
a fallback for SMS failures. Demo accounts have driver privileges.

## Normal SMS operation

Run the normal backend without demo mode. Configure `TWILIO_ACCOUNT_SID`,
`TWILIO_AUTH_TOKEN`, and `TWILIO_FROM_NUMBER` in the server process environment;
these are not read by the iPhone. Leave `ACCIDENT_RISK_DEV_MOCK_SMS_DELIVERY=false`.
The iPhone's demo toggle must also be off. Codes are sent by SMS and never echoed
in JSON or printed to server logs. Delivery failures return an actionable HTTP 503
instead of falsely reporting that a code was sent. New numbers and existing
accounts use the same delivery checks.

Existing tokens without an endpoint binding are discarded. Saved tokens for the
selected endpoint are verified by loading `/app/api/me`; the presence of a token
alone no longer opens the signed-in screen. HTTP 401/403 on authenticated requests
clears the saved login. Temporary network failure keeps a token available for
**Retry Sign-In Connection**, avoiding reuse of an already-consumed code.

## Diagnosing failures

- **HTML instead of JSON:** stale/expired tunnel, tunnel interstitial, or wrong URL.
  Check tunnel upstream and backend version. Unknown API paths now return JSON 404.
- **Cannot reach server:** ensure the Mac and backend are awake and running, ngrok
  is connected, and the phone has internet/local-network permission.
- **Private pairing key required:** use the current demo process's key and turn on
  demo login. Restarting the demo creates a new key and database.
- **SMS could not be delivered:** configure the real SMS provider or deliberately
  switch to the isolated demo. No code is exposed as a delivery-failure workaround.
- **Session expired:** sign in again. Invalid HTML, missing JSON fields, redirects,
  and failed HTTP status codes never count as successful login/session creation.
- **Backend stop not confirmed:** local capture has stopped, but the backend may
  still have an active session. Restore connectivity and stop again; starting a
  session is idempotent and returns the existing session if one is still active.

## Repeatable software checks

```sh
.venv/bin/python -m pytest -q
bash scripts/test_ios_auth.sh
xcodebuild -project ios/RoadSense.xcodeproj -scheme RoadSense \
  -configuration Debug -sdk iphonesimulator \
  -destination 'generic/platform=iOS Simulator' \
  -derivedDataPath /private/tmp/roadsense-login-build CODE_SIGNING_ALLOWED=NO build
```

The Swift harness copies the exact networking/auth sources used by the app into a
fresh temporary package and runs URLProtocol-based tests on macOS. It does not
replace the app's Swift package, alter the database, or contact a real server.
Coverage includes unreachable servers, expired-tunnel HTML, malformed JSON/MIME,
valid JSON, demo pairing, endpoint changes, token expiry and failed session starts.
Python tests cover new/existing accounts, successful/failed SMS, demo restrictions,
no public code disclosure, stale tokens, and session ownership.

A successful build and mocked tests do not prove physical-device reachability or
real SMS delivery. Record the iPhone connection/login/session result separately.
All changes in this repair remain local until the user approves a commit/push.

Verification on 2026-11-12: Python **132 passed** (67 dependency deprecation
warnings); Swift authentication tests **12 passed**; Debug simulator and signed
iPhone 12 Pro builds both succeeded. The updated app was installed in place and
launched on the selected physical phone. An external HTTPS probe reached the demo
API, and an unauthenticated code request returned 403 without a code. During the
user's phone test, the demo backend received successful code-request, verification
and profile requests. The user subsequently confirmed that the demo code displayed
and the driving session started on the selected physical iPhone 12 Pro. The captured
backend log independently confirms authentication, but contains no session-start
entry at the follow-up check; session start is therefore recorded as user-confirmed.
The phone's network during this test was not explicitly confirmed. Normal SMS
delivery and sustained sensor capture were not tested.
