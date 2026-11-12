"""Family accident notifications. MockNotificationSender is the only sender
implemented: it logs a notification and records it as delivered within the
app's own database, but sends nothing over SMS, push, or email. No real
delivery provider is wired up. Report accordingly -- never claim a real
notification was sent or received because this module ran.

A real provider (Twilio SMS, APNs, etc.) would implement the same `send`
signature and could be swapped in without changing callers.
"""

from __future__ import annotations

import logging
from typing import Any, Protocol

LOGGER = logging.getLogger("accident_risk.notifications")


class NotificationSender(Protocol):
    def send(self, *, contact_phone: str, contact_name: str, message: str) -> str:
        """Returns a delivery_status string describing what happened."""
        ...


class MockNotificationSender:
    """Local-only mock for automated tests and development. Keeps an
    in-memory record of everything it was asked to send so a test can
    assert on it directly, without needing a real phone or inbox."""

    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []

    def send(self, *, contact_phone: str, contact_name: str, message: str) -> str:
        record = {"contact_phone": contact_phone, "contact_name": contact_name, "message": message}
        self.sent.append(record)
        LOGGER.info("mock_family_notification", extra={"context": record})
        return "mock_sent"


def build_message(
    driver_name: str, incident_status: str, share_location: bool,
    latitude: float | None, longitude: float | None,
    *, confirmed_by_driver: bool = True,
) -> str:
    """One line, human-readable, consent-respecting, and status-honest:
    - "confirmed" means SOMEONE confirmed it -- but not necessarily the
      driver. An account manager can act on another user's alert
      (role_allows(..., "account_manager") in routes_app.py), and wording
      this as "{driver} confirmed" when a manager did it instead would
      misrepresent who actually attested to the collision. Checkpoint
      resume task #4 asks for exactly this distinction.
    - "unconfirmed_timeout" means ONLY a sensor trigger -- the driver never
      responded -- and must never be worded as a verified accident.
    Coordinates are included ONLY when share_location is True for THIS
    contact -- never because the incident happens to have them."""
    if incident_status == "confirmed" and confirmed_by_driver:
        base = f"RoadSense: {driver_name} confirmed they were in a collision."
    elif incident_status == "confirmed":
        base = f"RoadSense: a collision for {driver_name} was confirmed by an account manager, not the driver."
    elif incident_status == "unconfirmed_timeout":
        base = (
            f"RoadSense: a possible collision was detected for {driver_name}, "
            "but has NOT been confirmed (no response in time). This may be a false alarm."
        )
    else:
        # Defensive fallback for any other status this function is ever
        # called with -- still honest about not being a confirmed accident.
        base = f"RoadSense: an unconfirmed collision alert ({incident_status}) was recorded for {driver_name}."
    if share_location and latitude is not None and longitude is not None:
        return f"{base} Last known location: {latitude:.5f}, {longitude:.5f}."
    return base
