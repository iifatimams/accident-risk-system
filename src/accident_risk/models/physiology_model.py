"""Optional independent driver-state model; missing SpO2 is imputed during fitting."""

from accident_risk.models.vehicle_model import VehicleModel


class PhysiologyModel(VehicleModel):
    """Uses only physiology features and independently sourced driver-state labels."""

    def __init__(
        self,
        family: str = "logistic_regression",
        seed: int = 42,
        target: str = "driver_state_risk",
        score_labels: list[str] | None = None,
    ) -> None:
        if target != "driver_state_risk":
            raise ValueError("Physiology models must target driver_state_risk")
        super().__init__(family, seed, target, score_labels)
