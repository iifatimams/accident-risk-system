# Research vehicle-safety project

- Never fabricate dataset columns. Inspect actual schemas before writing mappings.
- Never modify files inside `data/raw` or download datasets during tests.
- Never introduce future-data leakage. Use participant/driver/vehicle/trip/event-aware splits; keep every event in one partition.
- Maintain identical preprocessing in training, replay, and live inference.
- Every new feature must have meaningful tests. Run pytest after meaningful changes.
- Use SI units internally where practical. Document timestamp units explicitly.
- Missing sensors must not crash inference. Physiology updates must never block faster vehicle/IMU inference.
- Never call an uncalibrated score a probability of an accident.
- Keep functions typed and documented. Prefer simple models before neural networks.
- Synthetic fixtures are software tests, never evidence of model performance.
