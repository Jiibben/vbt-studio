# Idea: ESP32 + IMU barbell VBT device

A wearable/on-bar velocity-based-training sensor built from an ESP32 and a 6-axis
IMU, as a second sensor modality alongside the existing camera-based VBT pipeline.

## Verdict
- **Demo-quality prototype in 2 weeks (one exercise): feasible.**
- **Validated, trustworthy, multi-exercise device in 2 weeks: not realistic** — that's
  a 1–3 month effort. The gap between "shows a velocity number" and "shows the *right*
  velocity number reliably" is the whole difficulty, and it's mostly empirical tuning
  that AI can't shortcut.
- Money is a non-issue (~$30). The cost is time and iteration on real noisy data.
- Algorithms are known/not-novel, so AI can write the heavy code.

## Why it's hard
Core problem: **double integration of a noisy, biased signal.**
- Velocity = integrate accel once; position = twice. Any constant error in accel grows
  linearly into velocity, quadratically into position (classic inertial drift).
- **The killer is gravity removal, not noise.** The accelerometer always reads 1g mixed
  into the motion. To subtract it you must know the sensor's orientation every instant
  → need a **6-axis IMU (accel + gyro)**, not just an accelerometer, plus a sensor-fusion
  filter (Madgwick / Mahony / complementary) to track tilt.
- A **1° orientation error injects ~0.17 m/s² of fake acceleration**, which over a
  half-second rep ruins the velocity reading. Tilt error dominates sensor noise by a lot.

## Why it's tractable
VBT's saving grace: **the bar is momentarily at rest at the start and end of every rep.**
- Use **ZUPT (Zero-Velocity Update)**: detect the stationary moments and reset integrated
  velocity to zero each rep → kills accumulated drift before it runs away. This is *the*
  known trick that makes IMU-based VBT work. Errors become bounded per-rep.
- Pipeline (all well-trodden): calibrate bias → sensor fusion for orientation → rotate out
  gravity → integrate with ZUPT → segment reps → report mean/peak concentric velocity.

## Realistic timeline (solo, full-time, AI-assisted)
| Phase | What | Time |
|---|---|---|
| 1 | ESP32 + IMU streaming data over USB/BLE, logging to CSV | 1–2 days |
| 2 | Orientation filter + gravity removal → clean linear accel | 2–3 days |
| 3 | Integration + ZUPT + rep segmentation → a velocity number | 3–4 days |
| 4 | **Validation & tuning against a reference** — the real time sink | open-ended |
| 5 | 3D-printed mount, battery, BLE readout to phone/laptop | 2–3 days |

Phases 1–3 + a basic enclosure realistically fill 2 weeks → a working **one-exercise demo**
(squat or bench). Phase 4 is what separates a demo from a product and never really ends.

## The validation problem (+ synergy)
- The trap: **no ground truth.** A plausible-looking velocity can be 30% off and you'd never
  know. Commercial devices validate against linear position transducers (GymAware).
- **Synergy:** the existing camera-based VBT pipeline can serve as the reference. Run both on
  the same rep and compare. Two independent sensor modalities agreeing is a strong story.
- Honest caveat: sports-science literature generally finds **IMU-based VBT less accurate /
  reliable than linear encoders or good camera setups.** The accelerometer version's selling
  points are portability + instant on-bar real-time feedback, not precision. It likely won't
  *beat* the camera project on accuracy.

## Budget (~$25–40; ~$55 with spares)
| Item | Cost |
|---|---|
| ESP32 dev board | $5–10 |
| 6-axis IMU | MPU6050 $2 (noisy starter) / **ICM-42688-P or LSM6DSO ~$10–15** (better) |
| LiPo battery + TP4056 charger | $5–10 |
| Wires, switch, JST connectors | ~$5 |
| 3D-printed enclosure | ~$2 filament |

### Hardware notes
- **Don't trust the MPU6050 for final accuracy** (noise + temperature drift). Fine for getting
  phases 1–3 working; start with **ICM-42688-P / LSM6DSO / BMI270** for good numbers.
- Sample at ~200–500 Hz (ESP32 handles this trivially).
- Mounting matters: a wobbly print adds vibration noise the filter then has to fight.

## What AI does vs. doesn't
- **AI does (fast):** Madgwick filter, ZUPT logic, ESP32/Arduino firmware, BLE plumbing,
  integration math, rep segmentation.
- **AI can't:** tune against your specific sensor noise, mounting, and lifting style — hours of
  collecting real data, plotting, adjusting thresholds. Code arrives fast; calibration is the slog.

## Bottom line
Feasible as a 2-week impressive demo for one exercise, cheap enough that cost is irrelevant, and
AI carries the heavy code. The real risk isn't the algorithm — it's that "looks like it works" and
"is actually accurate" are two different milestones, and only the first fits in two weeks. Pairing
it with the camera project for ground-truth validation is the move that makes it both technically
sound and genuinely resume-worthy.
