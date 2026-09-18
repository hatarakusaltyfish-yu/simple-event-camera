import cv2
import numpy as np
import os


# ============================================================
# Configuration
# ============================================================

EVENT_FILE = "result/events.npy"
OUTPUT_VIDEO = "result/event_visualization.mp4"

WIDTH = 1280
HEIGHT = 760
FPS = 30


# ============================================================
# Load events
# ============================================================

print("Loading events...")

events = np.load(EVENT_FILE)

print(f"Events loaded: {len(events):,}")

if len(events) == 0:
    raise RuntimeError("events.npy contains no events.")


# ============================================================
# Timestamp information
# ============================================================

t_min = events["t"].min()
t_max = events["t"].max()

print(f"Timestamp range: {t_min:.2f} ~ {t_max:.2f} us")
print(f"Duration: {(t_max - t_min) / 1_000_000:.3f} s")


# ============================================================
# Video writer
# ============================================================

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

writer = cv2.VideoWriter(
    OUTPUT_VIDEO,
    fourcc,
    FPS,
    (WIDTH, HEIGHT)
)

if not writer.isOpened():
    raise RuntimeError(
        "Cannot create MP4 video."
    )


# ============================================================
# Generate frames
# ============================================================

frame_interval_us = 1_000_000.0 / FPS

start_time = t_min

frame_index = 0

while True:

    frame_start = (
        start_time
        + frame_index * frame_interval_us
    )

    frame_end = (
        frame_start
        + frame_interval_us
    )

    if frame_start > t_max:
        break

    # --------------------------------------------------------
    # Select events in this time interval
    # --------------------------------------------------------

    mask = (
        (events["t"] >= frame_start)
        &
        (events["t"] < frame_end)
    )

    current = events[mask]

    # --------------------------------------------------------
    # Create event image
    # --------------------------------------------------------

    canvas = np.zeros(
        (HEIGHT, WIDTH, 3),
        dtype=np.uint8
    )

    if len(current) > 0:

        xs = current["x"]
        ys = current["y"]

        valid = (
            (xs >= 0)
            &
            (xs < WIDTH)
            &
            (ys >= 0)
            &
            (ys < HEIGHT)
        )

        xs = xs[valid]
        ys = ys[valid]
        polarity = current["p"][valid]

        # Positive events -> red
        pos = polarity == 1
        canvas[ys[pos], xs[pos]] = (0, 0, 255)

        # Negative events -> blue
        neg = polarity == 0
        canvas[ys[neg], xs[neg]] = (255, 0, 0)

    # --------------------------------------------------------
    # Save frame
    # --------------------------------------------------------

    writer.write(canvas)

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    cv2.imshow(
        "Event Camera Visualization",
        canvas
    )

    key = cv2.waitKey(
        max(1, int(1000 / FPS))
    )

    if key == 27:
        break

    frame_index += 1

    if frame_index % 30 == 0:
        print(
            f"\rFrames generated: {frame_index}",
            end=""
        )


# ============================================================
# Cleanup
# ============================================================

writer.release()
cv2.destroyAllWindows()

print()
print("=" * 60)
print("Visualization finished.")
print(f"Frames generated : {frame_index}")
print(f"Output video     : {OUTPUT_VIDEO}")

if os.path.exists(OUTPUT_VIDEO):

    size = os.path.getsize(OUTPUT_VIDEO)

    print(
        f"File size        : "
        f"{size / 1024 / 1024:.2f} MB"
    )

print("=" * 60)