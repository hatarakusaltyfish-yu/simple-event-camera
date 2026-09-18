import cv2
import numpy as np
import time
import os
from multiprocessing import Pool, cpu_count
from typing import Optional
import struct

from event_camera_simulator import ( # type: ignore
    init_worker,
    process_frame_pair,
)



# ============================================================
# High Performance Video Processor
# ============================================================

class HighPerformanceVideoProcessor:

    def __init__(
        self,
        threshold=0.15,
        noise_sigma=0.5,
        dt_resolution_us=10.0,
        workers=None,
        max_total_events=20_000_000,
        random_seed=42,
    ):

        self.threshold = threshold
        self.noise_sigma = noise_sigma
        self.dt_resolution_us = dt_resolution_us
        self.max_total_events = max_total_events
        self.random_seed = random_seed

        if workers is None:
            workers = max(1, cpu_count() - 2)

        self.workers = workers

    # --------------------------------------------------------
    # Read video
    # --------------------------------------------------------

    def read_video(self, video_path):

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise RuntimeError(
                f"Cannot open video: {video_path}"
            )

        fps = cap.get(cv2.CAP_PROP_FPS)

        width = int(
            cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        )

        height = int(
            cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        )

        frames = []

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            frames.append(frame)

        cap.release()

        if len(frames) < 2:
            raise RuntimeError(
                "Video must contain at least 2 frames."
            )

        return frames, fps, width, height

    # --------------------------------------------------------
    # Main processing
    # --------------------------------------------------------

    def process_video(
        self,
        video_path,
        output_path="events.npy",
        visualize=False,
    ):

        print("=" * 60)
        print("Event Camera Simulator")
        print("=" * 60)

        # ----------------------------------------------------
        # Read video
        # ----------------------------------------------------

        print("\n[1] Reading video...")

        start_total = time.perf_counter()

        frames, fps, width, height = \
            self.read_video(video_path)

        n_frames = len(frames)

        frame_interval_us = \
            1_000_000.0 / fps

        duration = n_frames / fps

        print(f"Resolution : {width} x {height}")
        print(f"FPS        : {fps:.2f}")
        print(f"Frames     : {n_frames}")
        print(f"Duration   : {duration:.3f} s")
        print(f"Workers    : {self.workers}")

        # ----------------------------------------------------
        # Prepare frame pairs
        # ----------------------------------------------------

        print("\n[2] Preparing frame pairs...")

        tasks = []

        for i in range(n_frames - 1):

            timestamp_us = (
                (i + 1) * frame_interval_us
            )

            tasks.append(
                (
                    i,
                    frames[i],
                    frames[i + 1],
                    timestamp_us,
                    frame_interval_us,
                )
            )

        # ----------------------------------------------------
        # Multiprocessing
        # ----------------------------------------------------

        print("\n[3] Processing frame pairs...")

        start_process = time.perf_counter()

        all_results = []

        with Pool(
            processes=self.workers,
            initializer=init_worker,
            initargs=(
                self.threshold,
                self.noise_sigma,
                self.dt_resolution_us,
                self.max_total_events,
                self.random_seed,
            ),
        ) as pool:

            # chunksize avoids too much multiprocessing
            # scheduling overhead
            chunksize = max(
                1,
                len(tasks) // (self.workers * 4)
            )

            for idx, events in pool.imap(
                process_frame_pair,
                tasks,
                chunksize=chunksize
            ):

                all_results.append(
                    (idx, events)
                )

                if len(all_results) % 10 == 0:

                    elapsed = (
                        time.perf_counter()
                        - start_process
                    )

                    processed_fps = \
                        len(all_results) / elapsed

                    print(
                        f"\rProcessed pairs: "
                        f"{len(all_results):4d}/"
                        f"{len(tasks)} | "
                        f"{processed_fps:6.2f} FPS",
                        end=""
                    )

        print()

        # ----------------------------------------------------
        # Sort frame order
        # ----------------------------------------------------

        all_results.sort(
            key=lambda x: x[0]
        )

        # ----------------------------------------------------
        # Concatenate events
        # ----------------------------------------------------

        print("\n[4] Combining events...")

        event_arrays = [
            events
            for _, events in all_results
            if len(events) > 0
        ]

        if event_arrays:
            all_events = np.concatenate(
                event_arrays
            )

        else:
            all_events = np.empty(
                0,
                dtype=np.dtype([
                    ("x", np.int32),
                    ("y", np.int32),
                    ("p", np.uint8),
                    ("t", np.float64),
                ])
            )

        # ----------------------------------------------------
        # Limit total events
        # ----------------------------------------------------

        if len(all_events) > self.max_total_events:

            print(
                f"\nWARNING: "
                f"event count {len(all_events):,} "
                f"> max_total_events "
                f"{self.max_total_events:,}"
            )

            all_events = \
                all_events[:self.max_total_events]

        # ----------------------------------------------------
        # Sort by timestamp
        # ----------------------------------------------------

        print("[5] Sorting events by timestamp...")

        if len(all_events) > 0:

            order = np.argsort(
                all_events["t"],
                kind="stable"
            )

            all_events = all_events[order]

        # ----------------------------------------------------
        # Save
        # ----------------------------------------------------

        print("[6] Saving events...")

        """
        np.save(
            output_path,
            all_events
        )
        

        np.savez_compressed(
            "result/events.npz",
            all_events
        )
        """

        np.savetxt(
            "result/all_events.txt",
            all_events,
            delimiter=" ",
            header="",
            comments="",
            fmt=["%d", "%d", "%d", "%d"]
        )          

        total_time = (
            time.perf_counter()
            - start_total
        )

        process_time = (
            time.perf_counter()
            - start_process
        )

        # ----------------------------------------------------
        # Statistics
        # ----------------------------------------------------

        n_events = len(all_events)

        positive = np.sum(
            all_events["p"] == 1
        )

        negative = np.sum(
            all_events["p"] == 0
        )

        pixels = width * height

        print("\n" + "=" * 60)
        print("RESULT")
        print("=" * 60)

        print(
            f"Frames             : {n_frames}"
        )

        print(
            f"Total events       : {n_events:,}"
        )

        print(
            f"Positive events    : {positive:,}"
        )

        print(
            f"Negative events    : {negative:,}"
        )

        print(
            f"Events / pixel     : "
            f"{n_events / pixels:.4f}"
        )

        print(
            f"Processing time    : "
            f"{process_time:.3f} s"
        )

        print(
            f"Total time         : "
            f"{total_time:.3f} s"
        )

        print(
            f"Effective FPS      : "
            f"{(n_frames - 1) / process_time:.2f}"
        )

        print(
            f"Realtime ratio     : "
            f"{process_time / duration:.2f}x"
        )
        """
        print(
            f"Output file        : "
            f"{output_path}"
        )
        """
        print("=" * 60)

        # ----------------------------------------------------
        # Optional visualization
        # ----------------------------------------------------

        if visualize:

            self.visualize_events(
                video_path,
                all_events
            )

        return all_events

    # --------------------------------------------------------
    # Event visualization
    # --------------------------------------------------------

    def visualize_events(
        self,
        video_path,
        events
    ):

        cap = cv2.VideoCapture(
            video_path
        )

        fps = cap.get(
            cv2.CAP_PROP_FPS
        )

        if fps <= 0:
            return

        frame_interval_us = \
            1_000_000.0 / fps

        index = 0

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            t_start = index * frame_interval_us
            t_end = (index + 1) * frame_interval_us

            mask = (
                (events["t"] >= t_start)
                &
                (events["t"] < t_end)
            )

            current = events[mask]

            canvas = np.zeros_like(frame)

            if len(current) > 0:

                xs = current["x"]
                ys = current["y"]

                valid = (
                    (xs >= 0)
                    &
                    (xs < frame.shape[1])
                    &
                    (ys >= 0)
                    &
                    (ys < frame.shape[0])
                )

                xs = xs[valid]
                ys = ys[valid]

                # Positive events -> bright
                pos = (
                    current["p"][valid] == 1
                )

                canvas[ys[pos], xs[pos]] = \
                    (255, 255, 255)

                # Negative events -> gray
                neg = ~pos

                canvas[ys[neg], xs[neg]] = \
                    (100, 100, 100)

            cv2.imshow(
                "Event Camera Simulation",
                canvas
            )

            key = cv2.waitKey(
                max(1, int(1000 / fps))
            )

            if key == 27:
                break

            index += 1

        cap.release()
        cv2.destroyAllWindows()


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # Configuration
    # --------------------------------------------------------

    VIDEO_PATH = "input/input_video.mp4"

    OUTPUT_PATH = "result/events.npy"

    # Start with 4 / 8 / 12 workers and benchmark.
    WORKERS = 8

    simulator = HighPerformanceVideoProcessor(

        # Event threshold
        threshold=0.15,

        # Sensor noise
        noise_sigma=0.5,

        # Timestamp resolution
        dt_resolution_us=10.0,

        # Number of processes
        workers=WORKERS,

        # Safety limit
        max_total_events=20_000_000,

        random_seed=42,
    )

    events = simulator.process_video(
        VIDEO_PATH,
        OUTPUT_PATH,
        visualize=False,
    )