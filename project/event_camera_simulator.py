# ============================================================
# Event Camera Simulator V2
# ============================================================
import cv2
import numpy as np
import time
import os
from multiprocessing import Pool, cpu_count
from typing import Optional
import struct

class EventCameraSimulatorV2:
    """
    High-performance event camera simulator.

    Input:
        High-frame-rate video

    Output:
        Events:
            x : pixel x
            y : pixel y
            p : polarity (1 = positive, 0 = negative)
            t : timestamp in microseconds

    Main optimizations:
        1. NumPy vectorization
        2. Analytical interpolation
        3. Multiprocessing over frame pairs
        4. Compact NumPy event storage
    """

    def __init__(
        self,
        threshold: float = 0.15,
        noise_sigma: float = 0.5,
        dt_resolution_us: float = 10.0,
        max_total_events: int = 20_000_000,
        random_seed: int = 42,
    ):
        self.threshold = float(threshold)
        self.noise_sigma = float(noise_sigma)
        self.dt_resolution_us = float(dt_resolution_us)
        self.max_total_events = int(max_total_events)
        self.random_seed = int(random_seed)

    # --------------------------------------------------------
    # Frame preprocessing
    # --------------------------------------------------------

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """
        Convert image to log intensity.
        """

        if frame.ndim == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        img = gray.astype(np.float32)

        if self.noise_sigma > 0:
            noise = np.random.normal(
                0.0,
                self.noise_sigma,
                img.shape
            ).astype(np.float32)

            img += noise

        img = np.maximum(img, 0.0)

        # Log intensity
        return np.log1p(img)


    # --------------------------------------------------------
    # Generate events between two frames
    # --------------------------------------------------------

    def generate_events(
        self,
        prev_log: np.ndarray,
        curr_log: np.ndarray,
        timestamp_us: float,
        frame_interval_us: float,
    ):
        """
        Generate events between two frames.

        Instead of generating thousands of intermediate frames,
        calculate event timestamps analytically.

        ΔL = L_curr - L_prev

        Positive:
            floor(ΔL / C)

        Negative:
            floor(-ΔL / C)
        """

        diff = curr_log - prev_log

        threshold = self.threshold

        # ----------------------------------------------------
        # Number of positive / negative events
        # ----------------------------------------------------

        positive_count = np.floor(
            np.maximum(diff, 0.0) / threshold
        ).astype(np.int16)

        negative_count = np.floor(
            np.maximum(-diff, 0.0) / threshold
        ).astype(np.int16)

        total_positive = int(positive_count.max())
        total_negative = int(negative_count.max())

        event_chunks = []

        # ----------------------------------------------------
        # Positive events
        # ----------------------------------------------------

        for k in range(1, total_positive + 1):

            mask = positive_count >= k

            if not np.any(mask):
                continue

            ys, xs = np.nonzero(mask)

            d = diff[ys, xs]

            # Linear interpolation
            alpha = (k * threshold) / d

            alpha = np.clip(alpha, 0.0, 1.0)

            t = (
                timestamp_us
                + alpha * frame_interval_us
            )

            # Timestamp quantization
            if self.dt_resolution_us > 0:
                t = (
                    np.round(t / self.dt_resolution_us)
                    * self.dt_resolution_us
                )

            n = len(xs)

            events = np.empty(
                n,
                dtype=np.dtype([
                    ("x", np.int32),
                    ("y", np.int32),
                    ("p", np.uint8),
                    ("t", np.float64),
                ])
            )

            events["x"] = xs
            events["y"] = ys
            events["p"] = 1
            events["t"] = t

            event_chunks.append(events)

        # ----------------------------------------------------
        # Negative events
        # ----------------------------------------------------

        for k in range(1, total_negative + 1):

            mask = negative_count >= k

            if not np.any(mask):
                continue

            ys, xs = np.nonzero(mask)

            d = -diff[ys, xs]

            alpha = (k * threshold) / d

            alpha = np.clip(alpha, 0.0, 1.0)

            t = (
                timestamp_us
                + alpha * frame_interval_us
            )

            if self.dt_resolution_us > 0:
                t = (
                    np.round(t / self.dt_resolution_us)
                    * self.dt_resolution_us
                )

            n = len(xs)

            events = np.empty(
                n,
                dtype=np.dtype([
                    ("x", np.int32),
                    ("y", np.int32),
                    ("p", np.uint8),
                    ("t", np.float64),
                ])
            )

            events["x"] = xs
            events["y"] = ys
            events["p"] = 0
            events["t"] = t

            event_chunks.append(events)

        if not event_chunks:
            return np.empty(
                0,
                dtype=np.dtype([
                    ("x", np.int32),
                    ("y", np.int32),
                    ("p", np.uint8),
                    ("t", np.float64),
                ])
            )

        return np.concatenate(event_chunks)


# ============================================================
# Multiprocessing worker
# ============================================================

_WORKER_SIMULATOR = None


def init_worker(
    threshold,
    noise_sigma,
    dt_resolution_us,
    max_total_events,
    random_seed
):
    """
    Initialize simulator inside each worker.
    """

    global _WORKER_SIMULATOR

    pid = os.getpid()

    # Give each worker a different random seed
    seed = random_seed + pid

    np.random.seed(seed)

    _WORKER_SIMULATOR = EventCameraSimulatorV2(
        threshold=threshold,
        noise_sigma=noise_sigma,
        dt_resolution_us=dt_resolution_us,
        max_total_events=max_total_events,
        random_seed=seed,
    )


def process_frame_pair(args):
    """
    Worker function.

    Input:
        (frame_index, prev_frame, curr_frame, timestamp, dt)

    Output:
        (frame_index, events)
    """

    global _WORKER_SIMULATOR

    (
        frame_index,
        prev_frame,
        curr_frame,
        timestamp_us,
        frame_interval_us,
    ) = args

    sim = _WORKER_SIMULATOR

    prev_log = sim.preprocess(prev_frame)
    curr_log = sim.preprocess(curr_frame)

    events = sim.generate_events(
        prev_log,
        curr_log,
        timestamp_us,
        frame_interval_us,
    )

    return frame_index, events