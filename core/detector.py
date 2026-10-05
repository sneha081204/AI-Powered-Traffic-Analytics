"""
ViolaWatch - Core Violation Detector Engine

Handles:
- Video-file processing
- Helmet violation verification
- ByteTrack ID based duplicate prevention
- Spatial duplicate prevention
- Evidence snapshot
- Database insertion
- Flask/GUI frame streaming
"""

import cv2
import os
import time
import threading
import base64

from datetime import datetime
from pathlib import Path
import sys

sys.path.insert(
    0,
    str(Path(__file__).parent.parent)
)

from utils.frame_processor import FrameProcessor
import config as cfg


class ViolationDetector:

    # =========================================================
    # INITIALIZATION
    # =========================================================

    def __init__(
        self,
        db=None,
        on_violation=None
    ):

        self.db = db
        self.on_violation = on_violation

        self.processor = FrameProcessor(
            cfg.CONFIDENCE
        )

        self.is_running = False
        self.cap = None

        self._lock = threading.Lock()
        self._current_frame = None

        # -----------------------------------------------------
        # STATISTICS
        # -----------------------------------------------------

        self.stats = {
            "fps": 0,
            "total": 0,
            "helmet": 0,
            "seatbelt": 0,
            "frames": 0
        }

        # -----------------------------------------------------
        # TRACK MEMORY
        # -----------------------------------------------------

        self._saved_tracks = set()

        # -----------------------------------------------------
        # SAVED EVENT MEMORY
        #
        # This is the important duplicate protection.
        #
        # Even if ByteTrack changes:
        #
        #     ID 14 -> ID 18
        #
        # the spatial check can still recognize that it is
        # the same vehicle/event.
        # -----------------------------------------------------

        self._saved_events = []

        # -----------------------------------------------------
        # VERIFICATION BUFFER
        # -----------------------------------------------------

        self._verify_buf = {}

        self.VERIFY_HITS = 3
        self.VERIFY_WINDOW = 30

        # -----------------------------------------------------
        # SPATIAL GRID
        # -----------------------------------------------------

        self.GRID = 120

        # -----------------------------------------------------
        # COOLDOWN
        # -----------------------------------------------------

        self.COOLDOWN = max(
            getattr(
                cfg,
                "COOLDOWN",
                15
            ),
            30
        )

    # =========================================================
    # START LIVE
    # =========================================================

    def start_live(self, source=0):

        self.cap = cv2.VideoCapture(
            source
        )

        if not self.cap.isOpened():

            self.cap = None

            raise RuntimeError(
                f"Cannot open source: {source}"
            )

        self.cap.set(
            cv2.CAP_PROP_FRAME_WIDTH,
            1280
        )

        self.cap.set(
            cv2.CAP_PROP_FRAME_HEIGHT,
            720
        )

        self.cap.set(
            cv2.CAP_PROP_BUFFERSIZE,
            1
        )

        self.is_running = True

        self._saved_tracks.clear()
        self._saved_events.clear()
        self._verify_buf.clear()

        thread = threading.Thread(
            target=self._live_loop,
            daemon=True
        )

        thread.start()

        print(
            f"[Detector] Live started: {source}"
        )

    # =========================================================
    # STOP
    # =========================================================

    def stop(self):

        print(
            "[Detector] Stop requested"
        )

        self.is_running = False

    # =========================================================
    # LIVE LOOP
    # =========================================================

    def _live_loop(self):

        frame_count = 0
        proc_frame = 0

        fps_time = time.time()
        fps_count = 0

        try:

            while self.is_running:

                if self.cap is None:
                    break

                ret, frame = self.cap.read()

                if not ret:

                    time.sleep(0.05)
                    continue

                frame_count += 1
                fps_count += 1

                if (
                    time.time()
                    - fps_time
                    >= 1.0
                ):

                    self.stats["fps"] = (
                        fps_count
                    )

                    fps_count = 0
                    fps_time = time.time()

                skip = max(
                    getattr(
                        cfg,
                        "FRAME_SKIP",
                        1
                    ),
                    1
                )

                if (
                    frame_count % skip
                    != 0
                ):

                    continue

                proc_frame += 1

                self.stats["frames"] = (
                    proc_frame
                )

                annotated, violations = (
                    self.processor.process(
                        frame
                    )
                )

                self._draw_hud(
                    annotated
                )

                for violation in violations:

                    if not self.is_running:
                        break

                    key = (
                        self._make_track_key(
                            violation
                        )
                    )

                    self._verify_violation(
                        frame,
                        violation,
                        key,
                        proc_frame,
                        source_type="live"
                    )

                with self._lock:

                    self._current_frame = (
                        annotated.copy()
                    )

        except Exception as e:

            print(
                f"[Detector] Live loop error: {e}"
            )

        finally:

            if self.cap is not None:

                try:
                    self.cap.release()
                except Exception:
                    pass

            self.cap = None
            self.is_running = False

            print(
                "[Detector] Live stopped"
            )

    # =========================================================
    # START VIDEO FILE
    # =========================================================

    def process_video_file(
        self,
        path,
        job_id=None,
        progress_cb=None,
        done_cb=None
    ):

        self.is_running = True

        # IMPORTANT:
        # Start a completely new event session.

        self._saved_tracks.clear()
        self._saved_events.clear()
        self._verify_buf.clear()

        self.stats["total"] = 0
        self.stats["helmet"] = 0
        self.stats["seatbelt"] = 0
        self.stats["frames"] = 0

        thread = threading.Thread(
            target=self._video_file_loop,
            args=(
                path,
                job_id,
                progress_cb,
                done_cb
            ),
            daemon=True
        )

        thread.start()

        print(
            "[Detector] Video processing started:",
            path
        )

    # =========================================================
    # VIDEO LOOP
    # =========================================================

    def _video_file_loop(
        self,
        path,
        job_id,
        progress_cb,
        done_cb
    ):

        cap = None

        violations_count = 0
        frame_count = 0
        processed_frames = 0

        try:

            # -------------------------------------------------
            # OPEN VIDEO
            # -------------------------------------------------

            cap = cv2.VideoCapture(
                path
            )

            if not cap.isOpened():

                print(
                    "[Detector] ERROR: "
                    "Cannot open video"
                )

                if done_cb:

                    done_cb(
                        0,
                        "Cannot open video"
                    )

                return

            self.cap = cap

            # -------------------------------------------------
            # VIDEO INFORMATION
            # -------------------------------------------------

            total_frames = int(
                cap.get(
                    cv2.CAP_PROP_FRAME_COUNT
                )
            )

            fps = (
                cap.get(
                    cv2.CAP_PROP_FPS
                )
                or 25
            )

            print(
                f"[Detector] Total frames: "
                f"{total_frames}"
            )

            print(
                f"[Detector] FPS: {fps}"
            )

            # -------------------------------------------------
            # FRAME SKIP
            # -------------------------------------------------

            skip = max(
                getattr(
                    cfg,
                    "FRAME_SKIP",
                    1
                ),
                1
            )

            print(
                f"[Detector] Frame skip: {skip}"
            )

            # -------------------------------------------------
            # DATABASE JOB
            # -------------------------------------------------

            if self.db and job_id:

                try:

                    self.db.update_job(
                        job_id,
                        status="processing",
                        total_frames=total_frames
                    )

                except Exception as e:

                    print(
                        "[Detector] Job update error:",
                        e
                    )

            # -------------------------------------------------
            # VIDEO LOOP
            # -------------------------------------------------

            while True:

                if not self.is_running:

                    print(
                        "[Detector] "
                        "Video stopped by user"
                    )

                    break

                ret, frame = cap.read()

                if not ret:
                    break

                frame_count += 1

                if (
                    frame_count % skip
                    != 0
                ):

                    continue

                processed_frames += 1

                self.stats["frames"] = (
                    processed_frames
                )

                # -------------------------------------------------
                # AI PROCESSING
                # -------------------------------------------------

                try:

                    annotated, violations = (
                        self.processor.process(
                            frame
                        )
                    )

                except Exception as e:

                    print(
                        "[Detector] "
                        f"Frame processing error: {e}"
                    )

                    continue

                # -------------------------------------------------
                # HUD
                # -------------------------------------------------

                self._draw_hud(
                    annotated
                )

                with self._lock:

                    self._current_frame = (
                        annotated.copy()
                    )

                # =================================================
                # PROCESS DETECTED VIOLATIONS
                # =================================================

                for violation in violations:

                    if not self.is_running:
                        break

                    key = (
                        self._make_track_key(
                            violation
                        )
                    )

                    # -------------------------------------------------
                    # DUPLICATE CHECK #1
                    # -------------------------------------------------

                    if self._already_saved_event(
                        violation
                    ):

                        continue

                    # -------------------------------------------------
                    # VERIFICATION
                    # -------------------------------------------------

                    buf = self._verify_buf.get(
                        key
                    )

                    # -------------------------------------------------
                    # NEW EVENT
                    # -------------------------------------------------

                    if buf is None:

                        self._verify_buf[key] = {

                            "hits": 1,

                            "first_frame":
                                processed_frames,

                            "last_frame":
                                frame.copy(),

                            "last_v":
                                violation,

                            "committed":
                                False
                        }

                        print(
                            "[Verify] New:",
                            key,
                            "hit=1/",
                            self.VERIFY_HITS
                        )

                        continue

                    # -------------------------------------------------
                    # ALREADY COMMITTED
                    # -------------------------------------------------

                    if buf["committed"]:

                        continue

                    # -------------------------------------------------
                    # CHECK VERIFICATION WINDOW
                    # -------------------------------------------------

                    if (
                        processed_frames
                        -
                        buf["first_frame"]
                        >
                        self.VERIFY_WINDOW
                    ):

                        buf["hits"] = 1

                        buf["first_frame"] = (
                            processed_frames
                        )

                        buf["last_frame"] = (
                            frame.copy()
                        )

                        buf["last_v"] = (
                            violation
                        )

                        buf["committed"] = (
                            False
                        )

                        print(
                            "[Verify] "
                            "Window reset:",
                            key
                        )

                        continue

                    # -------------------------------------------------
                    # ADD HIT
                    # -------------------------------------------------

                    buf["hits"] += 1

                    buf["last_frame"] = (
                        frame.copy()
                    )

                    buf["last_v"] = (
                        violation
                    )

                    print(
                        "[Verify]",
                        key,
                        f"hit={buf['hits']}/"
                        f"{self.VERIFY_HITS}"
                    )

                    # -------------------------------------------------
                    # CONFIRM
                    # -------------------------------------------------

                    if (
                        buf["hits"]
                        >=
                        self.VERIFY_HITS
                    ):

                        if buf["committed"]:
                            continue

                        # -------------------------------------------------
                        # DUPLICATE CHECK #2
                        #
                        # Check again immediately before saving.
                        # -------------------------------------------------

                        if self._already_saved_event(
                            buf["last_v"]
                        ):

                            buf["committed"] = (
                                True
                            )

                            continue

                        buf["committed"] = (
                            True
                        )

                        self._saved_tracks.add(
                            key
                        )

                        print(
                            "[Verify] CONFIRMED:",
                            key
                        )

                        self._save_violation(
                            buf["last_frame"],
                            buf["last_v"],
                            source_type="video"
                        )

                        # -------------------------------------------------
                        # Remember this saved event
                        # -------------------------------------------------

                        self._remember_saved_event(
                            buf["last_v"],
                            processed_frames
                        )

                        violations_count += 1

                # =================================================
                # REMOVE OLD VERIFICATION BUFFERS
                # =================================================

                old_keys = []

                for key, buf in (
                    self._verify_buf.items()
                ):

                    if (
                        processed_frames
                        -
                        buf["first_frame"]
                        >
                        self.VERIFY_WINDOW * 2
                    ):

                        old_keys.append(
                            key
                        )

                for key in old_keys:

                    self._verify_buf.pop(
                        key,
                        None
                    )

                # =================================================
                # PROGRESS
                # =================================================

                if (
                    progress_cb
                    and
                    frame_count
                    %
                    max(skip * 10, 1)
                    == 0
                ):

                    percent = int(
                        (
                            frame_count
                            /
                            max(
                                total_frames,
                                1
                            )
                        )
                        * 100
                    )

                    progress_cb(
                        percent,
                        frame_count,
                        violations_count
                    )

                # =================================================
                # DATABASE PROGRESS
                # =================================================

                if (
                    self.db
                    and job_id
                    and
                    frame_count
                    %
                    max(skip * 30, 1)
                    == 0
                ):

                    try:

                        self.db.update_job(
                            job_id,
                            processed=frame_count,
                            violations_found=
                                violations_count
                        )

                    except Exception as e:

                        print(
                            "[Detector] "
                            "Database progress error:",
                            e
                        )

            # =====================================================
            # COMPLETION
            # =====================================================

            if not self.is_running:

                print(
                    "[Detector] "
                    "Video stopped."
                )

                print(
                    "[Detector] "
                    f"Violations saved: "
                    f"{violations_count}"
                )

                if self.db and job_id:

                    try:

                        self.db.update_job(
                            job_id,
                            status="stopped",
                            processed=frame_count,
                            violations_found=
                                violations_count
                        )

                    except Exception:
                        pass

                if done_cb:

                    done_cb(
                        violations_count,
                        "Stopped by user"
                    )

                return

            print(
                "[Detector] "
                "Video processing completed."
            )

            print(
                "[Detector] "
                f"Violations saved: "
                f"{violations_count}"
            )

            print(
                "[Detector] "
                f"Unique events: "
                f"{len(self._saved_events)}"
            )

            if self.db and job_id:

                try:

                    self.db.update_job(
                        job_id,
                        status="done",
                        processed=total_frames,
                        violations_found=
                            violations_count,
                        completed_at=
                            datetime.now()
                    )

                except Exception as e:

                    print(
                        "[Detector] "
                        "Completion DB error:",
                        e
                    )

            if done_cb:

                done_cb(
                    violations_count,
                    None
                )

        except Exception as e:

            print(
                "[Detector] "
                f"Video loop error: {e}"
            )

            import traceback

            traceback.print_exc()

            if self.db and job_id:

                try:

                    self.db.update_job(
                        job_id,
                        status="error",
                        processed=frame_count,
                        violations_found=
                            violations_count
                    )

                except Exception:
                    pass

            if done_cb:

                done_cb(
                    violations_count,
                    str(e)
                )

        finally:

            if cap is not None:

                try:
                    cap.release()
                except Exception:
                    pass

            self.cap = None
            self.is_running = False

            print(
                "[Detector] "
                "Video capture released"
            )

    # =========================================================
    # CREATE TRACK KEY
    # =========================================================

    def _make_track_key(
        self,
        violation
    ):

        violation_type = (
            violation.get(
                "type",
                "unknown"
            )
        )

        track_id = (
            violation.get(
                "track_id"
            )
        )

        # -----------------------------------------------------
        # ByteTrack ID
        # -----------------------------------------------------

        if track_id is not None:

            try:

                track_id = int(
                    track_id
                )

            except Exception:
                pass

            return (
                violation_type,
                track_id
            )

        # -----------------------------------------------------
        # FALLBACK GRID
        # -----------------------------------------------------

        x1, y1, x2, y2 = (
            violation.get(
                "bbox",
                (0, 0, 100, 100)
            )
        )

        cx = int(
            (x1 + x2) / 2
        )

        cy = int(
            (y1 + y2) / 2
        )

        return (
            violation_type,
            "fallback",
            cx // self.GRID,
            cy // self.GRID
        )

    # =========================================================
    # DUPLICATE EVENT CHECK
    # =========================================================

    def _already_saved_event(
        self,
        violation
    ):

        violation_type = (
            violation.get(
                "type",
                "unknown"
            )
        )

        current_track = (
            violation.get(
                "track_id"
            )
        )

        current_bbox = (
            violation.get(
                "bbox",
                (0, 0, 0, 0)
            )
        )

        # -----------------------------------------------------
        # Check all saved events
        # -----------------------------------------------------

        for event in self._saved_events:

            if (
                event["type"]
                !=
                violation_type
            ):

                continue

            saved_track = (
                event.get(
                    "track_id"
                )
            )

            # -------------------------------------------------
            # CASE 1:
            # Same ByteTrack ID
            # -------------------------------------------------

            if (
                current_track is not None
                and
                saved_track is not None
                and
                int(current_track)
                ==
                int(saved_track)
            ):

                return True

            # -------------------------------------------------
            # CASE 2:
            # Spatial similarity
            #
            # Useful if ByteTrack changes ID.
            # -------------------------------------------------

            saved_bbox = (
                event["bbox"]
            )

            if self._boxes_are_same_event(
                current_bbox,
                saved_bbox
            ):

                return True

        return False

    # =========================================================
    # SAVE EVENT IN MEMORY
    # =========================================================

    def _remember_saved_event(
        self,
        violation,
        frame_number
    ):

        bbox = violation.get(
            "bbox",
            (0, 0, 0, 0)
        )

        self._saved_events.append({

            "type":
                violation.get(
                    "type",
                    "unknown"
                ),

            "track_id":
                violation.get(
                    "track_id"
                ),

            "bbox":
                tuple(
                    map(
                        int,
                        bbox
                    )
                ),

            "frame":
                frame_number,

            "time":
                time.time()
        })

        print(
            "[DuplicateGuard] "
            f"Saved event remembered. "
            f"Total events: "
            f"{len(self._saved_events)}"
        )

    # =========================================================
    # COMPARE TWO BOUNDING BOXES
    # =========================================================

    def _boxes_are_same_event(
        self,
        box_a,
        box_b
    ):

        try:

            ax1, ay1, ax2, ay2 = (
                map(
                    float,
                    box_a
                )
            )

            bx1, by1, bx2, by2 = (
                map(
                    float,
                    box_b
                )
            )

        except Exception:

            return False

        # -----------------------------------------------------
        # Centers
        # -----------------------------------------------------

        acx = (
            ax1 + ax2
        ) / 2

        acy = (
            ay1 + ay2
        ) / 2

        bcx = (
            bx1 + bx2
        ) / 2

        bcy = (
            by1 + by2
        ) / 2

        # -----------------------------------------------------
        # Sizes
        # -----------------------------------------------------

        aw = max(
            ax2 - ax1,
            1
        )

        ah = max(
            ay2 - ay1,
            1
        )

        bw = max(
            bx2 - bx1,
            1
        )

        bh = max(
            by2 - by1,
            1
        )

        # -----------------------------------------------------
        # Center distance
        # -----------------------------------------------------

        dx = abs(
            acx - bcx
        )

        dy = abs(
            acy - bcy
        )

        # Allow larger movement for larger vehicles.
        max_dx = max(
            aw,
            bw
        ) * 1.5

        max_dy = max(
            ah,
            bh
        ) * 1.5

        center_close = (
            dx <= max_dx
            and
            dy <= max_dy
        )

        if center_close:

            return True

        # -----------------------------------------------------
        # IoU
        # -----------------------------------------------------

        ix1 = max(
            ax1,
            bx1
        )

        iy1 = max(
            ay1,
            by1
        )

        ix2 = min(
            ax2,
            bx2
        )

        iy2 = min(
            ay2,
            by2
        )

        if (
            ix2 <= ix1
            or
            iy2 <= iy1
        ):

            return False

        intersection = (
            ix2 - ix1
        ) * (
            iy2 - iy1
        )

        area_a = (
            ax2 - ax1
        ) * (
            ay2 - ay1
        )

        area_b = (
            bx2 - bx1
        ) * (
            by2 - by1
        )

        union = (
            area_a
            +
            area_b
            -
            intersection
        )

        if union <= 0:

            return False

        iou = (
            intersection /
            union
        )

        return iou >= 0.15

    # =========================================================
    # GENERIC VERIFICATION
    # =========================================================

    def _verify_violation(
        self,
        frame,
        violation,
        key,
        proc_frame,
        source_type="live"
    ):

        # -----------------------------------------------------
        # Already saved by track ID
        # -----------------------------------------------------

        if key in self._saved_tracks:

            return

        # -----------------------------------------------------
        # Spatial duplicate
        # -----------------------------------------------------

        if self._already_saved_event(
            violation
        ):

            return

        buf = self._verify_buf.get(
            key
        )

        # -----------------------------------------------------
        # New
        # -----------------------------------------------------

        if (
            buf is None
            or
            proc_frame
            -
            buf["first_frame"]
            >
            self.VERIFY_WINDOW
        ):

            self._verify_buf[key] = {

                "hits": 1,

                "first_frame":
                    proc_frame,

                "last_frame":
                    frame.copy(),

                "last_v":
                    violation,

                "committed":
                    False
            }

            print(
                "[Verify] New:",
                key,
                "1/",
                self.VERIFY_HITS
            )

            return

        # -----------------------------------------------------
        # Already committed
        # -----------------------------------------------------

        if buf["committed"]:

            return

        # -----------------------------------------------------
        # Add hit
        # -----------------------------------------------------

        buf["hits"] += 1

        buf["last_frame"] = (
            frame.copy()
        )

        buf["last_v"] = (
            violation
        )

        print(
            "[Verify]",
            key,
            f"{buf['hits']}/"
            f"{self.VERIFY_HITS}"
        )

        # -----------------------------------------------------
        # Confirm
        # -----------------------------------------------------

        if (
            buf["hits"]
            >=
            self.VERIFY_HITS
        ):

            # Final duplicate check.
            if self._already_saved_event(
                buf["last_v"]
            ):

                buf["committed"] = True
                return

            buf["committed"] = True

            self._saved_tracks.add(
                key
            )

            self._save_violation(
                buf["last_frame"],
                buf["last_v"],
                source_type=source_type
            )

            self._remember_saved_event(
                buf["last_v"],
                proc_frame
            )

    # =========================================================
    # SAVE VIOLATION
    # =========================================================

    def _save_violation(
        self,
        frame,
        violation,
        source_type="video"
    ):

        violation_type = (
            violation.get(
                "type",
                "unknown"
            )
        )

        plate = (
            violation.get(
                "plate",
                "UNKNOWN"
            )
        )

        snap_filename = ""

        # =====================================================
        # SNAPSHOT
        # =====================================================

        if getattr(
            cfg,
            "SAVE_SNAPS",
            True
        ):

            os.makedirs(
                cfg.SNAPSHOT_DIR,
                exist_ok=True
            )

            timestamp = (
                datetime.now()
                .strftime(
                    "%Y%m%d_%H%M%S_%f"
                )
            )

            safe_plate = str(
                plate
            ).replace(
                " ",
                "_"
            )

            filename = (
                f"{violation_type}_"
                f"{safe_plate}_"
                f"{timestamp}.jpg"
            )

            snapshot_path = (
                os.path.join(
                    cfg.SNAPSHOT_DIR,
                    filename
                )
            )

            # -------------------------------------------------
            # BBOX
            # -------------------------------------------------

            x1, y1, x2, y2 = (
                violation.get(
                    "bbox",
                    (
                        0,
                        0,
                        frame.shape[1],
                        frame.shape[0]
                    )
                )
            )

            x1 = max(
                0,
                int(x1)
            )

            y1 = max(
                0,
                int(y1)
            )

            x2 = min(
                frame.shape[1],
                int(x2)
            )

            y2 = min(
                frame.shape[0],
                int(y2)
            )

            # -------------------------------------------------
            # CONTEXT CROP
            # -------------------------------------------------

            margin = 100

            crop_x1 = max(
                0,
                x1 - margin
            )

            crop_y1 = max(
                0,
                y1 - margin
            )

            crop_x2 = min(
                frame.shape[1],
                x2 + margin
            )

            crop_y2 = min(
                frame.shape[0],
                y2 + margin
            )

            crop = frame[
                crop_y1:crop_y2,
                crop_x1:crop_x2
            ]

            if (
                crop.size > 0
                and
                cv2.imwrite(
                    snapshot_path,
                    crop
                )
            ):

                snap_filename = filename

                print(
                    "[Snapshot] Saved:",
                    snapshot_path
                )

            else:

                print(
                    "[Snapshot] ERROR: "
                    "Could not save snapshot"
                )

        # =====================================================
        # DATABASE
        # =====================================================

        if self.db:

            location = (
                "Ramanagara-Bidadi Road"
            )

            data = {

                "timestamp":
                    datetime.now(),

                "violation_type":
                    violation_type,

                "plate_number":
                    plate,

                "confidence":
                    violation.get(
                        "confidence",
                        0
                    ),

                "snapshot_path":
                    snap_filename,

                "location":
                    location,

                "source_type":
                    source_type
            }

            try:

                self.db.insert_violation(
                    data
                )

                print(
                    "[Database] "
                    "Violation inserted:",
                    violation_type,
                    plate
                )

            except Exception as e:

                print(
                    "[Database] "
                    f"Insert error: {e}"
                )

        # =====================================================
        # STATISTICS
        # =====================================================

        self.stats["total"] += 1

        if "helmet" in violation_type:

            self.stats["helmet"] += 1

        if "seatbelt" in violation_type:

            self.stats["seatbelt"] += 1

        # =====================================================
        # CALLBACK
        # =====================================================

        if self.on_violation:

            try:

                self.on_violation(
                    violation,
                    snap_filename
                )

            except Exception as e:

                print(
                    "[Detector] "
                    f"Callback error: {e}"
                )

    # =========================================================
    # HUD
    # =========================================================

    def _draw_hud(
        self,
        frame
    ):

        height, width = (
            frame.shape[:2]
        )

        overlay = frame.copy()

        cv2.rectangle(
            overlay,
            (0, 0),
            (width, 58),
            (8, 12, 18),
            -1
        )

        cv2.addWeighted(
            overlay,
            0.75,
            frame,
            0.25,
            0,
            frame
        )

        pending = sum(

            1

            for buf
            in self._verify_buf.values()

            if not buf["committed"]

        )

        cv2.putText(

            frame,

            (
                f"VIDEO "
                f"Confirmed:"
                f"{self.stats['total']} "
                f"Verifying:{pending}"
            ),

            (10, 22),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.60,

            (0, 230, 120),

            2
        )

        cv2.putText(

            frame,

            (
                f"No Helmet:"
                f"{self.stats['helmet']} "
                f"Verification x"
                f"{self.VERIFY_HITS}"
            ),

            (10, 46),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.46,

            (200, 180, 80),

            1
        )

    # =========================================================
    # VERIFY LABEL
    # =========================================================

    def _put_verify_label(
        self,
        frame,
        text,
        pos
    ):

        x, y = pos

        y = max(
            y - 8,
            12
        )

        font_scale = 0.48
        thickness = 1

        (
            tw,
            th
        ), _ = cv2.getTextSize(

            text,

            cv2.FONT_HERSHEY_SIMPLEX,

            font_scale,

            thickness
        )

        cv2.rectangle(

            frame,

            (
                x,
                y - th - 4
            ),

            (
                x + tw + 6,
                y + 4
            ),

            (0, 160, 200),

            -1
        )

        cv2.putText(

            frame,

            text,

            (
                x + 3,
                y
            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            font_scale,

            (255, 255, 255),

            thickness
        )

    # =========================================================
    # FRAME TO BASE64
    # =========================================================

    def get_frame_b64(self):

        with self._lock:

            if self._current_frame is None:

                return ""

            success, buffer = (
                cv2.imencode(
                    ".jpg",
                    self._current_frame,
                    [
                        cv2.IMWRITE_JPEG_QUALITY,
                        78
                    ]
                )
            )

            if not success:

                return ""

            return (
                base64.b64encode(
                    buffer
                ).decode()
            )