"""ViolaWatch — Core Violation Detector Engine"""

import cv2
import os
import time
import threading
import base64

from datetime import datetime
from pathlib import Path
import sys

# ---------------------------------------------------------
# PROJECT PATH
# ---------------------------------------------------------

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.frame_processor import FrameProcessor
import config as cfg


class ViolationDetector:

    # =========================================================
    # INIT
    # =========================================================

    def __init__(self, db=None, on_violation=None):

        self.db = db
        self.on_violation = on_violation

        self.processor = FrameProcessor(cfg.CONFIDENCE)

        self.is_running = False
        self.cap = None

        self._lock = threading.Lock()
        self._current_frame = None

        self.stats = {
            "fps": 0,
            "total": 0,
            "helmet": 0,
            "seatbelt": 0,
            "frames": 0
        }

        # =====================================================
        # TRACK / VIOLATION MEMORY
        # =====================================================

        self._saved_tracks = set()

        # =====================================================
        # VERIFICATION BUFFER
        # =====================================================

        self._verify_buf = {}

        self.VERIFY_HITS = 3
        self.VERIFY_WINDOW = 30

        self.GRID = 120

        self.COOLDOWN = max(
            getattr(cfg, "COOLDOWN", 15),
            30
        )

    # =========================================================
    # LIVE CAMERA
    # =========================================================

    def start_live(self, source=0):

        self.cap = cv2.VideoCapture(source)

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

        print("[Detector] Stop requested")

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

                if time.time() - fps_time >= 1.0:

                    self.stats["fps"] = fps_count

                    fps_count = 0
                    fps_time = time.time()

                if frame_count % cfg.FRAME_SKIP != 0:
                    continue

                proc_frame += 1
                self.stats["frames"] += 1

                annotated, violations = \
                    self.processor.process(frame)

                self._draw_hud(annotated)

                for violation in violations:

                    if not self.is_running:
                        break

                    key = self._make_track_key(
                        violation
                    )

                    self._verify_violation(
                        frame,
                        violation,
                        key,
                        proc_frame,
                        source_type="live"
                    )

                with self._lock:
                    self._current_frame = \
                        annotated.copy()

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
    # VIDEO FILE
    # =========================================================

    def process_video_file(
        self,
        path,
        job_id=None,
        progress_cb=None,
        done_cb=None
    ):

        self.is_running = True

        self._saved_tracks.clear()
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
            f"[Detector] Video processing started: {path}"
        )

    # =========================================================
    # VIDEO FILE LOOP
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

        try:

            # =================================================
            # OPEN VIDEO
            # =================================================

            cap = cv2.VideoCapture(path)

            if not cap.isOpened():

                print(
                    "[Detector] ERROR: Cannot open video"
                )

                if done_cb:
                    done_cb(
                        0,
                        "Cannot open video"
                    )

                return

            self.cap = cap

            # =================================================
            # VIDEO INFORMATION
            # =================================================

            total_frames = int(
                cap.get(
                    cv2.CAP_PROP_FRAME_COUNT
                )
            )

            fps = cap.get(
                cv2.CAP_PROP_FPS
            ) or 25

            cooldown_frames = int(
                fps * max(
                    getattr(cfg, "COOLDOWN", 15),
                    15
                )
            )

            # =================================================
            # DATABASE JOB
            # =================================================

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

            # =================================================
            # MEMORY
            # =================================================

            frame_seen = {}

            self._verify_buf.clear()

            processed_frames = 0

            skip = max(
                getattr(cfg, "FRAME_SKIP", 5),
                5
            )

            print(
                "[Detector] Video processing started"
            )

            print(
                f"[Detector] Total frames: "
                f"{total_frames}"
            )

            print(
                f"[Detector] FPS: {fps}"
            )

            print(
                f"[Detector] Frame skip: {skip}"
            )

            # =================================================
            # MAIN VIDEO LOOP
            # =================================================

            while True:

                # -------------------------------------------------
                # STOP CHECK
                # -------------------------------------------------

                if not self.is_running:

                    print(
                        "[Detector] "
                        "Video processing stopped by user"
                    )

                    break

                # -------------------------------------------------
                # READ FRAME
                # -------------------------------------------------

                ret, frame = cap.read()

                if not ret:
                    break

                frame_count += 1

                # -------------------------------------------------
                # FRAME SKIP
                # -------------------------------------------------

                if frame_count % skip != 0:
                    continue

                processed_frames += 1

                self.stats["frames"] = processed_frames

                # -------------------------------------------------
                # STOP CHECK BEFORE AI
                # -------------------------------------------------

                if not self.is_running:

                    print(
                        "[Detector] "
                        "Stop requested before AI processing"
                    )

                    break

                # =================================================
                # AI PROCESSING
                # =================================================

                try:

                    annotated, violations = \
                        self.processor.process(frame)

                except Exception as e:

                    print(
                        "[Detector] "
                        f"Frame processing error: {e}"
                    )

                    continue

                # =================================================
                # UPDATE VIDEO PREVIEW
                # =================================================

                with self._lock:

                    self._current_frame = \
                        annotated.copy()

                # =================================================
                # DRAW HUD
                # =================================================

                self._draw_hud(
                    annotated
                )

                with self._lock:

                    self._current_frame = \
                        annotated.copy()

                # =================================================
                # PROCESS VIOLATIONS
                # =================================================

                for violation in violations:

                    if not self.is_running:

                        print(
                            "[Detector] "
                            "Stop requested during violation processing"
                        )

                        break

                    # -------------------------------------------------
                    # CREATE TRACK KEY
                    # -------------------------------------------------

                    key = self._make_track_key(
                        violation
                    )

                    # -------------------------------------------------
                    # COOLDOWN CHECK
                    # -------------------------------------------------

                    last_seen = frame_seen.get(
                        key,
                        -cooldown_frames - 1
                    )

                    if (
                        frame_count - last_seen
                        < cooldown_frames
                    ):

                        continue

                    # -------------------------------------------------
                    # GET VERIFICATION BUFFER
                    # -------------------------------------------------

                    buf = self._verify_buf.get(
                        key
                    )

                    # -------------------------------------------------
                    # NEW VIOLATION
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
                            "[Verify] "
                            f"New: {key} "
                            f"hit=1/"
                            f"{self.VERIFY_HITS}"
                        )

                        continue

                    # -------------------------------------------------
                    # ALREADY SAVED
                    # -------------------------------------------------

                    if buf["committed"]:
                        continue

                    # -------------------------------------------------
                    # VERIFICATION WINDOW
                    # -------------------------------------------------

                    if (
                        processed_frames
                        - buf["first_frame"]
                        > self.VERIFY_WINDOW
                    ):

                        buf["hits"] = 1

                        buf["first_frame"] = \
                            processed_frames

                        buf["last_frame"] = \
                            frame.copy()

                        buf["last_v"] = \
                            violation

                        buf["committed"] = False

                        print(
                            "[Verify] "
                            f"Window reset: {key}"
                        )

                        continue

                    # -------------------------------------------------
                    # ADD HIT
                    # -------------------------------------------------

                    buf["hits"] += 1

                    buf["last_frame"] = \
                        frame.copy()

                    buf["last_v"] = \
                        violation

                    print(
                        "[Verify] "
                        f"track={key} "
                        f"hit={buf['hits']}/"
                        f"{self.VERIFY_HITS}"
                    )

                    # -------------------------------------------------
                    # CONFIRM VIOLATION
                    # -------------------------------------------------

                    if (
                        buf["hits"]
                        >= self.VERIFY_HITS
                    ):

                        if buf["committed"]:
                            continue

                        buf["committed"] = True

                        self._saved_tracks.add(
                            key
                        )

                        frame_seen[key] = \
                            frame_count

                        print(
                            "[Verify] "
                            f"CONFIRMED: {key}"
                        )

                        self._save_violation(
                            buf["last_frame"],
                            buf["last_v"],
                            source_type="video"
                        )

                        violations_count += 1

                # =================================================
                # REMOVE OLD VERIFICATION BUFFERS
                # =================================================

                old_keys = []

                for key, buf in \
                        self._verify_buf.items():

                    if (
                        processed_frames
                        - buf["first_frame"]
                        > self.VERIFY_WINDOW * 2
                    ):

                        old_keys.append(key)

                for key in old_keys:

                    try:
                        del self._verify_buf[key]
                    except KeyError:
                        pass

                # =================================================
                # PROGRESS CALLBACK
                # =================================================

                if (
                    progress_cb
                    and frame_count % (skip * 10) == 0
                ):

                    percent = int(
                        (
                            frame_count
                            / max(total_frames, 1)
                        ) * 100
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
                    and frame_count % (skip * 30) == 0
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
            # VIDEO LOOP FINISHED
            # =====================================================

            if not self.is_running:

                print(
                    "[Detector] "
                    "Video stopped by user."
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

                    except Exception as e:

                        print(
                            "[Detector] "
                            "Stop DB update error:",
                            e
                        )

                if done_cb:

                    done_cb(
                        violations_count,
                        "Stopped by user"
                    )

                return

            # =====================================================
            # NORMAL COMPLETION
            # =====================================================

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
                f"Unique tracks saved: "
                f"{len(self._saved_tracks)}"
            )

            if self.db and job_id:

                try:

                    self.db.update_job(
                        job_id,
                        status="done",
                        processed=total_frames,
                        violations_found=
                            violations_count,
                        completed_at=datetime.now()
                    )

                except Exception as e:

                    print(
                        "[Detector] "
                        "Completion DB update error:",
                        e
                    )

            if done_cb:

                done_cb(
                    violations_count,
                    None
                )

        except Exception as e:

            # =================================================
            # UNEXPECTED ERROR
            # =================================================

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

            # =================================================
            # ALWAYS RELEASE VIDEO
            # =================================================

            if cap is not None:

                try:
                    cap.release()
                except Exception:
                    pass

            self.cap = None

            self.is_running = False

            print(
                "[Detector] Video capture released"
            )

    # =========================================================
    # CREATE TRACK KEY
    # =========================================================

    def _make_track_key(self, violation):

        """
        Create a unique key using:

            violation type
            +
            ByteTrack track ID

        Example:

            ("no_helmet", 1)
            ("no_helmet", 2)

        If track_id is unavailable,
        bounding-box position is used as fallback.
        """

        violation_type = violation.get(
            "type",
            "unknown"
        )

        track_id = violation.get(
            "track_id"
        )

        # -----------------------------------------------------
        # NORMAL CASE — ByteTrack ID
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
        # FALLBACK
        # -----------------------------------------------------

        x1, y1, x2, y2 = violation.get(
            "bbox",
            (0, 0, 100, 100)
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
    # VIOLATION VERIFICATION
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
        # Already saved
        # -----------------------------------------------------

        if key in self._saved_tracks:
            return

        # -----------------------------------------------------
        # Get buffer
        # -----------------------------------------------------

        buf = self._verify_buf.get(
            key
        )

        # -----------------------------------------------------
        # NEW VIOLATION
        # -----------------------------------------------------

        if (
            buf is None
            or proc_frame
            - buf["first_frame"]
            > self.VERIFY_WINDOW
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
                "[Verify] "
                f"New violation: {key} "
                f"hit=1/"
                f"{self.VERIFY_HITS}"
            )

            return

        # -----------------------------------------------------
        # Already committed
        # -----------------------------------------------------

        if buf["committed"]:
            return

        # -----------------------------------------------------
        # ADD HIT
        # -----------------------------------------------------

        buf["hits"] += 1

        buf["last_frame"] = \
            frame.copy()

        buf["last_v"] = \
            violation

        print(
            "[Verify] "
            f"{key} "
            f"hit={buf['hits']}/"
            f"{self.VERIFY_HITS}"
        )

        # -----------------------------------------------------
        # CONFIRM
        # -----------------------------------------------------

        if (
            buf["hits"]
            >= self.VERIFY_HITS
        ):

            buf["committed"] = True

            self._saved_tracks.add(
                key
            )

            print(
                "[Verify] "
                f"CONFIRMED: {key}"
            )

            self._save_violation(
                buf["last_frame"],
                buf["last_v"],
                source_type=source_type
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

        violation_type = violation.get(
            "type",
            "unknown"
        )

        plate = violation.get(
            "plate",
            "UNKNOWN"
        )

        snap_filename = ""

        # =====================================================
        # SAVE SNAPSHOT
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

            timestamp = datetime.now().strftime(
                "%Y%m%d_%H%M%S_%f"
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

            snapshot_path = os.path.join(
                cfg.SNAPSHOT_DIR,
                filename
            )

            # -------------------------------------------------
            # ORIGINAL BOUNDING BOX
            # -------------------------------------------------

            x1, y1, x2, y2 = violation.get(
                "bbox",
                (
                    0,
                    0,
                    frame.shape[1],
                    frame.shape[0]
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
            # LARGER CONTEXT CROP
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

            # -------------------------------------------------
            # SAVE
            # -------------------------------------------------

            if (
                crop.size > 0
                and cv2.imwrite(
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

            # -------------------------------------------------
            # FIXED LOCATION
            # -------------------------------------------------

            location = "Ramanagara-Bidadi Road"

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

                print(
                    "[Database] "
                    "Location:",
                    location
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
                    f"Violation callback error: {e}"
                )

    # =========================================================
    # HUD
    # =========================================================

    def _draw_hud(self, frame):

        height, width = frame.shape[:2]

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
            for buf in self._verify_buf.values()
            if not buf["committed"]
        )

        cv2.putText(
            frame,
            (
                f"VIDEO  "
                f"Confirmed:{self.stats['total']}  "
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
                f"No Helmet:{self.stats['helmet']}  "
                f"Verification x{self.VERIFY_HITS}"
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

        (tw, th), _ = cv2.getTextSize(
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
    # CURRENT FRAME → BASE64
    # =========================================================

    def get_frame_b64(self) -> str:

        with self._lock:

            if self._current_frame is None:
                return ""

            success, buffer = cv2.imencode(
                ".jpg",
                self._current_frame,
                [
                    cv2.IMWRITE_JPEG_QUALITY,
                    78
                ]
            )

            if not success:
                return ""

            return base64.b64encode(
                buffer
            ).decode()