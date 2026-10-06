import cv2
import numpy as np
import os
import sys
from typing import List, Tuple, Dict

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)


class FrameProcessor:

    COLORS = {
        "violation": (0, 0, 255),
        "safe": (0, 220, 80),
        "vehicle": (255, 140, 0),
        "plate": (0, 220, 255),
        "track": (255, 0, 255),
        "person": (255, 200, 0),
        "head": (0, 165, 255),
        "unknown": (150, 150, 150),
    }

    # ---------------------------------------------------------
    # INITIALIZATION
    # ---------------------------------------------------------

    def __init__(self, confidence=0.40):

        self.confidence = confidence

        self.model = None
        self.coco_model = None
        self.plate_model = None

        # =====================================================
        # SEATBELT MODEL
        # =====================================================
        self.seatbelt_model = None

        self.custom_model = False
        self.class_map = {}

        self.tracker_name = "bytetrack.yaml"

        self._load_models()

    # ---------------------------------------------------------
    # LOAD MODELS
    # ---------------------------------------------------------

    def _load_models(self):

        try:

            from ultralytics import YOLO
            import config as cfg

            # ---------------------------------------------
            # HELMET MODEL
            # ---------------------------------------------

            model_path = getattr(
                cfg,
                "HELMET_MODEL",
                None
            )

            if not model_path:

                model_path = self._find_helmet_model()

            self.model = YOLO(model_path)

            print(
                f"[Processor] Helmet model loaded: "
                f"{model_path}"
            )

            self.custom_model = True

            self._build_class_map(
                self.model.names
            )

            # ---------------------------------------------
            # COCO MODEL
            # Person + Car + Motorcycle
            # ---------------------------------------------

            self.coco_model = YOLO(
                "yolov8n.pt"
            )

            print(
                "[Processor] COCO person/vehicle "
                "detector loaded"
            )

            print(
                "[Processor] ByteTrack enabled"
            )

            # =================================================
            # SEATBELT MODEL
            # =================================================

            seatbelt_path = os.path.join(
                os.path.dirname(
                    os.path.dirname(
                        os.path.abspath(__file__)
                    )
                ),
                "models",
                "seatbelt_best.pt"
            )

            if os.path.exists(seatbelt_path):

                try:

                    self.seatbelt_model = YOLO(
                        seatbelt_path
                    )

                    print(
                        "[Processor] Seatbelt model loaded: "
                        f"{seatbelt_path}"
                    )

                    print(
                        "[Processor] Seatbelt classes:",
                        self.seatbelt_model.names
                    )

                except Exception as e:

                    print(
                        "[Processor] WARNING: "
                        "Could not load seatbelt model:",
                        e
                    )

                    self.seatbelt_model = None

            else:

                print(
                    "[Processor] WARNING: "
                    "seatbelt_best.pt not found:"
                )

                print(
                    f"[Processor] Expected path: "
                    f"{seatbelt_path}"
                )

            # ---------------------------------------------
            # PLATE MODEL
            # ---------------------------------------------

            plate_path = getattr(
                cfg,
                "PLATE_MODEL",
                None
            )

            if (
                plate_path
                and os.path.exists(
                    str(plate_path)
                )
            ):

                self.plate_model = YOLO(
                    plate_path
                )

                print(
                    "[Processor] Plate model loaded"
                )

        except Exception as e:

            print(
                f"[Processor] Model loading error: {e}"
            )

            self.model = None
            self.coco_model = None

    # ---------------------------------------------------------
    # FIND HELMET MODEL
    # ---------------------------------------------------------

    def _find_helmet_model(self):

        from pathlib import Path

        models_dir = (
            Path(__file__).resolve()
            .parent.parent
            / "models"
        )

        possible_models = [

            "helmet_full_half.pt",

            "helmet_yolov8n.pt",

            "helmet_best.pt",

            "best.pt",

        ]

        for name in possible_models:

            path = models_dir / name

            if path.exists():

                print(
                    f"[Processor] Using helmet model: "
                    f"{path}"
                )

                return str(path)

        return "yolov8n.pt"

    # ---------------------------------------------------------
    # BUILD CLASS MAP
    # ---------------------------------------------------------

    def _build_class_map(self, names):

        self.class_map = {}

        for cid, name in names.items():

            label = (
                str(name)
                .lower()
                .strip()
            )

            # Helmet present
            if (
                "full" in label
                or "half" in label
                or label in [
                    "helmet",
                    "with helmet",
                    "with_helmet",
                    "helmet on",
                ]
            ):

                self.class_map[cid] = "yes"

            # Helmet violation
            elif (
                "invalid" in label
                or "no helmet" in label
                or "no_helmet" in label
                or "without helmet" in label
                or "without_helmet" in label
            ):

                self.class_map[cid] = "no"

            else:

                self.class_map[cid] = "unknown"

        print(
            "[Processor] Class map:",
            {
                names[k]: v
                for k, v in self.class_map.items()
            }
        )

    # ---------------------------------------------------------
    # MAIN PROCESS
    # ---------------------------------------------------------

    def process(
        self,
        frame: np.ndarray
    ) -> Tuple[np.ndarray, List[Dict]]:

        if self.model is None:

            return frame.copy(), []

        try:

            return self._run_detection(
                frame
            )

        except Exception as e:

            print(
                f"[Processor] Detection error: {e}"
            )

            import traceback

            traceback.print_exc()

            return frame.copy(), []

    # ---------------------------------------------------------
    # MAIN DETECTION
    # ---------------------------------------------------------

    def _run_detection(self, frame):

        annotated = frame.copy()

        violations = []

        motorcycles = []

        cars = []

        persons = []

        # =====================================================
        # 1. COCO PERSON + VEHICLE TRACKING
        # =====================================================

        if self.coco_model is not None:

            try:

                results = self.coco_model.track(

                    frame,

                    conf=0.20,

                    classes=[
                        0,  # person
                        2,  # car
                        3,  # motorcycle
                    ],

                    persist=True,

                    tracker=self.tracker_name,

                    verbose=False,

                    imgsz=960,

                )[0]

                boxes = results.boxes

                if boxes is not None:

                    track_ids = boxes.id

                    if track_ids is not None:

                        track_ids = (
                            track_ids
                            .int()
                            .cpu()
                            .tolist()
                        )

                    else:

                        track_ids = [
                            None
                        ] * len(boxes)

                    for i, box in enumerate(boxes):

                        cls = int(
                            box.cls[0]
                        )

                        conf = float(
                            box.conf[0]
                        )

                        x1, y1, x2, y2 = map(
                            int,
                            box.xyxy[0]
                        )

                        track_id = (
                            track_ids[i]
                        )

                        obj = {

                            "bbox": (
                                x1,
                                y1,
                                x2,
                                y2
                            ),

                            "confidence": conf,

                            "track_id": (
                                int(track_id)
                                if track_id is not None
                                else None
                            )
                        }

                        # ---------------------------------
                        # PERSON
                        # ---------------------------------

                        if cls == 0:

                            persons.append(
                                obj
                            )

                            cv2.rectangle(

                                annotated,

                                (x1, y1),

                                (x2, y2),

                                self.COLORS[
                                    "person"
                                ],

                                1

                            )

                        # ---------------------------------
                        # CAR
                        # ---------------------------------

                        elif cls == 2:

                            cars.append(
                                obj
                            )

                            cv2.rectangle(

                                annotated,

                                (x1, y1),

                                (x2, y2),

                                self.COLORS[
                                    "vehicle"
                                ],

                                1

                            )

                            self._put_vehicle_label(

                                annotated,

                                f"CAR ID:{track_id}",

                                (x1, y1)

                            )

                        # ---------------------------------
                        # MOTORCYCLE
                        # ---------------------------------

                        elif cls == 3:

                            motorcycles.append(
                                obj
                            )

                            cv2.rectangle(

                                annotated,

                                (x1, y1),

                                (x2, y2),

                                self.COLORS[
                                    "vehicle"
                                ],

                                2

                            )

                            self._put_vehicle_label(

                                annotated,

                                f"MOTORCYCLE ID:{track_id}",

                                (x1, y1)

                            )

            except Exception as e:

                print(
                    "[Processor] ByteTrack error:",
                    e
                )

        # =====================================================
        # 2. PROCESS EACH MOTORCYCLE
        # =====================================================

        for moto in motorcycles:

            rider = self._find_rider(

                moto["bbox"],

                persons

            )

            # -------------------------------------------------
            # If COCO person detector missed the rider,
            # create a fallback rider region from motorcycle.
            # -------------------------------------------------

            if rider is None:

                rider_bbox = (
                    self._estimate_rider_region(
                        moto["bbox"],
                        frame.shape
                    )
                )

                rider = {

                    "bbox": rider_bbox,

                    "confidence": 0.20,

                    "track_id": None,

                    "estimated": True

                }

                print(
                    "[Processor] Person detector missed "
                    f"rider. Using estimated rider region "
                    f"for motorcycle ID "
                    f"{moto.get('track_id')}"
                )

            else:

                rider["estimated"] = False

            # -------------------------------------------------
            # Draw rider region
            # -------------------------------------------------

            rx1, ry1, rx2, ry2 = (
                rider["bbox"]
            )

            cv2.rectangle(

                annotated,

                (rx1, ry1),

                (rx2, ry2),

                self.COLORS["person"],

                1

            )

            # -------------------------------------------------
            # Detect helmet on rider
            # -------------------------------------------------

            helmet = self._detect_helmet_on_rider(

                frame,

                rider["bbox"],

                moto["bbox"]

            )

            if helmet is None:

                continue

            helmet_status = helmet[
                "status"
            ]

            helmet_class = helmet[
                "class"
            ]

            helmet_conf = helmet[
                "confidence"
            ]

            head_bbox = helmet.get(
                "head_bbox"
            )

            # =================================================
            # SAFE HELMET
            # =================================================

            if helmet_status == "yes":

                cv2.rectangle(

                    annotated,

                    (rx1, ry1),

                    (rx2, ry2),

                    self.COLORS["safe"],

                    2

                )

                self._put_label(

                    annotated,

                    f"HELMET: {helmet_class} "
                    f"{helmet_conf:.0%}",

                    (rx1, ry1),

                    self.COLORS["safe"]

                )

            # =================================================
            # HELMET VIOLATION
            # =================================================

            elif helmet_status == "no":

                cv2.rectangle(

                    annotated,

                    (rx1, ry1),

                    (rx2, ry2),

                    self.COLORS["violation"],

                    3

                )

                self._put_label(

                    annotated,

                    f"NO HELMET "
                    f"{helmet_conf:.0%}",

                    (rx1, ry1),

                    self.COLORS["violation"]

                )

                # ---------------------------------------------
                # Draw head detection too
                # ---------------------------------------------

                if head_bbox is not None:

                    hx1, hy1, hx2, hy2 = (
                        head_bbox
                    )

                    cv2.rectangle(

                        annotated,

                        (hx1, hy1),

                        (hx2, hy2),

                        self.COLORS[
                            "violation"
                        ],

                        2

                    )

                # ---------------------------------------------
                # Create violation record
                # ---------------------------------------------

                violations.append({

                    "type":
                        "no_helmet",

                    "bbox": (

                        rx1,

                        ry1,

                        rx2,

                        ry2

                    ),

                    "confidence":
                        helmet_conf,

                    "plate":
                        "UNKNOWN",

                    "track_id":
                        moto.get(
                            "track_id"
                        ),

                    "helmet_type":
                        helmet_class

                })

        # =====================================================
        # 3. SEATBELT DETECTION ON CARS
        # =====================================================

        for car in cars:

            car_bbox = car["bbox"]

            seatbelt = (
                self._detect_seatbelt_on_car(
                    frame,
                    car_bbox
                )
            )

            if seatbelt is None:

                continue

            seatbelt_class = (
                seatbelt["class"]
            )

            seatbelt_conf = (
                seatbelt["confidence"]
            )

            seatbelt_bbox = (
                seatbelt["bbox"]
            )

            sx1, sy1, sx2, sy2 = (
                seatbelt_bbox
            )

            # =================================================
            # SEATBELT PRESENT
            # =================================================

            if "with_seatbelt" in seatbelt_class:

                cv2.rectangle(

                    annotated,

                    (sx1, sy1),

                    (sx2, sy2),

                    self.COLORS["safe"],

                    2

                )

                self._put_label(

                    annotated,

                    f"SEATBELT "
                    f"{seatbelt_conf:.0%}",

                    (sx1, sy1),

                    self.COLORS["safe"]

                )

            # =================================================
            # NO SEATBELT
            # =================================================

            elif "without_seatbelt" in seatbelt_class:

                cx1, cy1, cx2, cy2 = (
                    car_bbox
                )

                cv2.rectangle(

                    annotated,

                    (cx1, cy1),

                    (cx2, cy2),

                    self.COLORS["violation"],

                    3

                )

                self._put_label(

                    annotated,

                    f"NO SEATBELT "
                    f"{seatbelt_conf:.0%}",

                    (cx1, cy1),

                    self.COLORS["violation"]

                )

                # ---------------------------------------------
                # Create seatbelt violation
                # ---------------------------------------------

                violations.append({

                    "type":
                        "no_seatbelt",

                    "bbox": (

                        cx1,

                        cy1,

                        cx2,

                        cy2

                    ),

                    "confidence":
                        seatbelt_conf,

                    "plate":
                        "UNKNOWN",

                    "track_id":
                        car.get(
                            "track_id"
                        ),

                    "seatbelt_type":
                        seatbelt_class

                })

        # =====================================================
        # 4. PLATE READING
        # =====================================================

        if violations:

            try:

                from utils.plate_reader import (
                    PlateReader
                )

                reader = PlateReader()

                for violation in violations:

                    plate = (
                        reader.read_from_region(

                            frame,

                            violation["bbox"],

                            expand=80

                        )
                    )

                    if plate:

                        violation[
                            "plate"
                        ] = plate

            except Exception as e:

                print(
                    "[Processor] Plate reader:",
                    e
                )

        return annotated, violations

    # =========================================================
    # SEATBELT DETECTION ON CAR
    # =========================================================

    def _detect_seatbelt_on_car(
        self,
        frame,
        car_bbox
    ):

        if self.seatbelt_model is None:

            return None

        x1, y1, x2, y2 = (
            car_bbox
        )

        frame_h, frame_w = (
            frame.shape[:2]
        )

        # -----------------------------------------------------
        # Keep coordinates inside frame
        # -----------------------------------------------------

        x1 = max(
            0,
            int(x1)
        )

        y1 = max(
            0,
            int(y1)
        )

        x2 = min(
            frame_w,
            int(x2)
        )

        y2 = min(
            frame_h,
            int(y2)
        )

        if x2 <= x1 or y2 <= y1:

            return None

        # -----------------------------------------------------
        # Extract car
        # -----------------------------------------------------

        car_crop = frame[
            y1:y2,
            x1:x2
        ]

        if car_crop.size == 0:

            return None

        try:

            results = self.seatbelt_model(

                car_crop,

                conf=0.40,

                imgsz=640,

                verbose=False

            )[0]

        except Exception as e:

            print(
                "[Processor] Seatbelt detection "
                f"error: {e}"
            )

            return None

        if results.boxes is None:

            return None

        detections = []

        # -----------------------------------------------------
        # Read seatbelt detections
        # -----------------------------------------------------

        for box in results.boxes:

            cls = int(
                box.cls[0]
            )

            conf = float(
                box.conf[0]
            )

            bx1, by1, bx2, by2 = map(

                int,

                box.xyxy[0]

            )

            label = str(

                self.seatbelt_model.names.get(
                    cls,
                    cls
                )

            ).lower().strip()

            # -------------------------------------------------
            # Convert crop coordinates into
            # original video coordinates
            # -------------------------------------------------

            original_bbox = (

                x1 + bx1,

                y1 + by1,

                x1 + bx2,

                y1 + by2

            )

            detections.append({

                "class":
                    label,

                "confidence":
                    conf,

                "bbox":
                    original_bbox

            })

        if not detections:

            return None

        # -----------------------------------------------------
        # Select highest-confidence detection
        # -----------------------------------------------------

        detections.sort(

            key=lambda d:
                d["confidence"],

            reverse=True

        )

        return detections[0]

    # =========================================================
    # FIND RIDER
    # =========================================================

    def _find_rider(
        self,
        motorcycle_bbox,
        persons
    ):

        if not persons:

            return None

        mx1, my1, mx2, my2 = (
            motorcycle_bbox
        )

        mw = max(
            mx2 - mx1,
            1
        )

        mh = max(
            my2 - my1,
            1
        )

        motorcycle_center_x = (
            mx1 + mx2
        ) / 2

        candidates = []

        # Search a generous region around motorcycle.
        zone_x1 = (
            mx1 - int(mw * 0.45)
        )

        zone_x2 = (
            mx2 + int(mw * 0.45)
        )

        zone_y1 = (
            my1 - int(mh * 2.5)
        )

        zone_y2 = (
            my2 + int(mh * 0.20)
        )

        for person in persons:

            px1, py1, px2, py2 = (
                person["bbox"]
            )

            pcx = (
                px1 + px2
            ) / 2

            pcy = (
                py1 + py2
            ) / 2

            # Center must be near motorcycle.
            if not (
                zone_x1 <= pcx <= zone_x2
                and
                zone_y1 <= pcy <= zone_y2
            ):

                continue

            # Horizontal distance.
            horizontal_distance = abs(
                pcx -
                motorcycle_center_x
            )

            horizontal_score = max(

                0.0,

                1.0 -
                horizontal_distance /
                max(mw * 1.5, 1)

            )

            # Person should overlap/approach
            # the motorcycle area.
            overlap = self._bbox_overlap_ratio(

                person["bbox"],

                motorcycle_bbox

            )

            # Prefer person whose lower body
            # is close to motorcycle.
            lower_distance = abs(
                py2 - my1
            )

            vertical_score = max(

                0.0,

                1.0 -
                lower_distance /
                max(mh * 2.0, 1)

            )

            score = (

                horizontal_score * 0.45

                +

                overlap * 0.30

                +

                vertical_score * 0.25

            )

            candidates.append(
                (
                    score,
                    person
                )
            )

        if not candidates:

            return None

        candidates.sort(
            key=lambda x: x[0],
            reverse=True
        )

        best_score, best_person = (
            candidates[0]
        )

        # Avoid unrelated people.
        if best_score < 0.20:

            return None

        return best_person

    # =========================================================
    # FALLBACK RIDER REGION
    # =========================================================

    def _estimate_rider_region(
        self,
        motorcycle_bbox,
        frame_shape
    ):

        frame_h, frame_w = (
            frame_shape[:2]
        )

        mx1, my1, mx2, my2 = (
            motorcycle_bbox
        )

        mw = max(
            mx2 - mx1,
            1
        )

        mh = max(
            my2 - my1,
            1
        )

        # Rider normally sits above motorcycle.
        x1 = int(
            mx1 - mw * 0.20
        )

        x2 = int(
            mx2 + mw * 0.20
        )

        # Extend substantially upward
        # for small/distant riders.
        y1 = int(
            my1 - mh * 1.30
        )

        y2 = int(
            my2 + mh * 0.05
        )

        x1 = max(
            0,
            x1
        )

        y1 = max(
            0,
            y1
        )

        x2 = min(
            frame_w - 1,
            x2
        )

        y2 = min(
            frame_h - 1,
            y2
        )

        return (
            x1,
            y1,
            x2,
            y2
        )

    # =========================================================
    # HELMET DETECTION ON RIDER
    # =========================================================

    def _detect_helmet_on_rider(
        self,
        frame,
        rider_bbox,
        motorcycle_bbox
    ):

        if self.model is None:

            return None

        x1, y1, x2, y2 = (
            rider_bbox
        )

        frame_h, frame_w = (
            frame.shape[:2]
        )

        x1 = max(
            0,
            x1
        )

        y1 = max(
            0,
            y1
        )

        x2 = min(
            frame_w,
            x2
        )

        y2 = min(
            frame_h,
            y2
        )

        if x2 <= x1 or y2 <= y1:

            return None

        rider_crop = frame[
            y1:y2,
            x1:x2
        ]

        if rider_crop.size == 0:

            return None

        crop_h, crop_w = (
            rider_crop.shape[:2]
        )

        # -----------------------------------------------------
        # HEAD REGION
        # -----------------------------------------------------

        # The head is normally in the upper part
        # of the rider bounding box.

        head_y2 = max(

            int(
                crop_h * 0.60
            ),

            20

        )

        head_crop = rider_crop[
            0:head_y2,
            :
        ]

        if head_crop.size == 0:

            return None

        # -----------------------------------------------------
        # UPSCALE SMALL RIDERS
        # -----------------------------------------------------

        target_width = 640

        if crop_w < target_width:

            scale = (
                target_width /
                max(crop_w, 1)
            )

            new_w = int(
                crop_w * scale
            )

            new_h = int(
                crop_h * scale
            )

            rider_crop = cv2.resize(

                rider_crop,

                (
                    new_w,
                    new_h
                ),

                interpolation=cv2.INTER_CUBIC

            )

            head_crop = rider_crop[
                0:int(new_h * 0.60),
                :
            ]

        # -----------------------------------------------------
        # RUN HELMET MODEL
        # -----------------------------------------------------

        detections = []

        try:

            results = self.model(

                rider_crop,

                conf=0.12,

                imgsz=640,

                verbose=False

            )[0]

            if results.boxes is not None:

                for box in results.boxes:

                    cls = int(
                        box.cls[0]
                    )

                    conf = float(
                        box.conf[0]
                    )

                    hx1, hy1, hx2, hy2 = map(

                        int,

                        box.xyxy[0]

                    )

                    label = str(

                        self.model.names.get(
                            cls,
                            cls
                        )

                    )

                    status = self.class_map.get(

                        cls,

                        "unknown"

                    )

                    detections.append({

                        "status":
                            status,

                        "class":
                            label,

                        "confidence":
                            conf,

                        "bbox": (

                            hx1,

                            hy1,

                            hx2,

                            hy2

                        )

                    })

        except Exception as e:

            print(
                "[Processor] Rider helmet "
                f"detection error: {e}"
            )

        # -----------------------------------------------------
        # IF FULL RIDER CROP DID NOT FIND ANYTHING,
        # RUN DIRECTLY ON HEAD CROP.
        # -----------------------------------------------------

        if not detections:

            try:

                head_results = self.model(

                    head_crop,

                    conf=0.10,

                    imgsz=640,

                    verbose=False

                )[0]

                if (
                    head_results.boxes
                    is not None
                ):

                    for box in (
                        head_results.boxes
                    ):

                        cls = int(
                            box.cls[0]
                        )

                        conf = float(
                            box.conf[0]
                        )

                        hx1, hy1, hx2, hy2 = map(

                            int,

                            box.xyxy[0]

                        )

                        label = str(

                            self.model.names.get(
                                cls,
                                cls
                            )

                        )

                        status = (
                            self.class_map.get(
                                cls,
                                "unknown"
                            )
                        )

                        detections.append({

                            "status":
                                status,

                            "class":
                                label,

                            "confidence":
                                conf,

                            "bbox": (

                                hx1,

                                hy1,

                                hx2,

                                hy2

                            )

                        })

            except Exception as e:

                print(
                    "[Processor] Head helmet "
                    f"detection error: {e}"
                )

        # -----------------------------------------------------
        # NO DETECTION
        # -----------------------------------------------------

        if not detections:

            return None

        # -----------------------------------------------------
        # REMOVE VERY WEAK / UNKNOWN RESULTS
        # -----------------------------------------------------

        detections = [

            d for d in detections

            if d["status"]
            in ("yes", "no")

        ]

        if not detections:

            return None

        # -----------------------------------------------------
        # SELECT BEST DETECTION
        # -----------------------------------------------------

        detections.sort(

            key=lambda d:
                d["confidence"],

            reverse=True

        )

        best = detections[0]

        # -----------------------------------------------------
        # CONVERT HEAD BOX BACK TO ORIGINAL FRAME
        # -----------------------------------------------------

        bx1, by1, bx2, by2 = (
            best["bbox"]
        )

        original_crop_w = (
            x2 - x1
        )

        # Because the crop may have been
        # upscaled, calculate scale.
        if original_crop_w > 0:

            scale_x = (
                original_crop_w /
                max(
                    rider_crop.shape[1],
                    1
                )
            )

        else:

            scale_x = 1.0

        # For practical evidence,
        # use rider upper region rather than
        # relying on tiny model box.
        head_original_x1 = int(
            x1 +
            bx1 * scale_x
        )

        head_original_x2 = int(
            x1 +
            bx2 * scale_x
        )

        original_crop_h = (
            y2 - y1
        )

        if original_crop_h > 0:

            scale_y = (
                original_crop_h /
                max(
                    rider_crop.shape[0],
                    1
                )
            )

        else:

            scale_y = 1.0

        head_original_y1 = int(
            y1 +
            by1 * scale_y
        )

        head_original_y2 = int(
            y1 +
            by2 * scale_y
        )

        head_original_x1 = max(
            0,
            min(
                frame_w - 1,
                head_original_x1
            )
        )

        head_original_y1 = max(
            0,
            min(
                frame_h - 1,
                head_original_y1
            )
        )

        head_original_x2 = max(
            0,
            min(
                frame_w - 1,
                head_original_x2
            )
        )

        head_original_y2 = max(
            0,
            min(
                frame_h - 1,
                head_original_y2
            )
        )

        best["head_bbox"] = (

            head_original_x1,

            head_original_y1,

            head_original_x2,

            head_original_y2

        )

        return best

    # =========================================================
    # BBOX OVERLAP
    # =========================================================

    def _bbox_overlap_ratio(
        self,
        box_a,
        box_b
    ):

        ax1, ay1, ax2, ay2 = box_a

        bx1, by1, bx2, by2 = box_b

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

        if ix2 <= ix1 or iy2 <= iy1:

            return 0.0

        intersection = (
            ix2 - ix1
        ) * (
            iy2 - iy1
        )

        area_a = max(
            (ax2 - ax1) *
            (ay2 - ay1),
            1
        )

        return (
            intersection /
            area_a
        )

    # =========================================================
    # VEHICLE LABEL
    # =========================================================

    def _put_vehicle_label(
        self,
        frame,
        text,
        pos
    ):

        x, y = pos

        cv2.putText(

            frame,

            text,

            (
                x,
                max(
                    y - 5,
                    15
                )
            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.42,

            self.COLORS["track"],

            1

        )

    # =========================================================
    # GENERAL LABEL
    # =========================================================

    def _put_label(
        self,
        frame,
        text,
        pos,
        color
    ):

        x, y = pos

        font_scale = 0.50

        thickness = 2

        (
            tw,
            th
        ), _ = cv2.getTextSize(

            text,

            cv2.FONT_HERSHEY_SIMPLEX,

            font_scale,

            thickness

        )

        y = max(
            y,
            th + 5
        )

        cv2.rectangle(

            frame,

            (
                x,
                y - th - 5
            ),

            (
                x + tw + 6,
                y + 4
            ),

            color,

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