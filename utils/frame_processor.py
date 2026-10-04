import cv2
import numpy as np
import os
import sys
from typing import List, Tuple, Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class FrameProcessor:

    COLORS = {
        "violation": (0, 0, 255),
        "safe": (0, 220, 80),
        "vehicle": (255, 140, 0),
        "plate": (0, 220, 255),
        "track": (255, 0, 255),
        "passenger": (120, 120, 120),
    }

    def __init__(self, confidence=0.40):
        self.confidence = confidence

        self.model = None
        self.coco_model = None
        self.plate_model = None

        self.custom_model = False
        self.class_map = {}

        # ByteTrack
        self.tracker_name = "bytetrack.yaml"

        self._load_models()

    # ---------------------------------------------------------
    # MODEL LOADING
    # ---------------------------------------------------------

    def _load_models(self):

        try:
            from ultralytics import YOLO
            import config as cfg

            model_path = getattr(cfg, "HELMET_MODEL", None)

            if not model_path:
                model_path = self._find_helmet_model()

            self.model = YOLO(model_path)

            print(f"[Processor] Helmet model loaded: {model_path}")

            self.custom_model = True

            self._build_class_map(self.model.names)

            # COCO model for vehicle + person detection
            self.coco_model = YOLO("yolov8n.pt")

            print("[Processor] COCO vehicle detector loaded")
            print("[Processor] ByteTrack enabled")

            # Plate model is kept compatible with the original project
            plate_path = getattr(cfg, "PLATE_MODEL", None)

            if plate_path and os.path.exists(str(plate_path)):
                self.plate_model = YOLO(plate_path)

        except Exception as e:

            print(f"[Processor] Model loading error: {e}")

            self.model = None
            self.coco_model = None

    # ---------------------------------------------------------
    # FIND HELMET MODEL
    # ---------------------------------------------------------

    def _find_helmet_model(self):

        from pathlib import Path

        models_dir = (
            Path(__file__).resolve().parent.parent / "models"
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
                print(f"[Processor] Using helmet model: {path}")
                return str(path)

        return "yolov8n.pt"

    # ---------------------------------------------------------
    # BUILD CLASS MAP
    # ---------------------------------------------------------

    def _build_class_map(self, names):

        self.class_map = {}

        for cid, name in names.items():

            label = str(name).lower().strip()

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
    # MAIN PROCESS FUNCTION
    # ---------------------------------------------------------

    def process(
        self,
        frame: np.ndarray
    ) -> Tuple[np.ndarray, List[Dict]]:

        if self.model is None:

            return frame.copy(), []

        try:

            return self._run_detection(frame)

        except Exception as e:

            print(f"[Processor] Detection error: {e}")

            import traceback
            traceback.print_exc()

            return frame.copy(), []

    # ---------------------------------------------------------
    # DETECTION
    # ---------------------------------------------------------

    def _run_detection(self, frame):

        annotated = frame.copy()

        violations = []

        # =====================================================
        # 1. HELMET MODEL
        # =====================================================

        helmet_results = self.model(
            frame,
            conf=0.30,
            verbose=False
        )[0]

        helmet_objects = []

        for box in helmet_results.boxes:

            cls = int(box.cls[0])

            conf = float(box.conf[0])

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )

            helmet_label = str(
                self.model.names.get(cls, cls)
            )

            helmet_status = self.class_map.get(
                cls,
                "unknown"
            )

            helmet_objects.append({

                "bbox": (
                    x1,
                    y1,
                    x2,
                    y2
                ),

                "confidence": conf,

                "helmet": (
                    True
                    if helmet_status == "yes"
                    else False
                    if helmet_status == "no"
                    else None
                ),

                "helmet_class": helmet_label
            })

        # =====================================================
        # 2. BYTE TRACK VEHICLES
        # =====================================================

        motorcycles = []

        cars = []

        if self.coco_model is not None:

            try:

                results = self.coco_model.track(

                    frame,

                    conf=0.30,

                    classes=[
                        2,  # car
                        3   # motorcycle
                    ],

                    persist=True,

                    tracker=self.tracker_name,

                    verbose=False

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

                        cls = int(box.cls[0])

                        conf = float(
                            box.conf[0]
                        )

                        x1, y1, x2, y2 = map(
                            int,
                            box.xyxy[0]
                        )

                        track_id = track_ids[i]

                        vehicle = {

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

                        # Motorcycle
                        if cls == 3:

                            motorcycles.append(
                                vehicle
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

                        # Car
                        elif cls == 2:

                            cars.append(
                                vehicle
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

            except Exception as e:

                print(
                    "[Processor] ByteTrack error:",
                    e
                )

        # =====================================================
        # 3. ASSOCIATE HELMET WITH MOTORCYCLE
        # =====================================================

        used_helmet = set()

        for moto in motorcycles:

            riders = self._heads_on_moto(

                moto["bbox"],

                helmet_objects

            )

            if not riders:

                continue

            rider = self._identify_rider(

                moto["bbox"],

                riders

            )

            if rider is None:

                continue

            rider_index = id(rider)

            used_helmet.add(
                rider_index
            )

            rx1, ry1, rx2, ry2 = (
                rider["bbox"]
            )

            helmet_status = rider.get(
                "helmet"
            )

            helmet_class = rider.get(
                "helmet_class",
                ""
            )

            # -------------------------------------------------
            # HELMET PRESENT
            # -------------------------------------------------

            if helmet_status is True:

                cv2.rectangle(

                    annotated,

                    (rx1, ry1),

                    (rx2, ry2),

                    self.COLORS["safe"],

                    3
                )

                self._put_label(

                    annotated,

                    f"HELMET: {helmet_class}",

                    (rx1, ry1),

                    self.COLORS["safe"]
                )

            # -------------------------------------------------
            # NO / INVALID HELMET
            # -------------------------------------------------

            elif helmet_status is False:

                cv2.rectangle(

                    annotated,

                    (rx1, ry1),

                    (rx2, ry2),

                    self.COLORS["violation"],

                    3
                )

                self._put_label(

                    annotated,

                    "NO HELMET",

                    (rx1, ry1),

                    self.COLORS["violation"]
                )

                # IMPORTANT:
                # track_id belongs to the motorcycle.
                #
                # detector.py will use this ID to make sure
                # the same rider is not saved repeatedly.

                violations.append({

                    "type": "no_helmet",

                    "bbox": (
                        rx1,
                        ry1,
                        rx2,
                        ry2
                    ),

                    "confidence": rider[
                        "confidence"
                    ],

                    "plate": "UNKNOWN",

                    "track_id": moto.get(
                        "track_id"
                    ),

                    "helmet_type": helmet_class
                })

        # =====================================================
        # 4. DRAW UNASSOCIATED HELMET DETECTIONS
        # =====================================================

        for obj in helmet_objects:

            if id(obj) in used_helmet:
                continue

            x1, y1, x2, y2 = obj[
                "bbox"
            ]

            status = obj.get(
                "helmet"
            )

            if status is True:

                color = self.COLORS[
                    "safe"
                ]

            elif status is False:

                color = self.COLORS[
                    "violation"
                ]

            else:

                color = (
                    150,
                    150,
                    150
                )

            cv2.rectangle(

                annotated,

                (x1, y1),

                (x2, y2),

                color,

                1
            )

        # =====================================================
        # 5. PLATE READING
        # =====================================================

        if violations:

            try:

                from utils.plate_reader import PlateReader

                reader = PlateReader()

                for violation in violations:

                    plate = reader.read_from_region(

                        frame,

                        violation["bbox"],

                        expand=80
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
    # FIND PERSON/HEAD ABOVE MOTORCYCLE
    # =========================================================

    def _heads_on_moto(
        self,
        motorcycle_bbox,
        persons
    ):

        mx1, my1, mx2, my2 = (
            motorcycle_bbox
        )

        mw = mx2 - mx1

        mh = my2 - my1

        # Search above motorcycle
        # for helmet/head detections.

        zone_x1 = (
            mx1 -
            int(mw * 0.30)
        )

        zone_x2 = (
            mx2 +
            int(mw * 0.30)
        )

        zone_y1 = (
            my1 -
            int(mh * 1.5)
        )

        zone_y2 = (
            my2 +
            int(mh * 0.15)
        )

        matched = []

        for person in persons:

            px1, py1, px2, py2 = (
                person["bbox"]
            )

            center_x = (
                px1 + px2
            ) / 2

            center_y = (
                py1 + py2
            ) / 2

            if (

                zone_x1 <= center_x <= zone_x2

                and

                zone_y1 <= center_y <= zone_y2

            ):

                matched.append(
                    person
                )

        return matched

    # =========================================================
    # SELECT RIDER
    # =========================================================

    def _identify_rider(
        self,
        motorcycle_bbox,
        persons
    ):

        if not persons:
            return None

        if len(persons) == 1:
            return persons[0]

        mx1, my1, mx2, my2 = (
            motorcycle_bbox
        )

        motorcycle_center = (
            mx1 + mx2
        ) / 2

        def score(person):

            x1, y1, x2, y2 = (
                person["bbox"]
            )

            person_center = (
                x1 + x2
            ) / 2

            horizontal_distance = abs(
                person_center -
                motorcycle_center
            )

            motorcycle_width = max(
                mx2 - mx1,
                1
            )

            horizontal_score = max(

                0,

                1 -
                horizontal_distance /
                motorcycle_width
            )

            # Prefer the person whose
            # center is lower/closer to
            # the motorcycle.

            vertical_score = (
                (y1 + y2) / 2
            )

            return (
                horizontal_score * 0.7
                +
                vertical_score * 0.3
            )

        return max(
            persons,
            key=score
        )

    # =========================================================
    # DRAW VEHICLE LABEL
    # =========================================================

    def _put_vehicle_label(
        self,
        frame,
        text,
        pos
    ):

        x, y = pos

        font_scale = 0.42

        thickness = 1

        cv2.putText(

            frame,

            text,

            (x, max(y - 5, 15)),

            cv2.FONT_HERSHEY_SIMPLEX,

            font_scale,

            self.COLORS["track"],

            thickness
        )

    # =========================================================
    # DRAW LABEL
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

        (tw, th), _ = cv2.getTextSize(

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