import cv2
import numpy as np
import os



# Main configs

VIDEO_PATH           = "/Path/to/video"

# Model blobs
VEHICLE_MODEL_PATH   = "models/yolov6nr3_coco_416x416.blob"
LP_MODEL_PATH        = "models/best_lp_openvino_2022.1_6shave.blob"
PADDLE_OCR_BLOB_PATH = "models/paddle_ocr_320x48.blob"

# NN input sizes
VEH_NN_SIZE = 416   # vehicle detector / tracker (square)
LP_NN_SIZE  = 640   # license-plate detector input size

# Visualization canvas
DISPLAY_W = 1280
DISPLAY_H = 720

# Drawing colours  (BGR)
COLOR_POLY  = (0, 255,   0)
COLOR_LINE  = (0,   0, 255)
COLOR_TEXT  = (255, 255, 255)
COLOR_TRK   = (0, 255, 255)   # cyan  — tracked vehicles
COLOR_PERS  = (0, 200,   0)   # green — tracked persons
COLOR_LP    = (0,   0, 255)   # red   — license plates
COLOR_BLINK = (0, 128, 255)   # orange blink on line-crossing event

# COCO class labels used by the vehicle model
COCO_LABELS = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe",
]

# Class ID constants
PERSON_CLASS_ID   = 0
VEHICLE_CLASS_IDS = {2, 3, 5, 7}            # car, motorcycle, bus, truck
TRACKED_CLASS_IDS = {PERSON_CLASS_ID} | VEHICLE_CLASS_IDS

# License-plate detection throttling
PLATE_MIN_H          = 60    # minimum bbox height (NN px) before requesting LP inference
PLATE_REQ_INTERVAL   = 10    # frames between LP attempts when no plate found yet
LP_UPDATE_INTERVAL   = 30    # frames between LP refresh for tracks that already have a plate
LP_ACCEPT_THRESH     = 0.60  # minimum LP detection confidence to accept
LP_IOU_THRESH        = 0.50
MAX_LP_REQ_PER_FRAME = 1     # cap LP inferences dispatched per frame

# OCR throttling
OCR_IMG_W            = 320
OCR_IMG_H            = 48
MAX_OCR_REQ_PER_FRAME = 1    # cap OCR inferences dispatched per frame
OCR_REQ_INTERVAL     = 10    # frames between OCR retries for the same track

# OCR vocabulary — PaddleOCR 320x48 output classes
OCR_CLASSES = [
    "*", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", ":", ";", "<",
    "=", ">", "?", "@", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
    "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X",
    "Y", "Z", "[", "\\", "]", "^", "_", "`", "a", "b", "c", "d", "e", "f",
    "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t",
    "u", "v", "w", "x", "y", "z", "{", "|", "}", "~", "!", "\"", "#", "$",
    "%", "&", "'", "(", ")", "*", "+", ",", "-", ".", "/", " ", " ",
]
OCR_N_CLASSES     = len(OCR_CLASSES)   # 97
OCR_IGNORED_INDEX = {0}                # index 0 is the CTC blank token

# Speed estimation — tune M_PER_PX for realistic km/h values
M_PER_PX = 0.05   # metres per pixel in VEH_NN_SIZE space

# Logging
LOG_BASE_DIR = "logs"


# Zone config helpers

def is_in_poly(point, polygon_contour) -> bool:
    """Return True if point lies inside (or on the edge of) polygon_contour."""
    if len(polygon_contour) < 3:
        return True
    return cv2.pointPolygonTest(polygon_contour, point, False) >= 0


def intersect(A, B, C, D) -> bool:
    """Return True if segment AB intersects segment CD."""
    def ccw(p1, p2, p3):
        return (p3[1] - p1[1]) * (p2[0] - p1[0]) > (p2[1] - p1[1]) * (p3[0] - p1[0])
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)


# Line-crossing helpers

def fmt_X(val: str, length: int) -> str:
    """Left-align string in a fixed-width field, truncating if necessary."""
    s = (val or "")
    if len(s) > length:
        s = s[:length]
    return s.ljust(length)


def fmt_9(num: int, length: int) -> str:
    """Zero-pad integer in a fixed-width field, keeping only the last `length` digits on overflow."""
    if num < 0:
        num = 0
    s = str(int(num))
    if len(s) > length:
        s = s[-length:]
    return s.zfill(length)



# Zone configuration setup
# Interactive setup window: draw monitoring polygon then trip-lines,
# all coordinates stored in VEH_NN_SIZE space.


class ZoneConfigurator:
    """
    Opens a setup UI on the first video frame.

    Controls
    --------
    Left-click      Add polygon vertex (polygon mode) or define line endpoints (lines mode)
    Z               Switch to lines mode
    T               Confirm and close
    R               Reset everything and start over
    """

    def __init__(self, nn_frame: np.ndarray):
        self.nn_h, self.nn_w = nn_frame.shape[:2]
        # Scale up for the setup UI display (does not affect stored coordinates)
        self.frame_disp = cv2.resize(nn_frame, (640, 640))
        self.scale_x = self.nn_w / 640.0
        self.scale_y = self.nn_h / 640.0

        self.polygon_points  = []
        self.trip_lines      = []
        self.current_line_start = None
        self.line_counter    = 0
        self.mode            = "polygon"
        self.done            = False

    def mouse_callback(self, event, x, y, flags, param):
        if self.done:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.mode == "polygon":
                self.polygon_points.append((x, y))
            elif self.mode == "lines":
                if self.current_line_start is None:
                    self.current_line_start = (x, y)
                else:
                    line_id = f"AC{self.line_counter:03d}"
                    self.trip_lines.append((self.current_line_start, (x, y), line_id))
                    self.current_line_start = None
                    self.line_counter += 1
        self._draw()

    def _draw(self):
        img = self.frame_disp.copy()

        if self.polygon_points:
            pts = np.array(self.polygon_points, np.int32).reshape((-1, 1, 2))
            is_closed = (self.mode != "polygon")
            cv2.polylines(img, [pts], is_closed, COLOR_POLY, 2)

        for p1, p2, lid in self.trip_lines:
            cv2.line(img, p1, p2, COLOR_LINE, 2)
            cv2.putText(img, lid, p1, cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_LINE, 2)

        if self.current_line_start:
            cv2.circle(img, self.current_line_start, 5, COLOR_LINE, -1)

        cv2.putText(img, f"Mode: {self.mode.upper()}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(img, "Z: Lines | T: Confirm | R: Reset",
                    (20, 610), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        cv2.imshow("Setup", img)

    def run(self):
        cv2.namedWindow("Setup", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Setup", 640, 640)
        cv2.setMouseCallback("Setup", self.mouse_callback)
        self._draw()

        while not self.done:
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('z'), ord('Z')):
                self.mode = "lines"
                self._draw()
            elif key in (ord('t'), ord('T')):
                self.done = True
            elif key in (ord('r'), ord('R')):
                self.polygon_points = []
                self.trip_lines     = []
                self.mode           = "polygon"
                self._draw()

        cv2.destroyWindow("Setup")

        # Map display coordinates back to VEH_NN_SIZE space
        final_poly = [
            (int(p[0] * self.scale_x), int(p[1] * self.scale_y))
            for p in self.polygon_points
        ]
        final_lines = [
            (
                (int(p1[0] * self.scale_x), int(p1[1] * self.scale_y)),
                (int(p2[0] * self.scale_x), int(p2[1] * self.scale_y)),
                lid,
            )
            for p1, p2, lid in self.trip_lines
        ]
        return final_poly, final_lines
