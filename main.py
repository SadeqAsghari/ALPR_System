import cv2
import depthai as dai
import numpy as np
import time
import os
from collections import defaultdict, deque
from datetime import datetime
import math


########## CONFIGURATION

VIDEO_PATH          = "/Path/to/your/video"  # the input video in this project is recorded video instead of device camera

# Main detector: yolov6nr3 416x416
VEHICLE_MODEL_PATH  = "models/yolov6nr3_coco_416x416.blob"

# License plate detector (YOLOv8 640x640) and OCR
LP_MODEL_PATH       = "models/best_lp_openvino_2022.1_6shave.blob"
PADDLE_OCR_BLOB_PATH= "models/paddle_ocr_320x48.blob"

# NN spaces
VEH_NN_SIZE = 416   # vehicle/tracking NN space (square)
LP_NN_SIZE  = 640   # LP model input size

# Visualization target (window canvas)
DISPLAY_W = 1280
DISPLAY_H = 720

COLOR_POLY  = (0, 255, 0)
COLOR_LINE  = (0, 0, 255)
COLOR_TEXT  = (255, 255, 255)
COLOR_TRK   = (0, 255, 255)     # yellow (vehicles)
COLOR_PERS  = (0, 200, 0)       # green (persons)
COLOR_LP    = (0, 0, 255)       # red (plates)
COLOR_BLINK = (0, 128, 255)     # line blink for line crossing event

# labels used in our project are person, car,motorcycle,but,truck
COCO_LABELS = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe"
]

# COCO IDs
PERSON_CLASS_ID   = 0
VEHICLE_CLASS_IDS = {2, 3, 5, 7}   # car, motorcycle, bus, truck
TRACKED_CLASS_IDS = {PERSON_CLASS_ID} | VEHICLE_CLASS_IDS

# License plate detection control (default, may be overridden by MODE)
PLATE_MIN_H          = 60      # NN bbox height threshold (in VEH_NN_SIZE) to try LP
PLATE_REQ_INTERVAL   = 10      # frames between LP attempts if no plate yet
LP_UPDATE_INTERVAL   = 30      # frames between LP "refresh" for tracks that already have a plate
LP_ACCEPT_THRESH     = 0.60    # min confidence to accept LP detection
LP_IOU_THRESH        = 0.50
MAX_LP_REQ_PER_FRAME = 1       # at most 1 LP inference per frame

# OCR control (default, may be overridden by MODE)
OCR_IMG_W            = 320
OCR_IMG_H            = 48
MAX_OCR_REQ_PER_FRAME= 1       # at most 1 OCR per frame
OCR_REQ_INTERVAL     = 10      # frames between OCR attempts for same track

# OCR classes from Paddle_Text_Recognition-320x48 
OCR_CLASSES = [
    "*","0","1","2","3","4","5","6","7","8","9",":",";","<","=",">","?","@",
    "A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R",
    "S","T","U","V","W","X","Y","Z","[","\\","]","^","_","`","a","b","c","d",
    "e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v",
    "w","x","y","z","{","|","}","~","!","\"","#","$","%","&","'","(",")","*",
    "+",",","-",".","/"," "," "
]
OCR_N_CLASSES     = len(OCR_CLASSES)  # 97
OCR_IGNORED_INDEX = {0}               # index 0 is CTC "blank"

# Speed estimation (for NN space)
M_PER_PX             = 0.05    # meters per pixel in NN space (we shold tune for realistic km/h)

# Logging
LOG_BASE_DIR         = "logs"
os.makedirs(LOG_BASE_DIR, exist_ok=True)


###### EXPERIMENT MODES TO SEE THEIR EFFECT ON CPU USAGE OF HOST

# MODE:
#   "H" = host-only baseline (no DepthAI pipeline, no detection/tracking/LP/OCR)
#   "A" = VPU: YOLOv6nr3 + tracker only (no LP/OCR)
#   "B" = VPU: full pipeline, aggressive LP/OCR (stress test)
#   "C" = VPU: full pipeline, throttled LP/OCR (current tuned config)
MODE = "C"

if MODE == "H":
    USE_DEPTHAI = False
    ENABLE_LP   = False
    ENABLE_OCR  = False
elif MODE == "A":
    USE_DEPTHAI = True
    ENABLE_LP   = False
    ENABLE_OCR  = False
elif MODE == "B":
    USE_DEPTHAI = True
    ENABLE_LP   = True
    ENABLE_OCR  = True
    PLATE_REQ_INTERVAL   = 1
    LP_UPDATE_INTERVAL   = 1
    MAX_LP_REQ_PER_FRAME = 10
    OCR_REQ_INTERVAL     = 1
    MAX_OCR_REQ_PER_FRAME= 10
else:  # "C"
    USE_DEPTHAI = True
    ENABLE_LP   = True
    ENABLE_OCR  = True


# GLOBAL STATS FOR TRACKING QUANTITATIVE MEASUREMENTS

STATS = {
    "frames": 0,
    "start_time": None,
    "total_time": 0.0,

    "total_tracks_started": 0,
    "person_tracks_started": 0,
    "vehicle_tracks_started": 0,
    "line_crossings": 0,

    "plate_requests": 0,
    "plate_accepted": 0,
    "ocr_requests": 0,
    "ocr_nonempty": 0,
}



####### ZONE CONFIGURATOR (polygon + lines in VEH_NN space)

class ZoneConfigurator:
    def __init__(self, nn_frame):
        self.nn_h, self.nn_w = nn_frame.shape[:2]
        self.frame_disp = cv2.resize(nn_frame, (640, 640))  # setup UI only
        self.scale_x = self.nn_w / 640.0
        self.scale_y = self.nn_h / 640.0

        self.polygon_points = []
        self.trip_lines = []
        self.current_line_start = None
        self.line_counter = 0
        self.mode = "polygon"
        self.done = False

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
        self.draw()

    def draw(self):
        img = self.frame_disp.copy()

        if len(self.polygon_points) > 0:
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
        cv2.putText(img, "Z: Lines | T: Start | R: Reset",
                    (20, 640 - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        cv2.imshow("Setup", img)

    def run(self):
        cv2.namedWindow("Setup", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Setup", 640, 640)
        cv2.setMouseCallback("Setup", self.mouse_callback)
        self.draw()

        while not self.done:
            key = cv2.waitKey(1) & 0xFF
            if key in [ord('z'), ord('Z')]:
                self.mode = "lines"
                self.draw()
            elif key in [ord('t'), ord('T')]:
                self.done = True
            elif key in [ord('r'), ord('R')]:
                self.polygon_points = []
                self.trip_lines = []
                self.mode = "polygon"
                self.draw()

        cv2.destroyWindow("Setup")

        final_poly = []
        for p in self.polygon_points:
            final_poly.append((int(p[0] * self.scale_x), int(p[1] * self.scale_y)))

        final_lines = []
        for p1, p2, lid in self.trip_lines:
            sp1 = (int(p1[0] * self.scale_x), int(p1[1] * self.scale_y))
            sp2 = (int(p2[0] * self.scale_x), int(p2[1] * self.scale_y))
            final_lines.append((sp1, sp2, lid))

        return final_poly, final_lines


########### UTILS

def is_in_poly(point, polygon_contour):
    if len(polygon_contour) < 3:
        return True
    return cv2.pointPolygonTest(polygon_contour, point, False) >= 0


def intersect(A, B, C, D):
    def ccw(p1, p2, p3):
        return (p3[1] - p1[1]) * (p2[0] - p1[0]) > (p2[1] - p1[1]) * (p3[0] - p1[0])
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)


def fmt_X(val: str, length: int) -> str:
    s = (val or "")
    if len(s) > length:
        s = s[:length]
    return s.ljust(length)


def fmt_9(num: int, length: int) -> str:
    if num < 0:
        num = 0
    s = str(int(num))
    if len(s) > length:
        s = s[-length:]
    return s.zfill(length)



##### LOGGING HELPER FUNCTIONS

class Logger:
    def __init__(self, base_dir: str):
        now = datetime.now()
        self.date_str = now.strftime("%Y-%m-%d")
        self.log_dir = base_dir
        os.makedirs(self.log_dir, exist_ok=True)

        self.log_path = os.path.join(self.log_dir, f"rilevazione_{self.date_str}.tt")
        self.frames_dir = os.path.join(self.log_dir, f"fg_{self.date_str}")
        os.makedirs(self.frames_dir, exist_ok=True)

        self.f = open(self.log_path, "a", encoding="utf-8")

        self.next_p = 1
        self.next_v = 1

        self.track_map = {}  # trackerId -> (logicalIdStr, P/V, progressive, isVehicle)

        self.pos_counter = defaultdict(int)
        self.dist_accum_m = defaultdict(float)
        self.start_time = {}
        self.last_center = {}

    def close(self):
        try:
            self.f.close()
        except Exception:
            pass

    def get_or_create_id(self, tid: int, label: int):
        global STATS

        if tid in self.track_map:
            return self.track_map[tid]

        is_vehicle = label in VEHICLE_CLASS_IDS
        if is_vehicle:
            pedVeh = "V"
            prog = self.next_v
            self.next_v += 1
            id_str = f"V{prog:07d}"
        else:
            pedVeh = "P"
            prog = self.next_p
            self.next_p += 1
            id_str = f"P{prog:07d}"

        self.track_map[tid] = (id_str, pedVeh, prog, is_vehicle)

        # stats
        STATS["total_tracks_started"] += 1
        if is_vehicle:
            STATS["vehicle_tracks_started"] += 1
        else:
            STATS["person_tracks_started"] += 1

        self.log_identification(pedVeh, prog, label, plate_text="", nationality="")
        self.start_time[(pedVeh, prog)] = time.time()
        return self.track_map[tid]

    def log_identification(self, pedVeh: str, prog: int, label: int,
                           plate_text: str = "", nationality: str = ""):
        if pedVeh == "V":
            if label == 2:
                t_veh = "A"  # car
            elif label == 3:
                t_veh = "MO"
            elif label == 5:
                t_veh = "BU"
            elif label == 7:
                t_veh = "TR"
            else:
                t_veh = "VV"
        else:
            t_veh = "PE"

        line = ""
        line += fmt_X(pedVeh, 1)
        line += fmt_9(prog, 7)
        line += fmt_X(t_veh, 2)
        line += fmt_X(nationality, 3)
        line += fmt_X(plate_text, 9)
        self.f.write(line + "\n")
        self.f.flush()

    def log_position(self, pedVeh: str, prog: int,
                     x: int, y: int,
                     speed_kmh: float,
                     dist_increment_m: float,
                     total_time_s: float):
        key = (pedVeh, prog)
        self.pos_counter[key] += 1
        pos_num = self.pos_counter[key]

        self.dist_accum_m[key] += max(0.0, dist_increment_m)
        total_dist_cm = int(self.dist_accum_m[key] * 100.0)

        speed_100 = int(max(0.0, speed_kmh * 100.0))

        line = ""
        line += fmt_X(pedVeh, 1)
        line += fmt_9(prog, 7)
        line += fmt_9(pos_num, 3)
        line += fmt_9(x, 5)
        line += fmt_9(y, 5)
        line += fmt_9(speed_100, 5)
        line += fmt_9(total_dist_cm, 7)
        line += fmt_9(int(total_time_s), 10)
        self.f.write(line + "\n")
        self.f.flush()

    def log_line_crossing(self, pedVeh: str, prog: int,
                          line_id: str,
                          frame_img_nn: np.ndarray):
        global STATS

        instant = int(time.time())
        line = ""
        line += fmt_X("A", 1)
        line += fmt_9(prog, 7)
        line += fmt_9(instant, 10)
        line += fmt_X(line_id, 5)
        self.f.write(line + "\n")
        self.f.flush()

        STATS["line_crossings"] += 1

        img_name = f"{prog:07d}_{instant}.jpeg"
        save_path = os.path.join(self.frames_dir, img_name)
        try:
            cv2.imwrite(save_path, frame_img_nn)
        except Exception as e:
            print(f"[LOG] WARNING: cannot save frame {save_path}: {e}")



################# LP REQUEST & DECODE – VEH_NN (416) -> LP_NN (640)

def send_plate_request(q_lp_in, plate_pending, frame_idx, track_id,
                       frame_nn, x1_nn, y1_nn, x2_nn, y2_nn):
    """
    Work in VEH_NN space (416x416), crop padded ROI, resize to LP_NN_SIZE (640x640)
    and send to LP YOLO. Store ROI & car box for later decode.
    """
    if not ENABLE_LP:
        return

    global STATS

    h_nn, w_nn = frame_nn.shape[:2]

    pad = int(0.25 * (y2_nn - y1_nn))
    rx1 = max(0, x1_nn - pad)
    ry1 = max(0, y1_nn - pad)
    rx2 = min(w_nn, x2_nn + pad)
    ry2 = min(h_nn, y2_nn + pad)

    if rx2 <= rx1 or ry2 <= ry1:
        return

    roi_nn = frame_nn[ry1:ry2, rx1:rx2]
    if roi_nn.size == 0:
        return

    roi_resized = cv2.resize(roi_nn, (LP_NN_SIZE, LP_NN_SIZE))
    img = dai.ImgFrame()
    img.setType(dai.ImgFrame.Type.BGR888p)
    img.setWidth(LP_NN_SIZE)
    img.setHeight(LP_NN_SIZE)
    img.setData(roi_resized.transpose(2, 0, 1).flatten())

    seq = frame_idx * 1000 + track_id
    img.setSequenceNum(seq)

    try:
        q_lp_in.send(img)
        plate_pending[seq] = {
            "track_id": track_id,
            "car_box_nn": (x1_nn, y1_nn, x2_nn, y2_nn),
            "roi_nn": (rx1, ry1, rx2, ry2),
            "roi_frame_lp": roi_resized,  # LP_NN_SIZE x LP_NN_SIZE
        }
        STATS["plate_requests"] += 1
    except RuntimeError as e:
        print(f"[LP] WARNING: failed to send LP request for track {track_id}: {e}")


def decode_lp_result(lp_detections, info, plate_results):
    global STATS

    track_id = info["track_id"]
    cx1, cy1, cx2, cy2 = info["car_box_nn"]
    rx1, ry1, rx2, ry2 = info["roi_nn"]
    roi_frame_lp = info["roi_frame_lp"]

    w_roi_nn = rx2 - rx1
    h_roi_nn = ry2 - ry1
    if w_roi_nn <= 0 or h_roi_nn <= 0:
        return

    best = None
    best_conf = 0.0
    for det in lp_detections.detections:
        conf = det.confidence
        if conf > best_conf:
            best_conf = conf
            best = det

    if best is None or best_conf < LP_ACCEPT_THRESH:
        return

    # Map LP detection (normalized [0,1] in ROI) back into VEH_NN space
    px1_nn = int(rx1 + best.xmin * w_roi_nn)
    py1_nn = int(ry1 + best.ymin * h_roi_nn)
    px2_nn = int(rx1 + best.xmax * w_roi_nn)
    py2_nn = int(ry1 + best.ymax * h_roi_nn)

    w_lp_nn = px2_nn - px1_nn
    h_lp_nn = py2_nn - py1_nn
    if w_lp_nn <= 0 or h_lp_nn <= 0:
        return

    ratio = w_lp_nn / float(h_lp_nn)
    if ratio < 1.5 or ratio > 6.0:
        return

    # Relative coords inside car box (VEH_NN space)
    w_car = max(1, cx2 - cx1)
    h_car = max(1, cy2 - cy1)
    rel_x1 = (px1_nn - cx1) / w_car
    rel_y1 = (py1_nn - cy1) / h_car
    rel_x2 = (px2_nn - cx1) / w_car
    rel_y2 = (py2_nn - cy1) / h_car

    # Plate patch for OCR from roi_frame_lp (LP_NN_SIZE)
    px1_roi = int(best.xmin * LP_NN_SIZE)
    py1_roi = int(best.ymin * LP_NN_SIZE)
    px2_roi = int(best.xmax * LP_NN_SIZE)
    py2_roi = int(best.ymax * LP_NN_SIZE)

    px1_roi = max(0, min(LP_NN_SIZE - 1, px1_roi))
    py1_roi = max(0, min(LP_NN_SIZE - 1, py1_roi))
    px2_roi = max(0, min(LP_NN_SIZE - 1, px2_roi))
    py2_roi = max(0, min(LP_NN_SIZE - 1, py2_roi))

    plate_img = roi_frame_lp[py1_roi:py2_roi, px1_roi:px2_roi]
    if plate_img.size == 0:
        return

    plate_results[track_id] = {
        "rel": (rel_x1, rel_y1, rel_x2, rel_y2),
        "conf": best_conf,
        "nn_box": (px1_nn, py1_nn, px2_nn, py2_nn),
        "plate_img": plate_img,
        "text": None
    }

    STATS["plate_accepted"] += 1



########################## OCR HELPER FUNCTIONS

def send_ocr_request(q_ocr_in, ocr_pending, frame_idx, track_id, plate_img):
    if not ENABLE_OCR:
        return

    if plate_img.size == 0:
        return

    global STATS

    roi_resized = cv2.resize(plate_img, (OCR_IMG_W, OCR_IMG_H))

    img = dai.ImgFrame()
    img.setType(dai.ImgFrame.Type.BGR888p)
    img.setWidth(OCR_IMG_W)
    img.setHeight(OCR_IMG_H)
    img.setData(roi_resized.transpose(2, 0, 1).flatten())

    seq = frame_idx * 1000 + track_id
    img.setSequenceNum(seq)

    try:
        q_ocr_in.send(img)
        ocr_pending[seq] = track_id
        STATS["ocr_requests"] += 1
    except RuntimeError as e:
        print(f"[OCR] WARNING: failed to send OCR request for track {track_id}: {e}")


def decode_ocr_result(nn_data):
    global STATS

    out = nn_data.getFirstLayerFp16()
    if out is None or len(out) == 0:
        return ""

    total = len(out)
    if total <= 0:
        return ""

    if total % OCR_N_CLASSES != 0:
        T = total // OCR_N_CLASSES
        if T <= 0:
            return ""
    else:
        T = total // OCR_N_CLASSES

    try:
        logits = np.array(out, dtype=np.float32).reshape(T, OCR_N_CLASSES)
    except ValueError:
        return ""

    indices = logits.argmax(axis=1)

    result_chars = []
    prev = None
    for idx in indices:
        if idx in OCR_IGNORED_INDEX:
            prev = idx
            continue
        if idx == prev:
            prev = idx
            continue
        if 0 <= idx < OCR_N_CLASSES:
            result_chars.append(OCR_CLASSES[idx])
        prev = idx

    text = "".join(result_chars)
    text = "".join(c for c in text.upper() if c.isalnum())

    if text:
        STATS["ocr_nonempty"] += 1

    return text



########### CREATING PIPELINE

def create_pipeline():
    pipeline = dai.Pipeline()

    xin = pipeline.create(dai.node.XLinkIn)
    xin.setStreamName("in")
    xin.setMaxDataSize(VEH_NN_SIZE * VEH_NN_SIZE * 3)
    xin.setNumFrames(4)

    veh_net = pipeline.create(dai.node.YoloDetectionNetwork)
    veh_net.setBlobPath(VEHICLE_MODEL_PATH)
    veh_net.setConfidenceThreshold(0.4)
    veh_net.setNumClasses(80)
    veh_net.setCoordinateSize(4)
    veh_net.setAnchors([
        10, 13, 16, 30, 33, 23,
        30, 61, 62, 45, 59, 119,
        116, 90, 156, 198, 373, 326
    ])
    veh_net.setAnchorMasks({
        "side80": [0, 1, 2],
        "side40": [3, 4, 5],
        "side20": [6, 7, 8],
    })
    veh_net.setIouThreshold(0.5)
    veh_net.setNumInferenceThreads(2)
    veh_net.input.setBlocking(False)

    tracker = pipeline.create(dai.node.ObjectTracker)
    tracker.setDetectionLabelsToTrack(list(TRACKED_CLASS_IDS))
    tracker.setTrackerType(dai.TrackerType.ZERO_TERM_COLOR_HISTOGRAM)
    tracker.setTrackerIdAssignmentPolicy(dai.TrackerIdAssignmentPolicy.UNIQUE_ID)
    tracker.setMaxObjectsToTrack(80)

    xin.out.link(veh_net.input)
    xin.out.link(tracker.inputTrackerFrame)
    veh_net.passthrough.link(tracker.inputDetectionFrame)
    veh_net.out.link(tracker.inputDetections)

    xout_trk = pipeline.create(dai.node.XLinkOut)
    xout_trk.setStreamName("tracklets")
    tracker.out.link(xout_trk.input)

    # LP pipeline (YOLOV8 IS ANCHORLESS)
    lp_in = pipeline.create(dai.node.XLinkIn)
    lp_in.setStreamName("lp_in")
    lp_in.setMaxDataSize(LP_NN_SIZE * LP_NN_SIZE * 3 * 2)

    lp_net = pipeline.create(dai.node.YoloDetectionNetwork)
    lp_net.setBlobPath(LP_MODEL_PATH)
    lp_net.setNumClasses(1)
    lp_net.setCoordinateSize(4)
    lp_net.setAnchors([])
    lp_net.setAnchorMasks({})
    lp_net.setIouThreshold(LP_IOU_THRESH)
    lp_net.setConfidenceThreshold(0.5)
    lp_net.input.setBlocking(False)
    lp_net.setNumInferenceThreads(1)

    lp_in.out.link(lp_net.input)

    lp_out = pipeline.create(dai.node.XLinkOut)
    lp_out.setStreamName("lp_out")
    lp_net.out.link(lp_out.input)

    # OCR pipeline
    ocr_in = pipeline.create(dai.node.XLinkIn)
    ocr_in.setStreamName("ocr_in")
    ocr_in.setMaxDataSize(OCR_IMG_W * OCR_IMG_H * 3 * 2)

    ocr_nn = pipeline.create(dai.node.NeuralNetwork)
    ocr_nn.setBlobPath(PADDLE_OCR_BLOB_PATH)
    ocr_nn.input.setBlocking(False)
    ocr_nn.setNumInferenceThreads(1)

    ocr_in.out.link(ocr_nn.input)

    ocr_out = pipeline.create(dai.node.XLinkOut)
    ocr_out.setStreamName("ocr_out")
    ocr_nn.out.link(ocr_out.input)

    return pipeline



####################### MAIN

def main():
    global STATS

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video: {VIDEO_PATH}")
        return

    print(f"DEBUG: Video opened. MODE = {MODE} (USE_DEPTHAI={USE_DEPTHAI})")
    ret, first_raw = cap.read()
    if not ret:
        print("ERROR: Cannot read first frame")
        return
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0
    dt = 1.0 / fps

    # Setup zones in VEH_NN space
    first_nn = cv2.resize(first_raw, (VEH_NN_SIZE, VEH_NN_SIZE))
    config = ZoneConfigurator(first_nn)
    poly_pts, trip_lines = config.run()
    poly_contour = np.array(poly_pts, dtype=np.int32).reshape((-1, 1, 2)) if poly_pts else []

    STATS["start_time"] = time.time()


    # MODE "H": HOST-ONLY BASELINE
  
    if not USE_DEPTHAI:
        print("DEBUG: Running HOST-ONLY baseline (no detection/tracking/LP/OCR)...")

        cv2.namedWindow("Live", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Live", DISPLAY_W, DISPLAY_H)

        frame_idx = 0
        while True:
            ret, raw = cap.read()
            if not ret:
                print("End of video.")
                break

            STATS["frames"] += 1
            raw_h, raw_w = raw.shape[:2]

            # scale NN (416) -> raw
            sx = raw_w / float(VEH_NN_SIZE)
            sy = raw_h / float(VEH_NN_SIZE)

            disp = raw.copy()

            # draw polygon & lines in raw space only (no detections)
            if len(poly_contour) > 0:
                poly_raw = (poly_contour * [sx, sy]).astype(np.int32)
                cv2.polylines(disp, [poly_raw], True, COLOR_POLY, 2)

            now_ts = time.time()
            for p1, p2, lid in trip_lines:
                p1r = (int(p1[0] * sx), int(p1[1] * sy))
                p2r = (int(p2[0] * sx), int(p2[1] * sy))
                col = COLOR_LINE
                cv2.line(disp, p1r, p2r, col, 2)
                cv2.putText(disp, lid, p1r, cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)

            # ---------- 1280x720 visualization, downscale-only with AR preserved ----------
            scale = min(1.0, min(DISPLAY_W / float(raw_w), DISPLAY_H / float(raw_h)))
            new_w = int(raw_w * scale)
            new_h = int(raw_h * scale)
            disp_scaled = cv2.resize(disp, (new_w, new_h)) if scale != 1.0 else disp

            canvas = np.zeros((DISPLAY_H, DISPLAY_W, 3), dtype=np.uint8)
            x_offset = (DISPLAY_W - new_w) // 2
            y_offset = (DISPLAY_H - new_h) // 2
            canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = disp_scaled

            cv2.imshow("Live", canvas)
            frame_idx += 1

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    
    # MODES "A", "B", "C": DEPTHAI PIPELINE
  
    else:
        print("DEBUG: Creating pipeline...")
        pipeline = create_pipeline()

        plate_pending = {}
        plate_results = {}
        lp_last_req = {}
        blink_states = {}
        track_history = defaultdict(lambda: deque(maxlen=2))

        ocr_pending = {}
        ocr_last_req = {}

        bbox_smooth = {}

        logger = Logger(LOG_BASE_DIR)

        with dai.Device(pipeline) as device:
            print("DEBUG: Device connected. Starting main loop...")

            q_in      = device.getInputQueue("in", maxSize=1, blocking=False)
            q_trk     = device.getOutputQueue("tracklets", maxSize=4, blocking=False)
            q_lp_in   = device.getInputQueue("lp_in", maxSize=4, blocking=False)
            q_lp_out  = device.getOutputQueue("lp_out", maxSize=4, blocking=False)
            q_ocr_in  = device.getInputQueue("ocr_in", maxSize=4, blocking=False)
            q_ocr_out = device.getOutputQueue("ocr_out", maxSize=4, blocking=False)

            cv2.namedWindow("Live", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Live", DISPLAY_W, DISPLAY_H)

            frame_idx = 0

            try:
                while True:
                    ret, raw = cap.read()
                    if not ret:
                        print("End of video.")
                        break

                    STATS["frames"] += 1
                    raw_h, raw_w = raw.shape[:2]

                    # Downscale raw to NN space for inference
                    frame_nn = cv2.resize(raw, (VEH_NN_SIZE, VEH_NN_SIZE))
                    h_nn, w_nn = frame_nn.shape[:2]

                    # scale factors NN -> raw
                    sx = raw_w / float(VEH_NN_SIZE)
                    sy = raw_h / float(VEH_NN_SIZE)

                    # Send NN frame to device
                    img = dai.ImgFrame()
                    img.setType(dai.ImgFrame.Type.BGR888p)
                    img.setWidth(VEH_NN_SIZE)
                    img.setHeight(VEH_NN_SIZE)
                    img.setData(frame_nn.transpose(2, 0, 1).flatten())
                    try:
                        q_in.send(img)
                    except RuntimeError as e:
                        if frame_idx % 30 == 0:
                            print(f"WARNING: send failed at frame {frame_idx}: {e}")

                    disp = raw.copy()

                    # OCR results
                    if ENABLE_OCR:
                        ocr_pkt = q_ocr_out.tryGet()
                        while ocr_pkt is not None:
                            seq = ocr_pkt.getSequenceNum()
                            if seq in ocr_pending:
                                tid = ocr_pending.pop(seq)
                                text = decode_ocr_result(ocr_pkt)
                                if text:
                                    if tid in plate_results:
                                        plate_results[tid]["text"] = text
                                        print(f"[OCR] Track {tid} text: {text}")
                            ocr_pkt = q_ocr_out.tryGet()

                    # LP results
                    if ENABLE_LP:
                        lp_pkt = q_lp_out.tryGet()
                        while lp_pkt is not None:
                            seq = lp_pkt.getSequenceNum()
                            if seq in plate_pending:
                                info = plate_pending.pop(seq)
                                decode_lp_result(lp_pkt, info, plate_results)
                            lp_pkt = q_lp_out.tryGet()

                    trk_data = q_trk.tryGet()
                    tracklets = trk_data.tracklets if trk_data is not None else []

                    if frame_idx % 20 == 0:
                        print(f"[DEBUG] Frame {frame_idx}: tracklets={len(tracklets)}, plates={len(plate_results)}")

                    # Draw polygon & lines in raw space
                    if len(poly_contour) > 0:
                        poly_raw = (poly_contour * [sx, sy]).astype(np.int32)
                        cv2.polylines(disp, [poly_raw], True, COLOR_POLY, 2)

                    now_ts = time.time()
                    for p1, p2, lid in trip_lines:
                        p1r = (int(p1[0] * sx), int(p1[1] * sy))
                        p2r = (int(p2[0] * sx), int(p2[1] * sy))
                        col = COLOR_BLINK if (lid in blink_states and now_ts < blink_states[lid]) else COLOR_LINE
                        cv2.line(disp, p1r, p2r, col, 2)
                        cv2.putText(disp, lid, p1r, cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)

                    lp_reqs_this_frame = 0
                    ocr_reqs_this_frame = 0

                    for t in tracklets:
                        if t.status not in (dai.Tracklet.TrackingStatus.TRACKED,
                                            dai.Tracklet.TrackingStatus.NEW):
                            continue
                        if t.label not in TRACKED_CLASS_IDS:
                            continue

                        tid = t.id
                        label = t.label

                        logical_id, pedVeh, prog, is_vehicle = logger.get_or_create_id(tid, label)

                        roi = t.roi.denormalize(w_nn, h_nn)
                        x1 = int(roi.topLeft().x)
                        y1 = int(roi.topLeft().y)
                        x2 = int(roi.bottomRight().x)
                        y2 = int(roi.bottomRight().y)

                        x1 = max(0, min(w_nn - 1, x1))
                        y1 = max(0, min(h_nn - 1, y1))
                        x2 = max(0, min(w_nn - 1, x2))
                        y2 = max(0, min(h_nn - 1, y2))

                        # EMA smoothing in NN space
                        alpha = 0.4
                        if tid in bbox_smooth:
                            px1, py1, px2, py2 = bbox_smooth[tid]
                            sx1 = int(alpha * x1 + (1 - alpha) * px1)
                            sy1 = int(alpha * y1 + (1 - alpha) * py1)
                            sx2 = int(alpha * x2 + (1 - alpha) * px2)
                            sy2 = int(alpha * y2 + (1 - alpha) * py2)
                        else:
                            sx1, sy1, sx2, sy2 = x1, y1, x2, y2
                        bbox_smooth[tid] = (sx1, sy1, sx2, sy2)

                        x1, y1, x2, y2 = sx1, sy1, sx2, sy2

                        cx = (x1 + x2) // 2
                        cy = (y1 + y2) // 2

                        inside_zone = len(poly_contour) == 0 or is_in_poly((cx, cy), poly_contour)

                        key = (pedVeh, prog)
                        dist_inc_m = 0.0
                        speed_kmh = 0.0
                        if key in logger.last_center:
                            prev_cx, prev_cy = logger.last_center[key]
                            dp = math.hypot(cx - prev_cx, cy - prev_cy)
                            dist_inc_m = dp * M_PER_PX
                            speed_m_s = dist_inc_m / dt
                            speed_kmh = speed_m_s * 3.6
                        logger.last_center[key] = (cx, cy)

                        start_t = logger.start_time.get(key, time.time())
                        total_time_s = max(0.0, time.time() - start_t)

                        track_history[tid].append((cx, cy))
                        if len(track_history[tid]) == 2:
                            p_prev, p_curr = track_history[tid][0], track_history[tid][1]
                            for (l_p1, l_p2, lid) in trip_lines:
                                if intersect(p_prev, p_curr, l_p1, l_p2):
                                    blink_states[lid] = time.time() + 0.5
                                    logger.log_line_crossing(pedVeh, prog, lid, frame_nn)

                        # Draw boxes in RAW space
                        rx1 = int(x1 * sx)
                        ry1 = int(y1 * sy)
                        rx2 = int(x2 * sx)
                        ry2 = int(y2 * sy)

                        if inside_zone:
                            color_box = COLOR_TRK if is_vehicle else COLOR_PERS
                            cv2.rectangle(disp, (rx1, ry1), (rx2, ry2), color_box, 2)

                            label_name = COCO_LABELS[label] if 0 <= label < len(COCO_LABELS) else str(label)
                            base_text = f"{logical_id} {label_name}"

                            if is_vehicle and tid in plate_results and plate_results[tid].get("text"):
                                base_text += f" [{plate_results[tid]['text']}]"

                            base_text += f" {speed_kmh:.1f}km/h"

                            cv2.putText(disp, base_text,
                                        (rx1, max(0, ry1 - 5)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_box, 2)

                            logger.log_position(pedVeh, prog, cx, cy, speed_kmh, dist_inc_m, total_time_s)

                        # LP trigger for vehicles only
                        if ENABLE_LP and is_vehicle and inside_zone:
                            h_box = y2 - y1
                            if h_box > PLATE_MIN_H:
                                last_req = lp_last_req.get(tid, -999999)
                                interval = LP_UPDATE_INTERVAL if tid in plate_results else PLATE_REQ_INTERVAL

                                if (frame_idx - last_req >= interval and
                                        lp_reqs_this_frame < MAX_LP_REQ_PER_FRAME):
                                    send_plate_request(
                                        q_lp_in, plate_pending,
                                        frame_idx, tid,
                                        frame_nn,
                                        x1, y1, x2, y2
                                    )
                                    lp_last_req[tid] = frame_idx
                                    lp_reqs_this_frame += 1

                        # LP box drawing on RAW
                        if ENABLE_LP and is_vehicle and tid in plate_results and inside_zone:
                            rel_x1, rel_y1, rel_x2, rel_y2 = plate_results[tid]["rel"]
                            w_car_nn = x2 - x1
                            h_car_nn = y2 - y1

                            px1_nn = int(x1 + rel_x1 * w_car_nn)
                            py1_nn = int(y1 + rel_y1 * h_car_nn)
                            px2_nn = int(x1 + rel_x2 * w_car_nn)
                            py2_nn = int(y1 + rel_y2 * h_car_nn)

                            pad_w = int(0.1 * (px2_nn - px1_nn))
                            pad_h = int(0.2 * (py2_nn - py1_nn))
                            px1_nn -= pad_w
                            px2_nn += pad_w
                            py1_nn -= pad_h
                            py2_nn += pad_h

                            px1_nn = max(0, px1_nn); py1_nn = max(0, py1_nn)
                            px2_nn = min(w_nn - 1, px2_nn); py2_nn = min(h_nn - 1, py2_nn)

                            prx1 = int(px1_nn * sx)
                            pry1 = int(py1_nn * sy)
                            prx2 = int(px2_nn * sx)
                            pry2 = int(py2_nn * sy)

                            cv2.rectangle(disp, (prx1, pry1), (prx2, pry2), COLOR_LP, 2)
                            cv2.putText(disp, "LP", (prx1, max(0, pry1 - 5)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_LP, 1)

                        # OCR scheduling
                        if (ENABLE_OCR and is_vehicle and tid in plate_results and
                                plate_results[tid].get("plate_img") is not None and
                                not plate_results[tid].get("text")):
                            if ocr_reqs_this_frame < MAX_OCR_REQ_PER_FRAME:
                                last_ocr = ocr_last_req.get(tid, -999999)
                                if frame_idx - last_ocr >= OCR_REQ_INTERVAL:
                                    plate_img = plate_results[tid]["plate_img"]
                                    send_ocr_request(q_ocr_in, ocr_pending, frame_idx, tid, plate_img)
                                    ocr_last_req[tid] = frame_idx
                                    ocr_reqs_this_frame += 1

                    # ---------- 1280x720 visualization, downscale-only with AR preserved ----------
                    scale = min(1.0, min(DISPLAY_W / float(raw_w), DISPLAY_H / float(raw_h)))
                    new_w = int(raw_w * scale)
                    new_h = int(raw_h * scale)
                    disp_scaled = cv2.resize(disp, (new_w, new_h)) if scale != 1.0 else disp

                    canvas = np.zeros((DISPLAY_H, DISPLAY_W, 3), dtype=np.uint8)
                    x_offset = (DISPLAY_W - new_w) // 2
                    y_offset = (DISPLAY_H - new_h) // 2
                    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = disp_scaled

                    cv2.imshow("Live", canvas)
                    frame_idx += 1

                    key = cv2.waitKey(1)
                    if key & 0xFF == ord('q'):
                        break

            finally:
                logger.close()

        cap.release()
        cv2.destroyAllWindows()

  
    # PRINT STATS FOR MEASUREMENTS

    if STATS["start_time"] is not None:
        STATS["total_time"] = time.time() - STATS["start_time"]

    print("\n========== RUNTIME STATS ==========")
    print(f"Mode: {MODE} (USE_DEPTHAI={USE_DEPTHAI}, ENABLE_LP={ENABLE_LP}, ENABLE_OCR={ENABLE_OCR})")
    print(f"Frames processed: {STATS['frames']}")
    print(f"Total time: {STATS['total_time']:.2f} s")
    if STATS["total_time"] > 0:
        avg_fps = STATS["frames"] / STATS["total_time"]
        print(f"Average FPS (end-to-end): {avg_fps:.2f}")

    print("\n========== TRACKING STATS ==========")
    print(f"Unique tracks started (all classes): {STATS['total_tracks_started']}")
    print(f"Unique person tracks:               {STATS['person_tracks_started']}")
    print(f"Unique vehicle tracks:              {STATS['vehicle_tracks_started']}")
    print(f"Line crossings (all lines):         {STATS['line_crossings']}")

    print("\n========== PLATE / OCR STATS ==========")
    print(f"Plate requests sent:                {STATS['plate_requests']}")
    print(f"Plate detections accepted:          {STATS['plate_accepted']}")
    print(f"OCR requests sent:                  {STATS['ocr_requests']}")
    print(f"OCR results with non-empty text:    {STATS['ocr_nonempty']}")
    print("=====================================\n")


if __name__ == "__main__":
    main()
