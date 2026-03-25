import cv2
import depthai as dai
import numpy as np

from utils import (
    VEH_NN_SIZE, LP_NN_SIZE, OCR_IMG_W, OCR_IMG_H,
    VEHICLE_MODEL_PATH, LP_MODEL_PATH, PADDLE_OCR_BLOB_PATH,
    TRACKED_CLASS_IDS, LP_ACCEPT_THRESH, LP_IOU_THRESH,
    OCR_N_CLASSES, OCR_IGNORED_INDEX, OCR_CLASSES,
)



# Pipeline builder

def create_pipeline() -> dai.Pipeline:
    
    pipeline = dai.Pipeline()

    # Vehicle detector + tracker

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
        116, 90, 156, 198, 373, 326,
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

    # License-plate detector (YOLOv8, anchorless)
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

    # OCR network (PaddleOCR 320x48)

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


 
# LICENSE-PLATE helpers

def send_plate_request(q_lp_in, plate_pending: dict, frame_idx: int, track_id: int,
                       frame_nn, x1_nn: int, y1_nn: int, x2_nn: int, y2_nn: int,
                       stats: dict):
    """
    Crop a padded ROI around the vehicle in VEH_NN space, resize to LP_NN_SIZE,
    and send it to the LP detector on the device.

    Stores request metadata in ``plate_pending`` keyed by sequence number so
    ``decode_lp_result`` can map the response back to the originating track.
    """
    h_nn, w_nn = frame_nn.shape[:2]

    pad = int(0.25 * (y2_nn - y1_nn))
    rx1 = max(0,    x1_nn - pad)
    ry1 = max(0,    y1_nn - pad)
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
            "track_id":    track_id,
            "car_box_nn":  (x1_nn, y1_nn, x2_nn, y2_nn),
            "roi_nn":      (rx1, ry1, rx2, ry2),
            "roi_frame_lp": roi_resized,   # LP_NN_SIZE × LP_NN_SIZE
        }
        stats["plate_requests"] += 1
    except RuntimeError as exc:
        print(f"[LP] WARNING: failed to send LP request for track {track_id}: {exc}")


def decode_lp_result(lp_detections, info: dict, plate_results: dict, stats: dict):
    """
    Map the LP detector output back to VEH_NN space and store the result
    in ``plate_results`` keyed by track ID.

    Silently drops detections below LP_ACCEPT_THRESH or with implausible aspect ratios.
    """
    track_id = info["track_id"]
    cx1, cy1, cx2, cy2 = info["car_box_nn"]
    rx1, ry1, rx2, ry2 = info["roi_nn"]
    roi_frame_lp        = info["roi_frame_lp"]

    w_roi_nn = rx2 - rx1
    h_roi_nn = ry2 - ry1
    if w_roi_nn <= 0 or h_roi_nn <= 0:
        return

    # Pick the best detection by confidence
    best, best_conf = None, 0.0
    for det in lp_detections.detections:
        if det.confidence > best_conf:
            best_conf = det.confidence
            best = det

    if best is None or best_conf < LP_ACCEPT_THRESH:
        return

    # LP bbox in VEH_NN space
    px1_nn = int(rx1 + best.xmin * w_roi_nn)
    py1_nn = int(ry1 + best.ymin * h_roi_nn)
    px2_nn = int(rx1 + best.xmax * w_roi_nn)
    py2_nn = int(ry1 + best.ymax * h_roi_nn)

    w_lp_nn = px2_nn - px1_nn
    h_lp_nn = py2_nn - py1_nn
    if w_lp_nn <= 0 or h_lp_nn <= 0:
        return

    # Sanity-check plate aspect ratio (width:height should be 1.5–6.0)
    ratio = w_lp_nn / float(h_lp_nn)
    if not (1.5 <= ratio <= 6.0):
        return

    # Relative position inside the car bbox (for stable overlay across frames)
    w_car = max(1, cx2 - cx1)
    h_car = max(1, cy2 - cy1)
    rel_x1 = (px1_nn - cx1) / w_car
    rel_y1 = (py1_nn - cy1) / h_car
    rel_x2 = (px2_nn - cx1) / w_car
    rel_y2 = (py2_nn - cy1) / h_car

    # Plate crop from the LP_NN_SIZE ROI image (used as OCR input)
    px1_roi = max(0, min(LP_NN_SIZE - 1, int(best.xmin * LP_NN_SIZE)))
    py1_roi = max(0, min(LP_NN_SIZE - 1, int(best.ymin * LP_NN_SIZE)))
    px2_roi = max(0, min(LP_NN_SIZE - 1, int(best.xmax * LP_NN_SIZE)))
    py2_roi = max(0, min(LP_NN_SIZE - 1, int(best.ymax * LP_NN_SIZE)))

    plate_img = roi_frame_lp[py1_roi:py2_roi, px1_roi:px2_roi]
    if plate_img.size == 0:
        return

    plate_results[track_id] = {
        "rel":       (rel_x1, rel_y1, rel_x2, rel_y2),
        "conf":      best_conf,
        "nn_box":    (px1_nn, py1_nn, px2_nn, py2_nn),
        "plate_img": plate_img,
        "text":      None,
    }
    stats["plate_accepted"] += 1



# OCR helpers

def send_ocr_request(q_ocr_in, ocr_pending: dict, frame_idx: int, track_id: int,
                     plate_img, stats: dict):
    """Resize plate crop to OCR input size and dispatch to the OCR network."""
    if plate_img.size == 0:
        return

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
        stats["ocr_requests"] += 1
    except RuntimeError as exc:
        print(f"[OCR] WARNING: failed to send OCR request for track {track_id}: {exc}")


def decode_ocr_result(nn_data, stats: dict) -> str:
    """
    Run CTC greedy decode on the OCR network output and return the
    alphanumeric plate text (upper-cased, empty string on failure).
    """
    out = nn_data.getFirstLayerFp16()
    if not out:
        return ""

    total = len(out)
    T = total // OCR_N_CLASSES
    if T <= 0:
        return ""

    try:
        logits = np.array(out, dtype=np.float32).reshape(T, OCR_N_CLASSES)
    except ValueError:
        return ""

    # CTC greedy decode: collapse repeated tokens, skip blanks
    result_chars = []
    prev = None
    for idx in logits.argmax(axis=1):
        if idx in OCR_IGNORED_INDEX or idx == prev:
            prev = idx
            continue
        if 0 <= idx < OCR_N_CLASSES:
            result_chars.append(OCR_CLASSES[idx])
        prev = idx

    text = "".join(c for c in "".join(result_chars).upper() if c.isalnum())
    if text:
        stats["ocr_nonempty"] += 1

    return text
