import cv2
import depthai as dai
import numpy as np
import time
import math
from collections import defaultdict, deque

from utils import (
    VIDEO_PATH,
    VEH_NN_SIZE, DISPLAY_W, DISPLAY_H,
    COLOR_POLY, COLOR_LINE, COLOR_BLINK, COLOR_TRK, COLOR_PERS, COLOR_LP,
    COCO_LABELS, TRACKED_CLASS_IDS, VEHICLE_CLASS_IDS,
    PLATE_MIN_H, PLATE_REQ_INTERVAL, LP_UPDATE_INTERVAL, MAX_LP_REQ_PER_FRAME,
    MAX_OCR_REQ_PER_FRAME, OCR_REQ_INTERVAL,
    M_PER_PX, LOG_BASE_DIR,
    ZoneConfigurator, is_in_poly, intersect,
)
from logger import Logger
from pipeline import (
    create_pipeline,
    send_plate_request, decode_lp_result,
    send_ocr_request,  decode_ocr_result,
)



# RUNTIME STATISTICS  (owned here; references passed into Logger & pipeline)

STATS = {
    "frames":                  0,
    "start_time":              None,
    "total_time":              0.0,

    "total_tracks_started":    0,
    "person_tracks_started":   0,
    "vehicle_tracks_started":  0,
    "line_crossings":          0,

    "plate_requests":          0,
    "plate_accepted":          0,
    "ocr_requests":            0,
    "ocr_nonempty":            0,
}




def main():
    
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video: {VIDEO_PATH}")
        return

    ret, first_raw = cap.read()
    if not ret:
        print("ERROR: Cannot read first frame")
        return
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    dt  = 1.0 / fps

    # Interactive zone / trip-line setup 
    first_nn = cv2.resize(first_raw, (VEH_NN_SIZE, VEH_NN_SIZE))
    configurator = ZoneConfigurator(first_nn)
    poly_pts, trip_lines = configurator.run()
    poly_contour = (
        np.array(poly_pts, dtype=np.int32).reshape((-1, 1, 2))
        if poly_pts else []
    )

    STATS["start_time"] = time.time()

    # DepthAI pipeline 
    print("DEBUG: Creating pipeline...")
    pipeline = create_pipeline()

    # Per-frame state
    plate_pending:  dict = {}
    plate_results:  dict = {}
    lp_last_req:    dict = {}
    ocr_pending:    dict = {}
    ocr_last_req:   dict = {}
    bbox_smooth:    dict = {}
    blink_states:   dict = {}
    track_history        = defaultdict(lambda: deque(maxlen=2))

    logger = Logger(LOG_BASE_DIR, STATS)

    with dai.Device(pipeline) as device:
        print("DEBUG: Device connected. Starting main loop...")

        q_in      = device.getInputQueue("in",       maxSize=1, blocking=False)
        q_trk     = device.getOutputQueue("tracklets", maxSize=4, blocking=False)
        q_lp_in   = device.getInputQueue("lp_in",    maxSize=4, blocking=False)
        q_lp_out  = device.getOutputQueue("lp_out",  maxSize=4, blocking=False)
        q_ocr_in  = device.getInputQueue("ocr_in",   maxSize=4, blocking=False)
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

                # Resize to NN space for inference
                frame_nn = cv2.resize(raw, (VEH_NN_SIZE, VEH_NN_SIZE))
                h_nn, w_nn = frame_nn.shape[:2]

                # Scale factors: NN coords → raw-frame coords
                scale_x = raw_w / float(VEH_NN_SIZE)
                scale_y = raw_h / float(VEH_NN_SIZE)

                # Send frame to vehicle detector
                img_frame = dai.ImgFrame()
                img_frame.setType(dai.ImgFrame.Type.BGR888p)
                img_frame.setWidth(VEH_NN_SIZE)
                img_frame.setHeight(VEH_NN_SIZE)
                img_frame.setData(frame_nn.transpose(2, 0, 1).flatten())
                try:
                    q_in.send(img_frame)
                except RuntimeError as exc:
                    if frame_idx % 30 == 0:
                        print(f"WARNING: send failed at frame {frame_idx}: {exc}")

                disp = raw.copy()

                # Drain async OCR results
                ocr_pkt = q_ocr_out.tryGet()
                while ocr_pkt is not None:
                    seq = ocr_pkt.getSequenceNum()
                    if seq in ocr_pending:
                        tid  = ocr_pending.pop(seq)
                        text = decode_ocr_result(ocr_pkt, STATS)
                        if text and tid in plate_results:
                            plate_results[tid]["text"] = text
                            print(f"[OCR] Track {tid}: {text}")
                    ocr_pkt = q_ocr_out.tryGet()

                # Drain async LP results 
                lp_pkt = q_lp_out.tryGet()
                while lp_pkt is not None:
                    seq = lp_pkt.getSequenceNum()
                    if seq in plate_pending:
                        info = plate_pending.pop(seq)
                        decode_lp_result(lp_pkt, info, plate_results, STATS)
                    lp_pkt = q_lp_out.tryGet()

                # Tracking output
                trk_data  = q_trk.tryGet()
                tracklets = trk_data.tracklets if trk_data is not None else []

                if frame_idx % 20 == 0:
                    print(f"[DEBUG] Frame {frame_idx}: tracklets={len(tracklets)}, "
                          f"plates={len(plate_results)}")

                # Draw monitoring zone & trip-lines
                if len(poly_contour) > 0:
                    poly_raw = (poly_contour * [scale_x, scale_y]).astype(np.int32)
                    cv2.polylines(disp, [poly_raw], True, COLOR_POLY, 2)

                now_ts = time.time()
                for p1, p2, lid in trip_lines:
                    p1r = (int(p1[0] * scale_x), int(p1[1] * scale_y))
                    p2r = (int(p2[0] * scale_x), int(p2[1] * scale_y))
                    col = (COLOR_BLINK
                           if lid in blink_states and now_ts < blink_states[lid]
                           else COLOR_LINE)
                    cv2.line(disp, p1r, p2r, col, 2)
                    cv2.putText(disp, lid, p1r, cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)

                # Per-tracklet processing
                lp_reqs_this_frame  = 0
                ocr_reqs_this_frame = 0

                for t in tracklets:
                    if t.status not in (dai.Tracklet.TrackingStatus.TRACKED,
                                        dai.Tracklet.TrackingStatus.NEW):
                        continue
                    if t.label not in TRACKED_CLASS_IDS:
                        continue

                    tid   = t.id
                    label = t.label

                    logical_id, pedVeh, prog, is_vehicle = logger.get_or_create_id(tid, label)

                    # Raw ROI in NN space, clamped to frame bounds
                    roi = t.roi.denormalize(w_nn, h_nn)
                    x1  = max(0, min(w_nn - 1, int(roi.topLeft().x)))
                    y1  = max(0, min(h_nn - 1, int(roi.topLeft().y)))
                    x2  = max(0, min(w_nn - 1, int(roi.bottomRight().x)))
                    y2  = max(0, min(h_nn - 1, int(roi.bottomRight().y)))

                    # EMA bbox smoothing (reduces jitter)
                    alpha = 0.4
                    if tid in bbox_smooth:
                        bx1, by1, bx2, by2 = bbox_smooth[tid]
                        x1 = int(alpha * x1 + (1 - alpha) * bx1)
                        y1 = int(alpha * y1 + (1 - alpha) * by1)
                        x2 = int(alpha * x2 + (1 - alpha) * bx2)
                        y2 = int(alpha * y2 + (1 - alpha) * by2)
                    bbox_smooth[tid] = (x1, y1, x2, y2)

                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2

                    inside_zone = (
                        len(poly_contour) == 0
                        or is_in_poly((cx, cy), poly_contour)
                    )

                    # Speed & distance estimation
                    key = (pedVeh, prog)
                    dist_inc_m = 0.0
                    speed_kmh  = 0.0
                    if key in logger.last_center:
                        prev_cx, prev_cy = logger.last_center[key]
                        dp         = math.hypot(cx - prev_cx, cy - prev_cy)
                        dist_inc_m = dp * M_PER_PX
                        speed_kmh  = (dist_inc_m / dt) * 3.6
                    logger.last_center[key] = (cx, cy)

                    start_t      = logger.start_time.get(key, time.time())
                    total_time_s = max(0.0, time.time() - start_t)

                    # Trip-line crossing detection
                    track_history[tid].append((cx, cy))
                    if len(track_history[tid]) == 2:
                        p_prev, p_curr = track_history[tid][0], track_history[tid][1]
                        for l_p1, l_p2, lid in trip_lines:
                            if intersect(p_prev, p_curr, l_p1, l_p2):
                                blink_states[lid] = time.time() + 0.5
                                logger.log_line_crossing(pedVeh, prog, lid, frame_nn)

                    # Draw detection box (raw-frame coords)
                    rx1 = int(x1 * scale_x)
                    ry1 = int(y1 * scale_y)
                    rx2 = int(x2 * scale_x)
                    ry2 = int(y2 * scale_y)

                    if inside_zone:
                        color_box  = COLOR_TRK if is_vehicle else COLOR_PERS
                        cv2.rectangle(disp, (rx1, ry1), (rx2, ry2), color_box, 2)

                        label_name = (COCO_LABELS[label]
                                      if 0 <= label < len(COCO_LABELS) else str(label))
                        label_text = f"{logical_id} {label_name}"

                        if is_vehicle and plate_results.get(tid, {}).get("text"):
                            label_text += f" [{plate_results[tid]['text']}]"
                        label_text += f" {speed_kmh:.1f}km/h"

                        cv2.putText(disp, label_text,
                                    (rx1, max(0, ry1 - 5)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_box, 2)

                        logger.log_position(pedVeh, prog, cx, cy,
                                            speed_kmh, dist_inc_m, total_time_s)

                    # LP inference trigger (vehicles only)
                    if is_vehicle and inside_zone:
                        h_box = y2 - y1
                        if h_box > PLATE_MIN_H:
                            last_req = lp_last_req.get(tid, -999_999)
                            interval = (LP_UPDATE_INTERVAL
                                        if tid in plate_results
                                        else PLATE_REQ_INTERVAL)
                            if (frame_idx - last_req >= interval
                                    and lp_reqs_this_frame < MAX_LP_REQ_PER_FRAME):
                                send_plate_request(
                                    q_lp_in, plate_pending,
                                    frame_idx, tid, frame_nn,
                                    x1, y1, x2, y2,
                                    STATS,
                                )
                                lp_last_req[tid]    = frame_idx
                                lp_reqs_this_frame += 1

                    # LP box overlay (raw-frame coords)
                    if is_vehicle and tid in plate_results and inside_zone:
                        rel_x1, rel_y1, rel_x2, rel_y2 = plate_results[tid]["rel"]
                        w_car_nn = x2 - x1
                        h_car_nn = y2 - y1

                        px1_nn = int(x1 + rel_x1 * w_car_nn)
                        py1_nn = int(y1 + rel_y1 * h_car_nn)
                        px2_nn = int(x1 + rel_x2 * w_car_nn)
                        py2_nn = int(y1 + rel_y2 * h_car_nn)

                        # Slight visual padding for readability
                        pad_w   = int(0.1 * (px2_nn - px1_nn))
                        pad_h   = int(0.2 * (py2_nn - py1_nn))
                        px1_nn  = max(0,        px1_nn - pad_w)
                        px2_nn  = min(w_nn - 1, px2_nn + pad_w)
                        py1_nn  = max(0,        py1_nn - pad_h)
                        py2_nn  = min(h_nn - 1, py2_nn + pad_h)

                        prx1 = int(px1_nn * scale_x)
                        pry1 = int(py1_nn * scale_y)
                        prx2 = int(px2_nn * scale_x)
                        pry2 = int(py2_nn * scale_y)

                        cv2.rectangle(disp, (prx1, pry1), (prx2, pry2), COLOR_LP, 2)
                        cv2.putText(disp, "LP", (prx1, max(0, pry1 - 5)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_LP, 1)

                    # OCR scheduling 
                    pr = plate_results.get(tid)
                    if (is_vehicle and pr is not None
                            and pr.get("plate_img") is not None
                            and not pr.get("text")
                            and ocr_reqs_this_frame < MAX_OCR_REQ_PER_FRAME):
                        last_ocr = ocr_last_req.get(tid, -999_999)
                        if frame_idx - last_ocr >= OCR_REQ_INTERVAL:
                            send_ocr_request(
                                q_ocr_in, ocr_pending,
                                frame_idx, tid, pr["plate_img"],
                                STATS,
                            )
                            ocr_last_req[tid]    = frame_idx
                            ocr_reqs_this_frame += 1

                # Compose display canvas (aspect-ratio preserving)
                s     = min(1.0, DISPLAY_W / float(raw_w), DISPLAY_H / float(raw_h))
                new_w = int(raw_w * s)
                new_h = int(raw_h * s)
                disp_scaled = cv2.resize(disp, (new_w, new_h)) if s != 1.0 else disp

                canvas   = np.zeros((DISPLAY_H, DISPLAY_W, 3), dtype=np.uint8)
                x_off    = (DISPLAY_W - new_w) // 2
                y_off    = (DISPLAY_H - new_h) // 2
                canvas[y_off:y_off + new_h, x_off:x_off + new_w] = disp_scaled

                cv2.imshow("Live", canvas)
                frame_idx += 1

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            logger.close()

    cap.release()
    cv2.destroyAllWindows()
    _print_stats()


# STATS REPORT

def _print_stats():
    if STATS["start_time"] is not None:
        STATS["total_time"] = time.time() - STATS["start_time"]

    print("\n========== RUNTIME STATS ==========")
    print(f"Frames processed : {STATS['frames']}")
    print(f"Total time       : {STATS['total_time']:.2f} s")
    if STATS["total_time"] > 0:
        print(f"Average FPS      : {STATS['frames'] / STATS['total_time']:.2f}")

    print("\n========== TRACKING STATS ==========")
    print(f"Unique tracks (all)   : {STATS['total_tracks_started']}")
    print(f"  — persons           : {STATS['person_tracks_started']}")
    print(f"  — vehicles          : {STATS['vehicle_tracks_started']}")
    print(f"Line crossings        : {STATS['line_crossings']}")

    print("\n========== PLATE / OCR STATS ==========")
    print(f"Plate requests sent   : {STATS['plate_requests']}")
    print(f"Plate detections acc. : {STATS['plate_accepted']}")
    print(f"OCR requests sent     : {STATS['ocr_requests']}")
    print(f"OCR non-empty results : {STATS['ocr_nonempty']}")
    print("=======================================\n")


if __name__ == "__main__":
    main()
