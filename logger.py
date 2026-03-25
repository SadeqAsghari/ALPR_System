import os
import cv2
import time
from collections import defaultdict
from datetime import datetime

from utils import VEHICLE_CLASS_IDS, fmt_X, fmt_9


class Logger:
    """
    Writes structured fixed-width log records to a daily `.tt` file and
    saves frame snapshots for line-crossing events.

    The caller owns the ``stats`` dict; Logger only mutates the tracking-related
    counters (total_tracks_started, vehicle_tracks_started, person_tracks_started,
    line_crossings).

    Log record types
    ----------------
    Identification  — written once when a new track ID is assigned
    Position        — written every frame the object is inside the zone
    Line-crossing   — written (with snapshot) when the track crosses a trip-line
    """

    def __init__(self, base_dir: str, stats: dict):
        self.stats = stats

        now = datetime.now()
        self.date_str = now.strftime("%Y-%m-%d")
        self.log_dir  = base_dir
        os.makedirs(self.log_dir, exist_ok=True)

        self.log_path   = os.path.join(self.log_dir, f"rilevazione_{self.date_str}.tt")
        self.frames_dir = os.path.join(self.log_dir, f"fg_{self.date_str}")
        os.makedirs(self.frames_dir, exist_ok=True)

        self._file = open(self.log_path, "a", encoding="utf-8")

        # Progressive ID counters (person / vehicle tracked separately)
        self.next_p = 1
        self.next_v = 1

        # tracker ID → (id_str, pedVeh, progressive, is_vehicle)
        self.track_map: dict = {}

        # Per-track accumulators used by the main loop for speed/distance
        self.pos_counter  = defaultdict(int)
        self.dist_accum_m = defaultdict(float)
        self.start_time:  dict = {}
        self.last_center: dict = {}

   
    # Public API
   
    def close(self):
        try:
            self._file.close()
        except Exception:
            pass

    def get_or_create_id(self, tid: int, label: int):
        """
        Return the logging tuple for tracker ID ``tid``, creating it on first
        sight.  Tuple format: (id_str, pedVeh, progressive, is_vehicle)
        """
        if tid in self.track_map:
            return self.track_map[tid]

        is_vehicle = label in VEHICLE_CLASS_IDS
        if is_vehicle:
            pedVeh = "V"
            prog   = self.next_v
            self.next_v += 1
            id_str = f"V{prog:07d}"
        else:
            pedVeh = "P"
            prog   = self.next_p
            self.next_p += 1
            id_str = f"P{prog:07d}"

        self.track_map[tid] = (id_str, pedVeh, prog, is_vehicle)

        # Update caller-owned stats
        self.stats["total_tracks_started"] += 1
        if is_vehicle:
            self.stats["vehicle_tracks_started"] += 1
        else:
            self.stats["person_tracks_started"] += 1

        self.log_identification(pedVeh, prog, label)
        self.start_time[(pedVeh, prog)] = time.time()
        return self.track_map[tid]

    def log_identification(self, pedVeh: str, prog: int, label: int,
                           plate_text: str = "", nationality: str = ""):
        """Write a single identification record."""
        if pedVeh == "V":
            t_veh = {2: "A", 3: "MO", 5: "BU", 7: "TR"}.get(label, "VV")
        else:
            t_veh = "PE"

        line = (
            fmt_X(pedVeh,      1)
            + fmt_9(prog,      7)
            + fmt_X(t_veh,     2)
            + fmt_X(nationality, 3)
            + fmt_X(plate_text,  9)
        )
        self._writeln(line)

    def log_position(self, pedVeh: str, prog: int,
                     x: int, y: int,
                     speed_kmh: float,
                     dist_increment_m: float,
                     total_time_s: float):
        """Write a position record and accumulate distance."""
        key = (pedVeh, prog)
        self.pos_counter[key] += 1
        pos_num = self.pos_counter[key]

        self.dist_accum_m[key] += max(0.0, dist_increment_m)
        total_dist_cm = int(self.dist_accum_m[key] * 100.0)
        speed_100     = int(max(0.0, speed_kmh * 100.0))

        line = (
            fmt_X(pedVeh,             1)
            + fmt_9(prog,             7)
            + fmt_9(pos_num,          3)
            + fmt_9(x,                5)
            + fmt_9(y,                5)
            + fmt_9(speed_100,        5)
            + fmt_9(total_dist_cm,    7)
            + fmt_9(int(total_time_s), 10)
        )
        self._writeln(line)

    def log_line_crossing(self, pedVeh: str, prog: int,
                          line_id: str,
                          frame_img_nn):
        """Write a line-crossing record and save the NN-space frame snapshot."""
        instant = int(time.time())
        line = (
            fmt_X("A",    1)
            + fmt_9(prog, 7)
            + fmt_9(instant, 10)
            + fmt_X(line_id,  5)
        )
        self._writeln(line)

        self.stats["line_crossings"] += 1

        img_name  = f"{prog:07d}_{instant}.jpeg"
        save_path = os.path.join(self.frames_dir, img_name)
        try:
            cv2.imwrite(save_path, frame_img_nn)
        except Exception as exc:
            print(f"[LOG] WARNING: cannot save frame {save_path}: {exc}")

    
    # Internal
    
    def _writeln(self, line: str):
        self._file.write(line + "\n")
        self._file.flush()
