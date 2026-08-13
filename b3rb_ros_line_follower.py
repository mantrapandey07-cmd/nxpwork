# Copyright 2024-2026 NXP
# Copyright 2016 Open Source Robotics Foundation, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import rclpy
from rclpy.node import Node
import math
from sensor_msgs.msg import Joy, LaserScan
from std_msgs.msg import String
from synapse_msgs.msg import EdgeVectors, ServerCommunication

QOS_PROFILE_DEFAULT = 10
PI = math.pi

# Control bounds
SPEED_MIN = 0.0
SPEED_MAX = 1.0
TURN_MIN = -1.0
TURN_MAX = 1.0

# DEBUG_LOG: flip to False to silence sign-board trace once verified working
DEBUG_LOG = True

# ── Sign-board ↔ Building-ID mappings (fixes bug #3) ──────────────
#   A → PATIENT_1   B → PATIENT_2   C → PATIENT_3
#   X → HOSPITAL_1  Y → HOSPITAL_2  Z → HOSPITAL_3
SIGN_TO_PATIENT  = {"A": 1, "B": 2, "C": 3}
PATIENT_TO_SIGN  = {1: "A", 2: "B", 3: "C"}
SIGN_TO_HOSPITAL = {"X": 1, "Y": 2, "Z": 3}
HOSPITAL_TO_SIGN = {1: "X", 2: "Y", 3: "Z"}


class LineFollower(Node):
    """
    Core controller Node for the B3RB buggy.

    Responsibilities:
      • Follow lane edge vectors.
      • Avoid static obstacles via LIDAR.
      • Interpret traffic sign boards at intersections.
      • Detect patient / hospital QR codes and verify against server assignment.
      • Communicate with the Municipality Server (rolling-uid ack protocol).
      • Park after mission completion.
    """

    def __init__(self):
        super().__init__('line_follower')

        # ═════════════ Subscriptions ═════════════

        self.subscription_vectors = self.create_subscription(
            EdgeVectors, '/edge_vectors',
            self.edge_vectors_callback, QOS_PROFILE_DEFAULT)

        self.subscription_lidar = self.create_subscription(
            LaserScan, '/scan',
            self.lidar_callback, QOS_PROFILE_DEFAULT)

        # Bug #1 fix: topic is /ServerCommunication (per README),
        # matching the dummy server after its fix.
        self.subscription_server = self.create_subscription(
            ServerCommunication, '/ServerCommunication',
            self.server_communication_callback, QOS_PROFILE_DEFAULT)

        self.subscription_qr = self.create_subscription(
            String, '/qr_detection',
            self.qr_detection_callback, QOS_PROFILE_DEFAULT)

        self.subscription_signs = self.create_subscription(
            String, '/sign_board_detection',
            self.sign_board_callback, QOS_PROFILE_DEFAULT)

        # ═════════════ Publishers ═════════════

        self.publisher_joy = self.create_publisher(
            Joy, '/cerebri/in/joy', QOS_PROFILE_DEFAULT)

        self.publisher_server = self.create_publisher(
            ServerCommunication, '/ServerCommunication', QOS_PROFILE_DEFAULT)

        # ═════════════ Control state ═════════════

        self.target_speed = 0.3
        self.target_turn  = 0.0

        # ── Mission state ──
        self.avoid             = False
        self.patient_id        = None
        self.hospital_id       = None
        # Bug #3 fix: destination holds a sign-board LETTER (A/B/C/X/Y/Z),
        # never a raw server string.  Buggy defaults to "A" (Patient 1).
        self.destination       = "A"
        self.mission_completed = False
        self.approaching      = False
        self.qr_data           = ""
        self.reached           = False
        self.parked_sent       = False
        self.parked_confirmed  = False
        self.park_timer_start  = None

        # ── Tuning constants ──
        self.expconst  = 0.4
        self.expconst2 = 0.2

        # ── Lane-following state ──
        self.direction      = ""
        self.lost_count     = 0
        self.stick_to_lane  = False
        self.path_width     = 0.5
        self.pending_turn   = ""
        self.turn_till      = 0.8   # seconds to stay biased after arming a turn
        self.timee          = 0.0   # timestamp when turn bias started

        # ── Sign-board debounce ──
        self.sign_candidate       = None
        self.sign_candidate_count = 0

        # Bug #5 fix: rolling UID counter for outgoing messages
        self.uid_counter = 0

        # ═════════════ Timer ═════════════
        self.control_timer = self.create_timer(0.1, self.publish_drive_commands)

        self.get_logger().info(
            "Line Follower initialized. "
            f"Destination sign='{self.destination}' (Patient 1). "
            "Safe Drive-Straight Mode active."
        )

    # ════════════════════════════════════════════════════════════════
    #  Drive-command publisher
    # ════════════════════════════════════════════════════════════════

    def publish_drive_commands(self):
        """Timer callback (10 Hz) — publishes current speed / steer."""
        msg = Joy()
        msg.buttons = [1, 0, 0, 0, 0, 0, 0, 1]
        msg.axes = [0.0, self.target_speed, 0.0, self.target_turn]
        self.publisher_joy.publish(msg)

    def rover_move_manual_mode(self, speed, turn):
        """Helper to immediately set control speed and steering angle."""
        self.target_speed = float(max(min(speed, SPEED_MAX), -SPEED_MAX))
        self.target_turn  = float(max(min(turn,  TURN_MAX),  -TURN_MAX))

    # ════════════════════════════════════════════════════════════════
    #  Server Communication  (fixes bugs #1, #3, #4, #5, #6)
    # ════════════════════════════════════════════════════════════════

    def send_server_update(self, text_msg, ack=0, uid=None):
        """
        Send a message to the server.

        Bug #4 fix: ``ack`` is now an explicit parameter — callers decide
        whether this is a fresh report (ack=0) or an acknowledgement
        (ack=1).  No more global ``self.ackno`` flag.

        Bug #5 fix: ``uid`` rolls automatically via ``self.uid_counter``
        when ``uid=None``.  Pass an explicit ``uid`` only when acking a
        specific server message (echoing its uid).
        """
        server_msg = ServerCommunication()
        server_msg.src  = 1      # Buggy
        server_msg.dest = 2      # Server
        if uid is None:
            self.uid_counter = (self.uid_counter + 1) % 256
            if self.uid_counter == 0:
                self.uid_counter = 1
            server_msg.uid = self.uid_counter
        else:
            server_msg.uid = uid
        server_msg.ack = ack
        server_msg.msg = text_msg
        self.publisher_server.publish(server_msg)
        self.get_logger().info(
            f"SENT -> uid={server_msg.uid} ack={ack} msg='{text_msg}'"
        )

    def server_communication_callback(self, message):
        """
        Receive coordination commands from the server.

        Protocol:
          • ack=1 from server  →  server is acking *our* previous send.
          • ack=0 from server  →  fresh instruction; we must ack it back
            (echoing the same uid) and then process the payload.
        """
        if message.dest != 1 or message.src != 2:
            return

        self.get_logger().info(
            f"RECV <- uid={message.uid} ack={message.ack} msg='{message.msg}'"
        )

        # ── Server ACK (bug #4 fix: don't set ackno=1 globally) ──
        if message.ack == 1:
            if message.msg == "OK":
                self.parked_confirmed = True
                self.get_logger().info("Parking confirmed by server! ✓")
            elif message.msg == "INVALID":
                self.get_logger().warn("Parking INVALID — will retry.")
                self.parked_sent = False
            else:
                # Regular ack for our QR report — clear sent QR data
                self.qr_data = ""
            return

        # ── Fresh instruction (ack=0) ──
        # Bug #4 fix: ack the server's message explicitly with its uid
        self.send_server_update("", ack=1, uid=message.uid)

        msg_text = message.msg.strip()

        # ── Hospital assignment (bug #3 fix) ──
        # Server sends "HOSPITAL_2" → buggy sets destination="Y"
        if msg_text.startswith("HOSPITAL_"):
            try:
                hid = int(msg_text.split("_")[-1])
                self.hospital_id = hid
                self.patient_id  = None
                self.destination = HOSPITAL_TO_SIGN.get(hid)
                self.reached     = False
                self.approaching = False
                self.target_speed = 0.3          # resume from any stop
                self.get_logger().info(
                    f"Assigned HOSPITAL_{hid}, sign='{self.destination}'"
                )
            except (ValueError, KeyError) as e:
                self.get_logger().warn(f"Bad hospital msg '{msg_text}': {e}")

        # ── Patient assignment ──
        # Server sends "PATIENT_2" → buggy sets destination="B"
        elif msg_text.startswith("PATIENT_"):
            try:
                pid = int(msg_text.split("_")[-1])
                self.patient_id   = pid
                self.hospital_id  = None
                self.destination  = PATIENT_TO_SIGN.get(pid)
                self.reached      = False
                self.approaching = False
                self.target_speed = 0.3
                self.get_logger().info(
                    f"Assigned PATIENT_{pid}, sign='{self.destination}'"
                )
            except (ValueError, KeyError) as e:
                self.get_logger().warn(f"Bad patient msg '{msg_text}': {e}")

        elif msg_text == "MISSION_COMPLETE":
            self.mission_completed = True
            self.destination        = None
            self.park_timer_start   = None
            self.get_logger().info(
                "Mission complete! Navigating to parking area."
            )

        else:
            self.get_logger().warn(f"Unknown server message: '{msg_text}'")

    # ════════════════════════════════════════════════════════════════
    #  QR Detection  (fixes bugs #2, #3, #7, #8)
    # ════════════════════════════════════════════════════════════════

    @staticmethod
    def _parse_qr_location(data):
        """
        Robustly parse a QR payload like ``{LOC: PATIENT_1}`` or
        ``{LOC: HOSPITAL_2}`` and return ``(type_str, id_int)``.

        Returns ``(None, None)`` if parsing fails.

        Bug #2 fix: handles the real ``{LOC: PATIENT_1}`` format
        (colon + space) instead of ``{LOC_PATIENT_1}``.
        Bug #7 fix: robust split-based parsing, not brittle strip/split.
        """
        s = data.strip()
        # Strip outer braces
        if s.startswith("{") and s.endswith("}"):
            s = s[1:-1].strip()
        # Split on first colon: "LOC" : "PATIENT_1"
        if ":" not in s:
            return None, None
        _, value = s.split(":", 1)
        value = value.strip()                      # "PATIENT_1"
        parts = value.split("_")                   # ["PATIENT", "1"]
        if len(parts) < 2:
            return None, None
        loc_type = parts[0].upper()                # "PATIENT" / "HOSPITAL" / "FAKE"
        try:
            loc_id = int(parts[-1])
        except ValueError:
            return None, None
        return loc_type, loc_id

    def qr_detection_callback(self, message):
        """
        Receive QR codes scanned from buildings.

        Real QR format (per README): ``{LOC: PATIENT_1}``
        """
        self.get_logger().info(f"Heard QR code: {message.data}")
        loc_type, loc_id = self._parse_qr_location(message.data)

        if loc_type is None:
            return

        # ── Patient QR ──
        if loc_type == "PATIENT":
            self.patient_id = loc_id
            self.get_logger().info(f"Identified Patient: {loc_id}")

            if self.destination in ("A", "B", "C"):
                expected = SIGN_TO_PATIENT[self.destination]
                if loc_id == expected:
                    self.get_logger().info(
                        f"Approaching target patient {loc_id}"
                    )
                    # Bug #8 fix: removed duplicate self.approaching = True
                    self.approaching = True
                    self.avoid       = False
                    self.qr_data     = message.data
                else:
                    self.get_logger().info(
                        f"Patient {loc_id} ≠ expected {expected}; ignoring."
                    )
            else:
                self.approaching = False

        # ── Hospital QR ──
        elif loc_type == "HOSPITAL":
            self.hospital_id = loc_id
            self.get_logger().info(f"Identified Hospital: {loc_id}")

            if self.destination in ("X", "Y", "Z"):
                expected = SIGN_TO_HOSPITAL[self.destination]
                if loc_id == expected:
                    self.get_logger().info(
                        f"Approaching target hospital {loc_id}"
                    )
                    self.approaching = True
                    self.avoid       = False
                    self.qr_data     = message.data
                else:
                    self.get_logger().info(
                        f"Hospital {loc_id} ≠ expected {expected}; ignoring."
                    )
            else:
                self.approaching = False

        # ── Fake Hospital QR ──
        elif loc_type == "FAKE":
            self.get_logger().warn(f"Ignoring FAKE hospital QR: {message.data}")
            self.approaching = False

    # ════════════════════════════════════════════════════════════════
    #  LIDAR  (fixes bug #6; documents bug #10)
    # ════════════════════════════════════════════════════════════════

    def lidar_callback(self, message):
        """
        LIDAR range processing for obstacle avoidance and building approach.

        NOTE (bug #10): Two different sector conventions are used:
          • Avoidance  — front-centre sectors around index n/2
          • Approach   — wider side sectors (building walls are off-centre)
        Verify these against the actual LIDAR angle convention of the B3RB
        simulation before relying on them in competition.
        """
        if not message.ranges:
            return
        n = len(message.ranges)

        # ── Parking detection (post-mission) ──
        if self.mission_completed and not self.parked_sent:
            if self.park_timer_start is None:
                self.park_timer_start = self.get_clock().now().nanoseconds / 1e9
            elapsed = self.get_clock().now().nanoseconds / 1e9 - self.park_timer_start
            if elapsed > 3.0:   # grace period to reach parking zone
                front = message.ranges[int(n * 8/18):int(n * 10/18)]
                valid = [r for r in front if r > 0]
                if valid and min(valid) < 0.5:
                    self.target_speed = 0
                    self.send_server_update("PARKED", ack=0)
                    self.parked_sent = True
                    self.get_logger().info("PARKED message sent!")
            return

        if self.mission_completed:
            return

        # ── Obstacle avoidance (normal driving) ──
        if not self.approaching:
            right_sector = message.ranges[int(n * 7/18):int(n * 9/18)]
            left_sector  = message.ranges[int(n * 9/18):int(n * 11/18)]

            min_right = min(right_sector) if right_sector else float('inf')
            min_left  = min(left_sector)  if left_sector  else float('inf')
            min_front = min(min_right, min_left)

            if min_front < 0.8:
                sector     = right_sector if min_right < min_left else left_sector
                lowerbound = n * 7/18 if sector is right_sector else n * 9/18
                min_val    = min(sector)
                offset     = 9 * n / 18 - (sector.index(min_val) + lowerbound)

                if offset != 0:
                    ang = (1 - abs(offset) / (2 * n / 18)) * (abs(offset) / offset)
                else:
                    ang = 0.1 if lowerbound == n * 7/18 else -0.1

                spd   = min(self.target_speed,
                            min_front * self.expconst / 0.8
                            + (1 - self.expconst) * self.target_speed)
                angle = self.expconst * ang + (1 - self.expconst) * self.target_turn
                self.avoid = True
                self.rover_move_manual_mode(spd, angle)
            else:
                self.avoid = False

        # ── Building approach mode ──
        else:
            left_part  = message.ranges[int(n * 10/18):int(n * 14/18)]
            right_part = message.ranges[int(n * 4/18):int(n * 8/18)]

            min_left  = min(left_part)  if left_part  else float('inf')
            min_right = min(right_part) if right_part else float('inf')
            min_side  = min(min_left, min_right)

            if min_side < 3.0:
                if min_left < min_right:
                    idx = int(n * 10/18) + left_part.index(min_left)
                else:
                    idx = int(n * 4/18) + right_part.index(min_right)

                offset = (n / 2 - idx) / (4 * n / 18)
                spd    = self.target_speed * (1 - abs(offset))
                self.rover_move_manual_mode(spd, self.target_turn)

                # Bug #6 fix: send QR exactly once, then clear reached
                if min_side < 0.5 and not self.reached:
                    self.target_speed = 0
                    self.approaching  = False
                    self.reached       = True
                    if self.qr_data:
                        self.send_server_update(self.qr_data, ack=0)
                        self.get_logger().info(
                            f"Reached building — sent QR: {self.qr_data}"
                        )

    # ════════════════════════════════════════════════════════════════
    #  Edge Vectors  (fixes bugs #11, #12)
    # ════════════════════════════════════════════════════════════════

    def edge_vectors_callback(self, message):
        if self.avoid:
            return

        # ── Detect turn onset from pending_turn ──
        m1, m2 = 0.0, 0.0
        if self.pending_turn != "" and self.pending_turn != "Straight":
            if message.vector_count == 2:
                dy1 = message.vector_1[1].y - message.vector_1[0].y
                dy2 = message.vector_2[1].y - message.vector_2[0].y
                m1 = (math.atan((message.vector_1[1].x - message.vector_1[0].x) / dy1) / (PI/2)
                      if dy1 != 0 else 0)
                m2 = (math.atan((message.vector_2[1].x - message.vector_2[0].x) / dy2) / (PI/2)
                      if dy2 != 0 else 0)
            elif message.vector_count == 1:
                vec = message.vector_1 if message.vector_1 else message.vector_2
                dy  = vec[1].y - vec[0].y
                m1  = math.atan((vec[1].x - vec[0].x) / dy) / (PI/2) if dy != 0 else 0

        if (abs(m1) > 0.3 or abs(m2) > 0.3) and self.pending_turn != "" \
                and self.pending_turn != "Straight":
            self.direction     = self.pending_turn
            self.pending_turn  = ""
            self.stick_to_lane = True
            self.timee         = self.get_clock().now().nanoseconds / 1e9

        # ── Execute turn (stick_to_lane mode) ──
        # Bug #11 fix: time-based auto-reset already exists via turn_till;
        # the else branch below clears stick_to_lane when the time window
        # expires, so the buggy does NOT stay biased indefinitely.
        if self.stick_to_lane:
            now_sec = self.get_clock().now().nanoseconds / 1e9

            if self.timee + self.turn_till > now_sec:
                farpoint = None

                # ── Left turn ──
                if self.direction == "Left":
                    if message.vector_count == 1:
                        farpoint = message.vector_1[0] if message.vector_1 else message.vector_2[0]
                    elif message.vector_count == 2:
                        if (message.vector_1[0].x - message.image_width / 2) < \
                           (message.vector_2[0].x - message.image_width / 2):
                            farpoint = message.vector_1[0]
                        else:
                            farpoint = message.vector_2[0]

                    if farpoint is not None:
                        dx = farpoint.x - message.image_width / 2
                        dy = message.image_height - farpoint.y
                        if dy == 0:
                            return
                        ofs   = dx + self.path_width / 5
                        ang   = -math.atan(ofs / dy) / (PI / 2)
                        angle = self.expconst * ang + (1 - self.expconst) * self.target_turn
                        speed = 1 - abs(ang) * 0.8
                        spd   = speed * self.expconst + (1 - self.expconst) * self.target_speed
                        self.rover_move_manual_mode(spd, angle)
                        return

                # ── Right turn ──
                if self.direction == "Right":
                    if message.vector_count == 1:
                        farpoint = message.vector_1[0] if message.vector_1 else message.vector_2[0]
                    elif message.vector_count == 2:
                        if (message.vector_1[0].x - message.image_width / 2) > \
                           (message.vector_2[0].x - message.image_width / 2):
                            farpoint = message.vector_1[0]
                        else:
                            farpoint = message.vector_2[0]

                    if farpoint is not None:
                        dx = farpoint.x - message.image_width / 2
                        dy = message.image_height - farpoint.y
                        if dy == 0:
                            return
                        ofs   = dx - self.path_width / 5
                        ang   = -math.atan(ofs / dy) / (PI / 2)
                        angle = self.expconst * ang + (1 - self.expconst) * self.target_turn
                        speed = 1 - abs(ang) * 0.8
                        spd   = speed * self.expconst + (1 - self.expconst) * self.target_speed
                        self.rover_move_manual_mode(spd, angle)
                        return
            else:
                # Time window expired — turn is geometrically complete
                self.stick_to_lane = False

        # ── Normal lane following ──

        if message.vector_count == 0:
            # Bug #12 fix: removed duplicate rover_move_manual_mode call
            self.lost_count += 1
            decay = min(self.lost_count / 10, 1.0)
            dirn  = 1.0 if self.direction == "Left" else (
                    -1.0 if self.direction == "Right" else 0.0)
            angle = self.expconst * decay * dirn + (1 - self.expconst) * self.target_turn
            self.rover_move_manual_mode(self.target_speed, angle)

        elif message.vector_count == 1:
            farpoint = message.vector_1[0] if message.vector_1 else message.vector_2[0]
            dx = farpoint.x - message.image_width / 2
            dy = message.image_height - farpoint.y
            if dy == 0:
                return
            raw_ang = math.atan(dx / dy) / (PI / 2)
            ang     = raw_ang if abs(raw_ang) > 0.2 else 0.0
            angle   = self.expconst * ang + (1 - self.expconst) * self.target_turn
            if not self.approaching:
                speed = 1 - abs(ang) * 0.8
                spd   = speed * self.expconst + (1 - self.expconst) * self.target_speed
            else:
                spd = self.target_speed
            self.lost_count = 0
            self.rover_move_manual_mode(spd, angle)

        elif message.vector_count == 2:
            # Update path width estimate
            if (message.vector_1[0].x - message.image_width / 2) * \
               (message.vector_2[0].x - message.image_width / 2) < 0:
                o = abs((message.vector_1[0].x - message.image_width / 2) -
                        (message.vector_2[0].x - message.image_width / 2))
                self.path_width = o if o > self.path_width else self.path_width

            farx = (message.vector_1[0].x + message.vector_2[0].x) / 2
            fary = (message.vector_1[0].y + message.vector_2[0].y) / 2
            dx   = farx - message.image_width / 2
            dy   = message.image_height - fary
            if dy == 0:
                return
            ang   = math.atan(dx / dy) / (PI / 2)
            angle = -self.expconst2 * ang + (1 - self.expconst2) * self.target_turn
            if not self.approaching:
                speed = 1 - abs(ang) * 0.8
                spd   = speed * self.expconst2 + (1 - self.expconst2) * self.target_speed
            else:
                spd = self.target_speed
            self.lost_count = 0
            self.rover_move_manual_mode(spd, angle)

    # ════════════════════════════════════════════════════════════════
    #  Sign Board
    # ════════════════════════════════════════════════════════════════

    def sign_board_callback(self, message):
        """
        Receive traffic sign boards.  The object recognizer publishes
        semicolon-separated ``Label:x_position`` entries, e.g.:
            "Left:0.3;A:0.4;Right:0.7"

        The runner finds the entry matching ``self.destination`` (A/B/C/X/Y/Z)
        and selects the nearest direction (Left/Right/Straight) to it.
        """
        self.get_logger().info(f"Heard Sign Board: {message.data}")

        entries = []
        for e in message.data.split(";"):
            e = e.strip()
            if not e:
                continue
            parts = e.split(":", 1)
            if len(parts) != 2:
                continue
            try:
                entries.append([parts[0].strip(), float(parts[1])])
            except ValueError:
                continue

        # Find the entry matching our current destination sign
        destentry = None
        for e in entries:
            if e[0] == self.destination:
                destentry = e

        candidates = [e for e in entries if e[0] in ("Left", "Right", "Straight")]
        if not candidates or destentry is None:
            if DEBUG_LOG:
                self.get_logger().info(
                    f"[DEBUG] no usable match — entries={entries} "
                    f"dest={self.destination}"
                )
            return

        nearest = min(candidates, key=lambda e: abs(e[1] - destentry[1]))
        if DEBUG_LOG:
            self.get_logger().info(
                f"[DEBUG] destentry={destentry} nearest={nearest}"
            )

        if abs(nearest[1] - destentry[1]) < 0.1:
            direction = nearest[0]

            # ── Straight: clear any turn bias ──
            if direction == 'Straight':
                self.direction             = 'Straight'
                self.stick_to_lane         = False
                self.sign_candidate        = None
                self.sign_candidate_count  = 0
                return

            # ── Debounce: require 3 consecutive matching detections ──
            if direction == self.sign_candidate:
                self.sign_candidate_count += 1
            else:
                self.sign_candidate       = direction
                self.sign_candidate_count = 1

            if DEBUG_LOG:
                self.get_logger().info(
                    f"[DEBUG] {direction} count="
                    f"{self.sign_candidate_count}/3"
                )

            if self.sign_candidate_count < 3:
                return

            self.pending_turn = direction
            self.get_logger().info(f"Pending turn armed: {direction}")


def main(args=None):
    rclpy.init(args=args)
    node = LineFollower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
