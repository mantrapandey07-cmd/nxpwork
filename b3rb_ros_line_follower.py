#!/usr/bin/env python3

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

import math
from typing import Optional, Tuple, List

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Joy, LaserScan
from std_msgs.msg import String
from synapse_msgs.msg import EdgeVectors, ServerCommunication


QOS_PROFILE_DEFAULT = 10

# ---------------------------------------------------------------------------
# Joy/control limits
# ---------------------------------------------------------------------------
SPEED_MIN = -1.0
SPEED_MAX = 1.0
TURN_MIN = -1.0
TURN_MAX = 1.0

# ---------------------------------------------------------------------------
# Mission/control tuning
# ---------------------------------------------------------------------------
CONTROL_PERIOD = 0.10                 # 10 Hz

BASE_SPEED = 0.15
APPROACH_SPEED = 0.10
MAX_TURN_SPEED_REDUCTION = 0.80

# Edge-vector steering smoothing.
# Higher = more responsive, lower = smoother.
LINE_ALPHA_ONE = 0.35
LINE_ALPHA_TWO = 0.35

# When the line is lost, remember the last turn direction and search gradually.
LINE_LOST_MAX_COUNT = 15
LINE_SEARCH_MAX_TURN = 0.55

# Sign-board turn override.
TURN_OVERRIDE_DURATION = 1.5
SIGN_TURN_VALUE = 0.70
SIGN_TURN_SPEED = 0.12

# ---------------------------------------------------------------------------
# LiDAR obstacle avoidance
# ---------------------------------------------------------------------------
OBSTACLE_DETECT_DISTANCE = 0.80
OBSTACLE_CLEAR_DISTANCE = 1.10
OBSTACLE_SIDE_CLEAR_DISTANCE = 0.75
OBSTACLE_TURN_SPEED = 0.10
OBSTACLE_MAX_TURN = 0.75
OBSTACLE_MAX_TIME = 8.0

# ---------------------------------------------------------------------------
# Destination/building approach
# ---------------------------------------------------------------------------
# QR detection tells us that this is the target. LiDAR then controls the
# approach. These values should be tuned against the actual buggy/arena.
DESTINATION_SLOW_DISTANCE = 3.0
DESTINATION_STOP_DISTANCE = 1.20
DESTINATION_HARD_STOP_DISTANCE = 0.75

# ---------------------------------------------------------------------------
# Mission states
# ---------------------------------------------------------------------------
STATE_FOLLOW = "FOLLOW"
STATE_AVOID = "AVOID"
STATE_APPROACH = "APPROACH"
STATE_STOPPED = "STOPPED"
STATE_COMPLETE = "COMPLETE"


class LineFollower(Node):
    """
    Main ROS2 controller for the B3RB buggy.

    Architecture:
        perception callbacks update state only
        -> one 10 Hz control loop decides the actual Joy command.

    This prevents edge-vector, LiDAR, QR and server callbacks from fighting
    over target_speed/target_turn asynchronously.
    """

    def __init__(self):
        super().__init__("line_follower")

        # ------------------------------------------------------------------
        # Subscriptions
        # ------------------------------------------------------------------
        self.subscription_vectors = self.create_subscription(
            EdgeVectors,
            "/edge_vectors",
            self.edge_vectors_callback,
            QOS_PROFILE_DEFAULT,
        )

        self.subscription_lidar = self.create_subscription(
            LaserScan,
            "/scan",
            self.lidar_callback,
            QOS_PROFILE_DEFAULT,
        )

        self.subscription_server = self.create_subscription(
            ServerCommunication,
            "/ServerCommunication",
            self.server_communication_callback,
            QOS_PROFILE_DEFAULT,
        )

        self.subscription_qr = self.create_subscription(
            String,
            "/qr_detection",
            self.qr_detection_callback,
            QOS_PROFILE_DEFAULT,
        )

        self.subscription_signs = self.create_subscription(
            String,
            "/sign_board_detection",
            self.sign_board_callback,
            QOS_PROFILE_DEFAULT,
        )

        # ------------------------------------------------------------------
        # Publishers
        # ------------------------------------------------------------------
        self.publisher_joy = self.create_publisher(
            Joy,
            "/cerebri/in/joy",
            QOS_PROFILE_DEFAULT,
        )

        self.publisher_server = self.create_publisher(
            ServerCommunication,
            "/ServerCommunication",
            QOS_PROFILE_DEFAULT,
        )

        # ------------------------------------------------------------------
        # Controller state
        # ------------------------------------------------------------------
        self.state = STATE_FOLLOW

        self.target_speed = BASE_SPEED
        self.target_turn = 0.0

        # Latest line-following command/state.
        self.line_turn = 0.0
        self.line_speed = BASE_SPEED
        self.last_turn_direction = 0.0
        self.lost_count = 0

        # Latest LiDAR information.
        self.front_distance = float("inf")
        self.left_distance = float("inf")
        self.right_distance = float("inf")
        self.obstacle_distance = float("inf")
        self.obstacle_side = "left"
        self.lidar_valid = False

        # Obstacle avoidance state.
        self.avoid_turn = 0.0
        self.avoid_start_time = 0.0

        # Mission state.
        self.patient_id: Optional[int] = None
        self.hospital_id: Optional[int] = None

        # Preserve the original project convention:
        # A/B/C -> patient 1/2/3
        # X/Y/Z -> hospital 1/2/3
        self.destination: Optional[str] = "A"

        self.qr_data = ""
        self.reached = False
        self.mission_completed = False
        self.arrival_reported = False

        # Sign-board routing state.
        self.direction = "Straight"
        self.override_till = 0.0

        # Server communication.
        self.ack_pending = False
        self.next_uid = 1
        self.last_received_uid = None

        # Timer.
        self.control_timer = self.create_timer(
            CONTROL_PERIOD,
            self.publish_drive_commands,
        )

        self.get_logger().info(
            "Line follower controller initialized. State=FOLLOW"
        )

    # ======================================================================
    # Utility functions
    # ======================================================================

    @staticmethod
    def clamp(value: float, low: float, high: float) -> float:
        return max(low, min(high, value))

    def now_sec(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def set_command(self, speed: float, turn: float) -> None:
        self.target_speed = self.clamp(speed, SPEED_MIN, SPEED_MAX)
        self.target_turn = self.clamp(turn, TURN_MIN, TURN_MAX)

    def stop(self) -> None:
        self.target_speed = 0.0
        self.target_turn = 0.0

    def publish_drive_commands(self) -> None:
        """
        Single owner of the final drive command.

        Callbacks update state; this function selects the command based on
        the current state and publishes at a fixed 10 Hz rate.
        """
        if self.mission_completed or self.state == STATE_COMPLETE:
            self.stop()

        elif self.state == STATE_STOPPED:
            self.stop()

        elif self.state == STATE_AVOID:
            self.control_obstacle_avoidance()

        elif self.state == STATE_APPROACH:
            self.control_destination_approach()

        else:
            self.control_line_following()

        msg = Joy()
        msg.buttons = [1, 0, 0, 0, 0, 0, 0, 1]
        msg.axes = [0.0, self.target_speed, 0.0, self.target_turn]
        self.publisher_joy.publish(msg)

    # ======================================================================
    # Edge-vector / lane following
    # ======================================================================

    def edge_vectors_callback(self, message: EdgeVectors) -> None:
        """
        Process the latest edge-vector detection.

        This callback never directly publishes or drives the buggy.
        """
        if self.state in (STATE_STOPPED, STATE_COMPLETE):
            return

        if message.vector_count == 0:
            self.lost_count += 1

            # Keep the last useful turn direction instead of always searching
            # to the same side.
            if self.last_turn_direction == 0.0:
                self.last_turn_direction = (
                    1.0 if self.line_turn >= 0.0 else -1.0
                )

            decay = min(
                self.lost_count / float(LINE_LOST_MAX_COUNT),
                1.0,
            )

            self.line_turn = self.clamp(
                self.last_turn_direction * LINE_SEARCH_MAX_TURN * decay,
                TURN_MIN,
                TURN_MAX,
            )

            # Slow down while the line is lost.
            self.line_speed = max(
                0.08,
                BASE_SPEED * (1.0 - 0.60 * decay),
            )
            return

        # --------------------------------------------------------------
        # One edge vector
        # --------------------------------------------------------------
        if message.vector_count == 1:
            points = list(message.vector_1) if message.vector_1 else list(message.vector_2)

            if not points:
                self.lost_count += 1
                return

            farpoint = points[0]

            dx = farpoint.x - (message.image_width / 2.0)
            dy = message.image_height - farpoint.y

            if abs(dy) < 1e-6:
                dy = 1e-6

            raw_angle = math.atan2(dx, dy) / (math.pi / 2.0)
            raw_angle = self.clamp(raw_angle, -1.0, 1.0)

            # Smooth deadband without the discontinuity of:
            # "if abs(angle) < 0.2: angle = 0"
            if abs(raw_angle) < 0.05:
                raw_angle = 0.0

            self.line_turn = (
                LINE_ALPHA_ONE * raw_angle
                + (1.0 - LINE_ALPHA_ONE) * self.line_turn
            )

            self.line_turn = self.clamp(
                self.line_turn,
                TURN_MIN,
                TURN_MAX,
            )

            self.last_turn_direction = (
                1.0 if self.line_turn > 0.02
                else -1.0 if self.line_turn < -0.02
                else self.last_turn_direction
            )

            self.line_speed = self.speed_from_turn(self.line_turn)
            self.lost_count = 0
            return

        # --------------------------------------------------------------
        # Two edge vectors
        # --------------------------------------------------------------
        if message.vector_count == 2:
            if not message.vector_1 or not message.vector_2:
                return

            p1 = message.vector_1[0]
            p2 = message.vector_2[0]

            farx = (p1.x + p2.x) / 2.0
            fary = (p1.y + p2.y) / 2.0

            dx = farx - (message.image_width / 2.0)
            dy = message.image_height - fary

            if abs(dy) < 1e-6:
                dy = 1e-6

            raw_angle = math.atan2(dx, dy) / (math.pi / 2.0)
            raw_angle = self.clamp(raw_angle, -1.0, 1.0)

            self.line_turn = (
                LINE_ALPHA_TWO * raw_angle
                + (1.0 - LINE_ALPHA_TWO) * self.line_turn
            )

            self.line_turn = self.clamp(
                self.line_turn,
                TURN_MIN,
                TURN_MAX,
            )

            self.last_turn_direction = (
                1.0 if self.line_turn > 0.02
                else -1.0 if self.line_turn < -0.02
                else self.last_turn_direction
            )

            self.line_speed = self.speed_from_turn(self.line_turn)
            self.lost_count = 0

    def speed_from_turn(self, turn: float) -> float:
        """
        Reduce forward speed smoothly as steering demand increases.
        """
        reduction = MAX_TURN_SPEED_REDUCTION * min(abs(turn), 1.0)
        return self.clamp(
            BASE_SPEED * (1.0 - reduction),
            0.07,
            BASE_SPEED,
        )

    def control_line_following(self) -> None:
        # A sign-board left/right override temporarily takes priority.
        now = self.now_sec()

        if now < self.override_till:
            if self.direction == "Left":
                turn = SIGN_TURN_VALUE
            elif self.direction == "Right":
                turn = -SIGN_TURN_VALUE
            else:
                turn = self.line_turn

            self.set_command(SIGN_TURN_SPEED, turn)
            return

        self.set_command(self.line_speed, self.line_turn)

    # ======================================================================
    # LiDAR processing
    # ======================================================================

    def get_sector(
        self,
        scan: LaserScan,
        center_deg: float,
        half_width_deg: float,
    ) -> List[float]:
        """
        Extract a LiDAR angular sector using LaserScan metadata.

        This avoids assuming that n/2 is the front. The scan may be ordered
        differently depending on the LiDAR driver.
        """
        if not scan.ranges or scan.angle_increment == 0.0:
            return []

        center = math.radians(center_deg)
        half_width = math.radians(half_width_deg)

        values = []

        for i, distance in enumerate(scan.ranges):
            angle = scan.angle_min + i * scan.angle_increment

            # Normalize angular difference to [-pi, pi].
            diff = math.atan2(
                math.sin(angle - center),
                math.cos(angle - center),
            )

            if abs(diff) <= half_width:
                if math.isfinite(distance):
                    if scan.range_min <= distance <= scan.range_max:
                        values.append(distance)

        return values

    @staticmethod
    def safe_min(values: List[float]) -> float:
        if not values:
            return float("inf")
        return min(values)

    def lidar_callback(self, message: LaserScan) -> None:
        """
        Update obstacle/building distances.

        Assumption:
            LaserScan angle 0 rad is the buggy's forward direction, which is
            the normal ROS LaserScan convention. If the physical LiDAR is
            mounted with a yaw offset, compensate for that offset here.
        """
        front = self.get_sector(message, 0.0, 25.0)
        left = self.get_sector(message, 60.0, 25.0)
        right = self.get_sector(message, -60.0, 25.0)

        self.front_distance = self.safe_min(front)
        self.left_distance = self.safe_min(left)
        self.right_distance = self.safe_min(right)

        all_valid = front + left + right
        self.lidar_valid = bool(all_valid)

        if not self.lidar_valid:
            return

        self.obstacle_distance = self.front_distance

        # --------------------------------------------------------------
        # Destination approach
        # --------------------------------------------------------------
        if self.state == STATE_APPROACH:
            if self.front_distance <= DESTINATION_HARD_STOP_DISTANCE:
                self.finish_destination()
                return

            return

        # --------------------------------------------------------------
        # Normal lane following: detect front obstacle.
        # --------------------------------------------------------------
        if self.state == STATE_FOLLOW:
            if self.front_distance < OBSTACLE_DETECT_DISTANCE:
                self.start_obstacle_avoidance()

    def start_obstacle_avoidance(self) -> None:
        if self.state != STATE_FOLLOW:
            return

        # Select the side with more clearance.
        if self.left_distance > self.right_distance:
            self.obstacle_side = "left"
            self.avoid_turn = OBSTACLE_MAX_TURN
        else:
            self.obstacle_side = "right"
            self.avoid_turn = -OBSTACLE_MAX_TURN

        self.avoid_start_time = self.now_sec()
        self.state = STATE_AVOID

        self.get_logger().info(
            f"Obstacle detected. Avoiding to {self.obstacle_side}."
        )

    def control_obstacle_avoidance(self) -> None:
        """
        Simple reactive obstacle-avoidance state.

        Behaviour:
          1. Turn toward the side with greater clearance.
          2. Continue until the front becomes clear.
          3. Hold a reduced steering command while clearing.
          4. Return to line following.

        This is intentionally conservative rather than trying to estimate
        a full obstacle geometry from one scan.
        """
        elapsed = self.now_sec() - self.avoid_start_time

        # Safety timeout: do not stay in avoidance forever.
        if elapsed > OBSTACLE_MAX_TIME:
            self.get_logger().warn(
                "Obstacle avoidance timeout. Stopping for safety."
            )
            self.stop()
            self.state = STATE_STOPPED
            return

        # Obstacle still directly ahead.
        if self.front_distance < OBSTACLE_DETECT_DISTANCE:
            # Increase turn if the selected side is becoming blocked.
            if self.obstacle_side == "left":
                if self.left_distance < OBSTACLE_SIDE_CLEAR_DISTANCE:
                    self.avoid_turn = -OBSTACLE_MAX_TURN
                    self.obstacle_side = "right"
                else:
                    self.avoid_turn = OBSTACLE_MAX_TURN
            else:
                if self.right_distance < OBSTACLE_SIDE_CLEAR_DISTANCE:
                    self.avoid_turn = OBSTACLE_MAX_TURN
                    self.obstacle_side = "left"
                else:
                    self.avoid_turn = -OBSTACLE_MAX_TURN

            self.set_command(
                OBSTACLE_TURN_SPEED,
                self.avoid_turn,
            )
            return

        # Front is clear enough. Keep turning briefly while passing the
        # obstacle, then hand control back to the line follower.
        if self.front_distance < OBSTACLE_CLEAR_DISTANCE:
            self.set_command(
                OBSTACLE_TURN_SPEED,
                0.60 * self.avoid_turn,
            )
            return

        # Clear.
        self.state = STATE_FOLLOW
        self.get_logger().info("Obstacle cleared. Returning to line following.")

    # ======================================================================
    # Destination approach
    # ======================================================================

    def control_destination_approach(self) -> None:
        if not self.lidar_valid:
            # Do not blindly drive toward a building without LiDAR data.
            self.set_command(0.0, self.line_turn)
            return

        if self.front_distance <= DESTINATION_STOP_DISTANCE:
            self.finish_destination()
            return

        # Slow down continuously as the target gets closer.
        distance = self.front_distance
        if distance >= DESTINATION_SLOW_DISTANCE:
            speed = APPROACH_SPEED
        else:
            ratio = (
                distance - DESTINATION_STOP_DISTANCE
            ) / (
                DESTINATION_SLOW_DISTANCE - DESTINATION_STOP_DISTANCE
            )
            ratio = self.clamp(ratio, 0.0, 1.0)

            speed = 0.04 + ratio * (
                APPROACH_SPEED - 0.04
            )

        # Continue using the latest line heading while approaching.
        self.set_command(
            speed,
            self.clamp(self.line_turn, -0.35, 0.35),
        )

    def finish_destination(self) -> None:
        if self.state == STATE_STOPPED:
            return

        self.stop()
        self.state = STATE_STOPPED
        self.reached = True

        self.get_logger().info(
            f"Reached destination: {self.qr_data or self.destination}"
        )

        self.report_arrival_once()

    # ======================================================================
    # Server communication
    # ======================================================================

    def server_communication_callback(
        self,
        message: ServerCommunication,
    ) -> None:
        """
        Receive server commands and ACKs.

        Project convention retained:
            Buggy = src 1
            Server = dest 2
            Messages intended for buggy have dest == 1.
        """
        if message.dest != 1:
            return

        self.get_logger().info(
            f"Received Server Message: msg={message.msg!r}, "
            f"uid={message.uid}, ack={message.ack}"
        )

        # ACK from server.
        if message.ack == 1:
            self.last_received_uid = message.uid
            self.ack_pending = False

            # Once the server acknowledges an arrival, prepare for the next
            # mission leg.
            if self.reached and not self.mission_completed:
                self.prepare_next_leg()

        # Mission destination command.
        if message.msg:
            destination = message.msg.strip().strip("{}").strip()

            # The routing code expects A/B/C/X/Y/Z.
            if destination in ("A", "B", "C", "X", "Y", "Z"):
                self.destination = destination
                self.get_logger().info(
                    f"New destination received: {self.destination}"
                )

                # A new destination means a new navigation leg.
                self.reached = False
                self.arrival_reported = False
                self.qr_data = ""

                if destination in ("A", "B", "C"):
                    self.patient_id = None
                else:
                    self.hospital_id = None

                if self.state == STATE_STOPPED:
                    self.state = STATE_FOLLOW

    def prepare_next_leg(self) -> None:
        """
        Clear arrival state after a server ACK.

        If the final hospital is Z, the mission is considered complete.
        """
        if self.hospital_id == 3 and self.destination == "Z":
            self.mission_completed = True
            self.state = STATE_COMPLETE
            self.stop()

            self.get_logger().info(
                "Final hospital reached. Mission complete."
            )
            return

        self.reached = False
        self.arrival_reported = False
        self.qr_data = ""

        self.state = STATE_FOLLOW

        # Do not retain the previous target's identifiers.
        self.patient_id = None
        self.hospital_id = None

    def report_arrival_once(self) -> None:
        if self.arrival_reported:
            return

        if not self.qr_data:
            self.get_logger().warn(
                "Reached destination but no QR payload is available; "
                "arrival report not sent."
            )
            return

        self.send_server_update(self.qr_data)
        self.arrival_reported = True

    def send_server_update(self, text_msg: str) -> None:
        """
        Send one status/arrival message to the server.

        UID is now a rolling counter instead of the fixed 100 from the
        original implementation.
        """
        if not text_msg:
            return

        server_msg = ServerCommunication()
        server_msg.src = 1
        server_msg.dest = 2
        server_msg.uid = self.next_uid
        server_msg.ack = 1
        server_msg.msg = text_msg

        self.next_uid += 1
        self.publisher_server.publish(server_msg)

        self.ack_pending = True

        self.get_logger().info(
            f"Sent server update: uid={server_msg.uid}, msg={text_msg!r}"
        )

    # ======================================================================
    # QR detection
    # ======================================================================

    def qr_detection_callback(self, message: String) -> None:
        payload = message.data.strip()

        if not payload:
            return

        self.get_logger().info(f"Heard QR code: {payload}")

        patient_map = {"A": 1, "B": 2, "C": 3}
        hospital_map = {"X": 1, "Y": 2, "Z": 3}

        # --------------------------------------------------------------
        # Patient QR
        # Expected format from the current code:
        #     {LOC_PATIENT_1}
        # --------------------------------------------------------------
        if payload.startswith("{LOC_PATIENT_"):
            patient_id = self.parse_location_id(payload, "LOC_PATIENT_")

            if patient_id is None:
                return

            self.patient_id = patient_id

            if (
                self.destination in patient_map
                and patient_map[self.destination] == patient_id
            ):
                self.get_logger().info(
                    f"Target patient detected: {patient_id}"
                )

                self.qr_data = payload
                self.reached = False
                self.arrival_reported = False
                self.state = STATE_APPROACH
                self.obstacle_distance = float("inf")
            else:
                self.get_logger().info(
                    f"Non-target patient QR ignored: {patient_id}"
                )

            return

        # --------------------------------------------------------------
        # Hospital QR
        # Expected format:
        #     {LOC_HOSPITAL_1}
        # --------------------------------------------------------------
        if payload.startswith("{LOC_HOSPITAL_"):
            hospital_id = self.parse_location_id(
                payload,
                "LOC_HOSPITAL_",
            )

            if hospital_id is None:
                return

            self.hospital_id = hospital_id

            if (
                self.destination in hospital_map
                and hospital_map[self.destination] == hospital_id
            ):
                self.get_logger().info(
                    f"Target hospital detected: {hospital_id}"
                )

                self.qr_data = payload
                self.reached = False
                self.arrival_reported = False
                self.state = STATE_APPROACH
                self.obstacle_distance = float("inf")
            else:
                self.get_logger().info(
                    f"Non-target hospital QR ignored: {hospital_id}"
                )

    def parse_location_id(
        self,
        payload: str,
        prefix: str,
    ) -> Optional[int]:
        """
        Safely parse:
            {LOC_PATIENT_1}
            {LOC_HOSPITAL_2}
        """
        try:
            cleaned = payload.strip("{}")
            if not cleaned.startswith(prefix):
                return None

            number_text = cleaned[len(prefix):]
            value = int(number_text)

            if value not in (1, 2, 3):
                self.get_logger().warn(
                    f"Invalid location ID in QR: {payload}"
                )
                return None

            return value

        except (ValueError, TypeError):
            self.get_logger().warn(
                f"Could not parse QR location: {payload}"
            )
            return None

    # ======================================================================
    # Sign-board routing
    # ======================================================================

    def sign_board_callback(self, message: String) -> None:
        """
        Parse sign-board detections.

        Expected format:
            A:0.96;Left:0.95;Right:1.40;Straight:2.10

        The destination entry is compared against Left/Right/Straight
        distances. The closest matching directional entry is selected.
        """
        payload = message.data.strip()

        if not payload:
            return

        self.get_logger().info(
            f"Heard Sign Board: {payload}"
        )

        entries = []

        for raw_entry in payload.split(";"):
            raw_entry = raw_entry.strip()

            if not raw_entry:
                continue

            parts = raw_entry.split(":", 1)

            if len(parts) != 2:
                self.get_logger().warn(
                    f"Ignoring malformed sign entry: {raw_entry}"
                )
                continue

            label = parts[0].strip()

            try:
                distance = float(parts[1])
            except ValueError:
                self.get_logger().warn(
                    f"Ignoring sign with invalid distance: {raw_entry}"
                )
                continue

            if not math.isfinite(distance):
                continue

            entries.append((label, distance))

        if not entries:
            return

        if self.destination is None:
            return

        destination_entry = next(
            (
                entry for entry in entries
                if entry[0] == self.destination
            ),
            None,
        )

        if destination_entry is None:
            self.get_logger().info(
                f"Destination {self.destination!r} not present in sign board."
            )
            return

        candidates = [
            entry for entry in entries
            if entry[0] in ("Left", "Right", "Straight")
        ]

        if not candidates:
            return

        nearest = min(
            candidates,
            key=lambda entry: abs(
                entry[1] - destination_entry[1]
            ),
        )

        error = abs(
            nearest[1] - destination_entry[1]
        )

        # Keep the original 0.1 distance matching rule.
        if error >= 0.10:
            return

        self.direction = nearest[0]

        if self.direction == "Left":
            self.override_till = (
                self.now_sec() + TURN_OVERRIDE_DURATION
            )

        elif self.direction == "Right":
            self.override_till = (
                self.now_sec() + TURN_OVERRIDE_DURATION
            )

        else:
            # Straight means no steering override.
            self.override_till = 0.0

        self.get_logger().info(
            f"Sign routing: destination={self.destination}, "
            f"direction={self.direction}, "
            f"distance_error={error:.3f}"
        )


def main(args=None):
    rclpy.init(args=args)

    node = LineFollower()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Controller interrupted by user.")
    finally:
        node.stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
