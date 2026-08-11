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
import time
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

# CONFIGURATION:
# The buggy is driven in manual mode by publishing standard controller Joy messages to /cerebri/in/joy.
# The layout is: msg.axes = [0.0, speed, 0.0, turn]
# - speed: positive for forward, negative for reverse. Range: [-1.0, 1.0]
# - turn: positive for left steer, negative for right steer. Range: [-1.0, 1.0]
# msg.buttons = [1, 0, 0, 0, 0, 0, 0, 1] (Keep buttons set to this pattern for manual override mode)

class LineFollower(Node):
    """
    Core controller Node for the B3RB buggy.
    By default, it publishes a safe drive-straight command on a timer loop.
    Implement logic inside the callbacks to steer, dodge obstacles, detect destinations,
    communicate with the server, and park.
    """
    def __init__(self):
        super().__init__('line_follower')

        # ------------------ Subscriptions ------------------

        # 1. Lane Edge Vectors (from edge_vectors_publisher)
        self.subscription_vectors = self.create_subscription(
            EdgeVectors,
            '/edge_vectors',
            self.edge_vectors_callback,
            QOS_PROFILE_DEFAULT)

        # 2. LIDAR Obstacle Scanner
        self.subscription_lidar = self.create_subscription(
            LaserScan,
            '/scan',
            self.lidar_callback,
            QOS_PROFILE_DEFAULT)

        # 3. Server Communication Feedback Loop
        self.subscription_server = self.create_subscription(
            ServerCommunication,
            '/ServerCommunication',
            self.server_communication_callback,
            QOS_PROFILE_DEFAULT)

        # 4. QR Code Detections (from qr_detector)
        self.subscription_qr = self.create_subscription(
            String,
            '/qr_detection',
            self.qr_detection_callback,
            QOS_PROFILE_DEFAULT)

        # 5. Sign Board Detections (from object_recognizer)
        self.subscription_signs = self.create_subscription(
            String,
            '/sign_board_detection',
            self.sign_board_callback,
            QOS_PROFILE_DEFAULT)

        # ------------------ Publishers ------------------

        # Publisher to drive/steer the buggy
        self.publisher_joy = self.create_publisher(
            Joy,
            '/cerebri/in/joy',
            QOS_PROFILE_DEFAULT)

        # Publisher to send messages to the Server
        self.publisher_server = self.create_publisher(
            ServerCommunication,
            '/ServerCommunication',
            QOS_PROFILE_DEFAULT)

        # ------------------ State Variables & Timer ------------------

        # Default controls: drive straight slowly
        self.target_speed = 0.15
        self.target_turn = 0.0

        # State variables
        self.avoid = False
        self.patient_id = None
        self.hospital_id = None
        self.destination = None
        self.mission_completed = False
        self.approaching = False
        self.qr_data = ""
        self.expconst = 0.4
        self.expconst2 = 0.2
        self.reached = False
        self.direction = ""
        self.lost_count = 0
        self.ackno = 0
        self.override_till = 0
        self.override_dur = 1.5

        # Rolling message id counter for Buggy -> Server messages
        self.uid_counter = 0

        # Distance band (meters) that counts as "right outside the building".
        # Must be close enough to be at the door, but not so close we ram it.
        self.stop_zone_far = 1.0
        self.stop_zone_near = 0.4

        # True once stopped in the zone outside a building and reported to the
        # server. While True, line-following and obstacle-avoidance are both
        # suppressed so nothing overrides the stop on the next control tick.
        self.parked = False

        # Timer to publish drive commands at 10Hz
        self.control_timer = self.create_timer(0.1, self.publish_drive_commands)

        self.get_logger().info("Line Follower controller initialized. Safe Drive-Straight Mode active.")

    def publish_drive_commands(self):
        """Timer callback that periodically publishes the current speed and steer command."""
        msg = Joy()
        msg.buttons = [1, 0, 0, 0, 0, 0, 0, 1]  # Manual override button configuration
        msg.axes = [0.0, self.target_speed, 0.0, self.target_turn]
        self.publisher_joy.publish(msg)

    def rover_move_manual_mode(self, speed, turn):
        """Helper to immediately set control speed and steering angle."""
        self.target_speed = float(max(min(speed, SPEED_MAX), -SPEED_MAX))
        self.target_turn = float(max(min(turn, TURN_MAX), -TURN_MAX))

    # ------------------ Callback Implementations ------------------

    def edge_vectors_callback(self, message):
        if self.parked:
            return
        if self.avoid:
            return
        if self.get_clock().now().nanoseconds * (10 ** (-9)) < self.override_till:
            ang = 0.7 if self.direction == "Left" else -0.7
            self.target_turn = self.expconst * ang + (1 - self.expconst) * self.target_turn
            self.target_speed = max(self.target_speed * 0.7, 0.15)
            return
        if message.vector_count == 0:
            self.lost_count = self.lost_count + 1
            decay = min(self.lost_count / 10, 1.0)
            angle = self.expconst * decay + (1 - self.expconst) * self.target_turn
            self.rover_move_manual_mode(self.target_speed, angle)
        elif message.vector_count == 1:
            farpoint = message.vector_1[0] if message.vector_1 else message.vector_2[0]
            dx = farpoint.x - message.image_width / 2

            dy = message.image_height - farpoint.y
            if dy == 0:
                return
            ang = math.atan(dx / dy) / (PI / 2) if abs(math.atan(dx / dy) / (PI / 2)) > 0.2 else 0
            angle = self.expconst * ang + (1 - self.expconst) * self.target_turn
            if not self.approaching:
                speed = (1 - abs(ang) * 0.8)

                spd = speed * self.expconst + (1 - self.expconst) * self.target_speed
            else:
                spd = self.target_speed
            self.lost_count = 0
            self.rover_move_manual_mode(spd, angle)

        elif message.vector_count == 2:
            farx = (message.vector_1[0].x + message.vector_2[0].x) / 2
            fary = (message.vector_1[0].y + message.vector_2[0].y) / 2
            dx = farx - message.image_width / 2
            dy = message.image_height - fary
            if dy == 0:
                return
            ang = math.atan(dx / dy) / (PI / 2)
            angle = -self.expconst2 * ang + (1 - self.expconst2) * self.target_turn
            if not self.approaching:
                speed = (1 - abs(ang) * 0.8)
                spd = speed * self.expconst2 + (1 - self.expconst2) * self.target_speed
            else:
                spd = self.target_speed
            self.lost_count = 0
            self.rover_move_manual_mode(spd, angle)

    def lidar_callback(self, message):
        """
        Receives LIDAR range measurements.

        - `message.ranges` is an array of distances in meters around the buggy.
        - Front-left/front-right sectors are used both for general obstacle avoidance
          (when not approaching a target) and for controlled deceleration into the
          target building (when self.approaching is True).
        - While approaching, distance to the building is taken as the closest of
          the front, front-left, and front-right sectors — the building may sit
          beside the track rather than dead ahead, so a front-only check can miss
          it entirely. Once that distance falls inside self.stop_zone_far, the
          buggy hard-stops, latches self.parked = True (so nothing overrides the
          stop on the next control tick), and reports arrival to the server.
        """
        if self.parked:
            return

        n = len(message.ranges)
        right_sector = message.ranges[int(n * 7 / 18): int(n * 9 / 18)]
        left_sector = message.ranges[int(n * 9 / 18): int(n * 11 / 18)]
        sector = right_sector if min(right_sector) < min(left_sector) else left_sector

        if not self.approaching:
            if min(sector) < 0.8:
                spd = min(self.target_speed, min(sector) * self.expconst / 0.8 + (1 - self.expconst) * self.target_speed)
                self.avoid = True
                if sector == right_sector:
                    lowerbound = n * 7 / 18
                else:
                    lowerbound = n * 9 / 18
                offset = (9 * n / 18 - (sector.index(min(sector)) + lowerbound))
                if offset != 0:
                    ang = (1 - abs(offset) / (2 * n / 18)) * (abs(offset) / offset)
                else:
                    ang = 0.1 if lowerbound == n * 7 / 18 else -0.1
                angle = self.expconst * ang + (1 - self.expconst) * self.target_turn

                self.rover_move_manual_mode(spd, angle)
            else:
                self.avoid = False
        else:
            # Narrow directly-ahead slice, used alongside the lateral sectors
            # below since the building may not be exactly dead-ahead.
            front_sector = message.ranges[int(n * 17 / 36):int(n * 19 / 36)]
            front_sector = [r for r in front_sector if r > 0.0 and not math.isinf(r)]
            front_distance = min(front_sector) if front_sector else float('inf')

            left_part = message.ranges[int(n * 10 / 18):int(n * 14 / 18)]
            right_part = message.ranges[int(n * 4 / 18):int(n * 8 / 18)]
            left_part_f = [r for r in left_part if r > 0.0 and not math.isinf(r)]
            right_part_f = [r for r in right_part if r > 0.0 and not math.isinf(r)]
            min_left = min(left_part_f) if left_part_f else float('inf')
            min_right = min(right_part_f) if right_part_f else float('inf')

            # The building can be dead ahead OR beside the track depending on
            # track layout, so take whichever sector is actually closest.
            building_distance = min(front_distance, min_left, min_right)

            # Once inside the stop zone, hard-stop and latch parked so nothing
            # (line-following, avoidance) can override the stop afterwards.
            if building_distance <= self.stop_zone_far:
                self.rover_move_manual_mode(0.0, self.target_turn)
                self.target_speed = 0.0
                self.target_turn = 0.0
                self.approaching = False
                self.reached = True
                self.parked = True
                self.get_logger().info(
                    f"In stop zone ({building_distance:.2f}m) outside building — reporting arrival.")

                if self.patient_id in [1, 2, 3] or self.hospital_id in [1, 2, 3]:
                    self.send_server_update(self.qr_data)

                if self.hospital_id == 3:
                    self.mission_completed = 1
                return

            # Still approaching but not yet in the stop zone: keep centering
            # on the building while slowing proportionally to lateral offset.
            if min(min_left, min_right) < 3:
                if min_left < min_right:
                    idx = int(n * 10 / 18) + left_part.index(min_left)
                    offset = (n / 2 - idx) / (4 * n / 18)
                    spd = self.target_speed * (1 - abs(offset))
                    self.rover_move_manual_mode(spd, self.target_turn)
                else:
                    idx = int(n * 4 / 18) + right_part.index(min_right)
                    offset = (n / 2 - idx) / (4 * n / 18)
                    spd = self.target_speed * (1 - abs(offset))
                    self.rover_move_manual_mode(spd, self.target_turn)

    def server_communication_callback(self, message):
        """
        Receives coordination commands from the server.

        - Check if the message is destined for the Buggy (`message.dest == 1`).
        - Check for ACK messages from server and clear pending state accordingly.
        - The server communicates mission info in the `message.msg` payload string.
        """
        if message.dest == 1:
            self.get_logger().info(f"Received Server Message: {message.msg}")
            self.destination = message.msg
            self.ackno = 1

            if message.ack == 1:
                self.qr_data = ""
                self.patient_id = 0
                self.hospital_id = 0
                # Server has acknowledged the pickup/drop at this stop — release
                # the park latch and resume line-following toward the next leg.
                self.parked = False
                self.reached = False
                self.target_speed = 0.15
                self.target_turn = 0.0

        # NOTE: the primary arrival report now fires inside lidar_callback the
        # moment the buggy actually stops outside the building, so it's not
        # dependent on the timing of an incoming server message. This block is
        # kept as a safety-net resend in case `reached` is still True when the
        # next server message arrives (e.g. report was lost).
        if self.reached:
            if self.patient_id in [1, 2, 3]:
                self.send_server_update(self.qr_data)
            if self.hospital_id in [1, 2, 3]:
                self.send_server_update(self.qr_data)
            if self.hospital_id == 3:
                self.mission_completed = 1

    def send_server_update(self, text_msg):
        """Sends status messages to the server, with a rolling uid and ACK flag."""
        server_msg = ServerCommunication()
        server_msg.src = 1       # Buggy
        server_msg.dest = 2      # Server
        server_msg.uid = self.uid_counter
        server_msg.ack = self.ackno
        server_msg.msg = text_msg
        self.publisher_server.publish(server_msg)
        self.get_logger().info(f"Sent update to server (uid={server_msg.uid}): {text_msg}")
        self.uid_counter = (self.uid_counter + 1) % 256
        self.ackno = 0

    def qr_detection_callback(self, message):
        """
        Receives QR codes scanned from the buildings.

        Expected payload format: "{LOC: HOSPITAL_2}", "{LOC: PATIENT_1}",
        "{LOC: FAKE_HOSPITAL_3}". FAKE_* markers must be ignored.

        If the scanned code matches the current target destination, sets
        self.approaching = True so lidar_callback takes over the controlled
        stop-and-report sequence.
        """
        raw = message.data.strip()
        self.get_logger().info(f"Heard QR code: {raw}")

        content = raw.strip("{}")            # e.g. "LOC: HOSPITAL_2"
        if ":" not in content:
            self.get_logger().warn(f"Malformed QR payload, ignoring: {raw}")
            return

        _, payload = content.split(":", 1)
        payload = payload.strip()            # e.g. "HOSPITAL_2" or "FAKE_HOSPITAL_2"

        # Must be checked BEFORE the HOSPITAL/PATIENT match, since
        # "FAKE_HOSPITAL_2" contains "HOSPITAL_2" as a substring.
        if payload.startswith("FAKE_"):
            self.get_logger().info(f"Ignoring fake marker: {payload}")
            return

        try:
            loc_type, loc_num_str = payload.rsplit("_", 1)
            loc_num = int(loc_num_str)
        except ValueError:
            self.get_logger().warn(f"Could not parse location id from: {payload}")
            return

        map1 = {"X": 1, "Y": 2, "Z": 3}   # hospitals
        map2 = {"A": 1, "B": 2, "C": 3}   # patients

        if loc_type == "PATIENT":
            self.patient_id = loc_num
            self.get_logger().info(f"Identified Patient: {self.patient_id}")
            if self.destination in map2 and self.patient_id == map2[self.destination]:
                self.get_logger().info(f"Approaching target patient location: {self.patient_id}")
                self.approaching = True
                self.avoid = False
                self.qr_data = raw
            else:
                self.approaching = False
                self.patient_id = 0

        elif loc_type == "HOSPITAL":
            self.hospital_id = loc_num
            self.get_logger().info(f"Identified Hospital: {self.hospital_id}")
            if self.destination in map1 and self.hospital_id == map1[self.destination]:
                self.get_logger().info(f"Approaching target hospital location: {self.hospital_id}")
                self.approaching = True
                self.avoid = False
                self.qr_data = raw
            else:
                self.approaching = False
                self.reached = False
                self.hospital_id = 0

        else:
            self.get_logger().warn(f"Unknown location type in payload: {payload}")

    def sign_board_callback(self, message):
        """
        Receives traffic sign boards.

        GUIDELINE (Sign Board Routing):
        - Use the detected signs to choose the quickest route at intersections.
        """
        self.get_logger().info(f"Heard Sign Board: {message.data}")
        entries = []
        for e in message.data.split(";"):
            parts = e.split(":")
            entries.append([parts[0], float(parts[1])])
        destentry = None
        for e in entries:
            if e[0] == self.destination:
                destentry = e

        candidates = [e for e in entries if e[0] in ("Left", "Right", "Straight")]
        if not candidates:
            return
        nearest = min(candidates, key=lambda e: abs(e[1] - destentry[1]))
        if abs(nearest[1] - destentry[1]) < 0.1:
            self.direction = nearest[0]
            if self.direction in ['Left', 'Right']:
                self.override_till = self.get_clock().now().nanoseconds * (10 ** (-9)) + self.override_dur


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
