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

# ---------------------------------------------------------------------------
# Sign-board lane-stick controller
# ---------------------------------------------------------------------------
LANE_STICK_DISTANCE_PX = 120.0
LANE_STICK_KP = 0.0030
LANE_STICK_KANGLE = 0.10
LANE_STICK_MAX_TURN = 0.30
LANE_STICK_SPEED = 0.13

# A sign only arms a turn. The edge geometry must show a real bend before
# lane-stick starts, so a sign visible at spawn cannot immediately steer us.
SIGN_MAX_AGE = 8.0
TURN_START_ANGLE = 0.22
TURN_COMPLETE_ANGLE = 0.08
TURN_COMPLETE_FRAMES = 6
TURN_MAX_TIME = 2.5

# Smooth hand-back to the normal centre-following controller.
CENTER_RETURN_BLEND = 0.12
CENTER_RETURN_FRAMES = 12

# ---------------------------------------------------------------------------
# DEBUG
# ---------------------------------------------------------------------------
# Flip to False to silence the state-machine trace once you've diagnosed
# the issue. Every DEBUG_LOG line below is new -- search "DEBUG_LOG" to find
# every logging statement added for this fix.
DEBUG_LOG = True
DEBUG_LOG_EVERY_N_EDGE_CALLBACKS = 5   # throttle the high-rate edge trace

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

        # State variables (You can add your own state flags / state machines here)
        self.avoid = False
        self.patient_id = None
        self.hospital_id = None
        self.destination = None
        self.mission_completed = False
        self.approaching=False
        self.qr_data=""
        self.expconst=0.8 #earlier 0.4
        self.expconst2=0.8 #earlier it was 0.2
        self.reached=False
        self.direction=""
        self.lost_count=0
        self.ackno=0
        self.override_till=0
        self.override_dur=1.5

        # ------------------------------------------------------------------
        # New routing state.
        # The old line follower remains the default controller.
        # ------------------------------------------------------------------
        self.pending_turn = None          # "Left" / "Right"
        self.pending_turn_time = 0.0
        self.sign_candidate = None
        self.sign_candidate_count = 0

        self.lane_mode = "CENTER"         # CENTER / STICK / TURNING / RETURN_CENTER
        self.stick_side = None             # "Left" / "Right"
        self.turn_start_time = 0.0
        self.turn_complete_count = 0
        self.return_count = 0

        # DEBUG_LOG: bookkeeping for throttled trace + change-only logging.
        self._dbg_edge_call_count = 0
        self._dbg_last_lane_mode = self.lane_mode
        self._dbg_last_avoid = self.avoid
        self._dbg_last_stick_side = self.stick_side



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
        """
        Existing line follower + sign-board lane sticking.

        IMPORTANT:
        A sign detection NEVER directly commands a turn.
        It only creates pending_turn.  The buggy keeps using the old working
        line follower until the selected edge actually bends at the
        intersection.
        """
        # --------------------------------------------------------------
        # FIX #1 (root cause of "detected Right but still went Left"):
        # Previously `if self.avoid: return` bypassed the ENTIRE turn state
        # machine (STICK/TURNING/RETURN_CENTER) any time lidar_callback set
        # self.avoid = True. Left and right junctions expose different
        # corner/wall geometry to the LIDAR, so it was easy for a right
        # turn -- but not a left turn -- to trip the 0.8 m obstacle
        # threshold right as lane-stick should have taken over. When that
        # happened, this function returned immediately, lidar_callback's
        # generic avoidance steering took over instead, and the sign
        # decision was silently discarded for that frame (and often the
        # whole turn, since avoidance can keep re-triggering).
        #
        # Fix: once a turn is actively armed/executing (STICK or TURNING),
        # don't let raw obstacle-avoidance override the turn controller.
        # Lane-stick already keeps a safe, controlled distance from the
        # curb/wall it is following, so suppressing generic avoidance here
        # is safe. Plain CENTER/RETURN_CENTER driving still honors avoid.
        # --------------------------------------------------------------
        if self.avoid and self.lane_mode not in ("STICK", "TURNING"):
            if DEBUG_LOG:
                self.get_logger().info(
                    "[DEBUG_LOG] edge_vectors_callback: bypassed by LIDAR "
                    f"avoidance (avoid=True, lane_mode={self.lane_mode})"
                )
            return

        now = self.get_clock().now().nanoseconds * 1e-9

        # Expire stale sign instructions.
        if (
            self.pending_turn is not None
            and now - self.pending_turn_time > SIGN_MAX_AGE
        ):
            self.get_logger().info(
                f"Ignoring stale pending turn: {self.pending_turn}"
            )
            if DEBUG_LOG:
                self.get_logger().info(
                    "[DEBUG_LOG] pending_turn EXPIRED "
                    f"(age={now - self.pending_turn_time:.2f}s > {SIGN_MAX_AGE}s). "
                    "If this fires right before a missed turn, the sign was "
                    "armed but geometry never confirmed it (see "
                    "turn_geometry_detected logs)."
                )
            self.pending_turn = None
            self.sign_candidate = None
            self.sign_candidate_count = 0

        # DEBUG_LOG: throttled state trace so you can correlate lane_mode /
        # avoid / pending_turn / stick_side frame-by-frame against the bag.
        if DEBUG_LOG:
            self._dbg_edge_call_count += 1
            state_changed = (
                self.lane_mode != self._dbg_last_lane_mode
                or self.avoid != self._dbg_last_avoid
                or self.stick_side != self._dbg_last_stick_side
            )
            if state_changed or (
                self._dbg_edge_call_count % DEBUG_LOG_EVERY_N_EDGE_CALLBACKS == 0
            ):
                self.get_logger().info(
                    "[DEBUG_LOG] state: "
                    f"lane_mode={self.lane_mode} "
                    f"pending_turn={self.pending_turn} "
                    f"stick_side={self.stick_side} "
                    f"avoid={self.avoid} "
                    f"vector_count={message.vector_count} "
                    f"target_turn={self.target_turn:.3f} "
                    f"target_speed={self.target_speed:.3f}"
                )
            self._dbg_last_lane_mode = self.lane_mode
            self._dbg_last_avoid = self.avoid
            self._dbg_last_stick_side = self.stick_side

        # --------------------------------------------------------------
        # Active lane-stick controller
        # --------------------------------------------------------------
        if self.lane_mode in ("STICK", "TURNING"):
            self.update_lane_stick(message)
            return

        # --------------------------------------------------------------
        # Smooth return to normal centre following
        # --------------------------------------------------------------
        if self.lane_mode == "RETURN_CENTER":
            normal_turn, normal_speed = self.calculate_normal_line_follow(message)

            # Blend gradually. This prevents a hard steering jump after a turn.
            self.target_turn = (
                (1.0 - CENTER_RETURN_BLEND) * self.target_turn
                + CENTER_RETURN_BLEND * normal_turn
            )
            self.target_speed = (
                (1.0 - CENTER_RETURN_BLEND) * self.target_speed
                + CENTER_RETURN_BLEND * normal_speed
            )

            self.return_count += 1
            if self.return_count >= CENTER_RETURN_FRAMES:
                self.lane_mode = "CENTER"
                self.stick_side = None
                self.pending_turn = None
                self.return_count = 0
                self.get_logger().info(
                    "Centre-return complete. Normal line following resumed."
                )
            return

        # --------------------------------------------------------------
        # CENTER mode: decide whether a pending sign is actually becoming a
        # turn. Otherwise execute the ORIGINAL working line follower.
        # --------------------------------------------------------------
        if self.pending_turn in ("Left", "Right"):
            geometry_ok = self.turn_geometry_detected(message)
            if DEBUG_LOG:
                distance, angle = self.selected_edge_geometry(message)
                self.get_logger().info(
                    f"[DEBUG_LOG] CENTER mode, pending_turn={self.pending_turn}: "
                    f"turn_geometry_detected={geometry_ok} "
                    f"(selected_edge distance={distance}, angle={angle:.3f}, "
                    f"threshold={TURN_START_ANGLE})"
                )
            if geometry_ok:
                self.start_lane_stick(self.pending_turn)
                self.update_lane_stick(message)
                return

        normal_turn, normal_speed = self.calculate_normal_line_follow(message)
        self.rover_move_manual_mode(normal_speed, normal_turn)

    def calculate_normal_line_follow(self, message):
        """The old working line-following controller, isolated unchanged."""
        if message.vector_count == 0:
            self.lost_count += 1
            decay = min(self.lost_count / 10.0, 1.0)
            angle = (
                self.expconst * decay
                + (1.0 - self.expconst) * self.target_turn
            )
            return self.target_turn if self.approaching else angle, self.target_speed

        if message.vector_count == 1:
            points = message.vector_1 if message.vector_1 else message.vector_2
            if not points:
                return self.target_turn, self.target_speed

            farpoint = points[0]
            dx = farpoint.x - message.image_width / 2.0
            dy = message.image_height - farpoint.y
            if abs(dy) < 1e-6:
                return self.target_turn, self.target_speed

            raw = math.atan(dx / dy) / (PI / 2.0)
            ang = raw if abs(raw) > 0.2 else 0.0
            angle = (
                self.expconst * ang
                + (1.0 - self.expconst) * self.target_turn
            )

            if not self.approaching:
                speed = 1.0 - abs(ang) * 0.8
                spd = (
                    speed * self.expconst
                    + (1.0 - self.expconst) * self.target_speed
                )
            else:
                spd = self.target_speed

            self.lost_count = 0
            return angle, spd

        if message.vector_count == 2:
            if not message.vector_1 or not message.vector_2:
                return self.target_turn, self.target_speed

            farx = (message.vector_1[0].x + message.vector_2[0].x) / 2.0
            fary = (message.vector_1[0].y + message.vector_2[0].y) / 2.0
            dx = farx - message.image_width / 2.0
            dy = message.image_height - fary
            if abs(dy) < 1e-6:
                return self.target_turn, self.target_speed

            ang = math.atan(dx / dy) / (PI / 2.0)
            angle = (
                -self.expconst2 * ang
                + (1.0 - self.expconst2) * self.target_turn
            )

            if not self.approaching:
                speed = 1.0 - abs(ang) * 0.8
                spd = (
                    speed * self.expconst2
                    + (1.0 - self.expconst2) * self.target_speed
                )
            else:
                spd = self.target_speed

            self.lost_count = 0
            return angle, spd

        return self.target_turn, self.target_speed

    @staticmethod
    def edge_geometry(points, image_width, image_height):
        """Return (distance_from_center, signed_angle) for one edge."""
        if not points:
            return None, 0.0

        # Closest/bottom point is used for the actual lane offset.
        near = max(points, key=lambda p: p.y)
        far = min(points, key=lambda p: p.y)

        center = image_width / 2.0
        distance = abs(near.x - center)

        dy = near.y - far.y
        if abs(dy) < 1e-6:
            return distance, 0.0

        # Positive = edge trends toward the right as it goes forward.
        signed_angle = math.atan2(
            far.x - near.x,
            dy,
        ) / (PI / 2.0)
        signed_angle = max(-1.0, min(1.0, signed_angle))

        return distance, signed_angle

    def selected_edge_geometry(self, message):
        """Get geometry for the requested left/right edge."""
        if message.vector_count < 1:
            return None, 0.0

        if self.pending_turn == "Left" or self.stick_side == "Left":
            if message.vector_1 and message.vector_2:
                p1 = max(message.vector_1, key=lambda p: p.y)
                p2 = max(message.vector_2, key=lambda p: p.y)
                points = message.vector_1 if p1.x < p2.x else message.vector_2
            else:
                points = message.vector_1 or message.vector_2
        else:
            if message.vector_1 and message.vector_2:
                p1 = max(message.vector_1, key=lambda p: p.y)
                p2 = max(message.vector_2, key=lambda p: p.y)
                points = message.vector_1 if p1.x > p2.x else message.vector_2
            else:
                points = message.vector_1 or message.vector_2

        return self.edge_geometry(
            points,
            message.image_width,
            message.image_height,
        )

    def turn_geometry_detected(self, message):
        """
        Require both lane boundaries and a real bend before acting on a sign.
        This is deliberately conservative so a sign seen at spawn cannot
        immediately produce a steering command.
        """
        if message.vector_count != 2:
            if DEBUG_LOG:
                self.get_logger().info(
                    "[DEBUG_LOG] turn_geometry_detected: False "
                    f"(vector_count={message.vector_count} != 2)"
                )
            return False
        if not message.vector_1 or not message.vector_2:
            if DEBUG_LOG:
                self.get_logger().info(
                    "[DEBUG_LOG] turn_geometry_detected: False "
                    f"(vector_1 empty={not message.vector_1}, "
                    f"vector_2 empty={not message.vector_2})"
                )
            return False

        distance, angle = self.selected_edge_geometry(message)
        if distance is None:
            if DEBUG_LOG:
                self.get_logger().info(
                    "[DEBUG_LOG] turn_geometry_detected: False (distance=None)"
                )
            return False

        # A normal straight edge is close to vertical in the image. The
        # selected edge must actually bend before lane-stick begins.
        result = abs(angle) >= TURN_START_ANGLE
        if DEBUG_LOG and not result:
            self.get_logger().info(
                f"[DEBUG_LOG] turn_geometry_detected: False "
                f"(|angle|={abs(angle):.3f} < TURN_START_ANGLE={TURN_START_ANGLE})"
            )
        return result

    def start_lane_stick(self, direction):
        self.stick_side = direction
        self.lane_mode = "STICK"
        self.turn_start_time = (
            self.get_clock().now().nanoseconds * 1e-9
        )
        self.turn_complete_count = 0
        self.get_logger().info(
            f"Starting {direction} lane-stick: sign armed, intersection reached."
        )

    def update_lane_stick(self, message):
        """Maintain a safe distance from the selected track edge."""
        if self.stick_side not in ("Left", "Right"):
            if DEBUG_LOG:
                self.get_logger().info(
                    "[DEBUG_LOG] update_lane_stick: invalid stick_side="
                    f"{self.stick_side!r}, forcing lane_mode back to CENTER"
                )
            self.lane_mode = "CENTER"
            return

        # During an active turn, use stick_side rather than pending_turn.
        old_pending = self.pending_turn
        self.pending_turn = self.stick_side
        distance, angle = self.selected_edge_geometry(message)
        self.pending_turn = old_pending

        if distance is None:
            # Never spin blindly when the selected edge disappears.
            if DEBUG_LOG:
                self.get_logger().info(
                    f"[DEBUG_LOG] update_lane_stick ({self.stick_side}): "
                    "selected edge LOST -> decaying turn, capping speed"
                )
            self.target_turn *= 0.75
            self.target_speed = min(self.target_speed, LANE_STICK_SPEED)
            return

        error = distance - LANE_STICK_DISTANCE_PX

        # IMPORTANT SIGN CONVENTION FROM THE OLD FILE:
        # positive turn = left, negative turn = right.
        if self.stick_side == "Right":
            turn = -(LANE_STICK_KP * error)
            turn += -(LANE_STICK_KANGLE * angle)
        else:
            turn = +(LANE_STICK_KP * error)
            turn += +(LANE_STICK_KANGLE * angle)

        turn = max(-LANE_STICK_MAX_TURN, min(LANE_STICK_MAX_TURN, turn))

        # Smooth the command; never jump to a full steering value.
        self.target_turn = 0.25 * turn + 0.75 * self.target_turn
        self.target_turn = max(
            -LANE_STICK_MAX_TURN,
            min(LANE_STICK_MAX_TURN, self.target_turn),
        )
        self.target_speed = LANE_STICK_SPEED

        if DEBUG_LOG:
            self.get_logger().info(
                f"[DEBUG_LOG] update_lane_stick ({self.stick_side}, {self.lane_mode}): "
                f"distance={distance:.1f} error={error:.1f} angle={angle:.3f} "
                f"raw_turn={turn:.3f} smoothed_turn={self.target_turn:.3f} "
                f"turn_complete_count={self.turn_complete_count}"
            )

        # A bend has to straighten for several frames before we call the turn
        # complete. This prevents immediate hand-back at the intersection.
        now = self.get_clock().now().nanoseconds * 1e-9
        elapsed = now - self.turn_start_time

        if abs(angle) < TURN_COMPLETE_ANGLE:
            self.turn_complete_count += 1
        else:
            self.turn_complete_count = 0

        if (
            self.lane_mode == "STICK"
            and abs(angle) >= TURN_START_ANGLE
        ):
            self.lane_mode = "TURNING"
            if DEBUG_LOG:
                self.get_logger().info(
                    f"[DEBUG_LOG] lane_mode STICK -> TURNING ({self.stick_side})"
                )

        if (
            self.lane_mode == "TURNING"
            and (
                self.turn_complete_count >= TURN_COMPLETE_FRAMES
                or elapsed >= TURN_MAX_TIME
            )
        ):
            self.lane_mode = "RETURN_CENTER"
            self.return_count = 0
            self.turn_complete_count = 0
            self.get_logger().info(
                "Turn complete. Blending back to normal centre following."
            )
            if DEBUG_LOG:
                self.get_logger().info(
                    f"[DEBUG_LOG] lane_mode TURNING -> RETURN_CENTER "
                    f"(elapsed={elapsed:.2f}s, "
                    f"hit_time_cap={elapsed >= TURN_MAX_TIME})"
                )

    def lidar_callback(self, message):
        """
        Receives LIDAR range measurements.
        
        GUIDELINE (Obstacle Avoidance & Building Range):
        - `message.ranges` is an array of distances in meters around the buggy.
        - The laser scans cover 360 degrees. Find which indices correspond to the front of the buggy.
        - If a range value in the front sector is below a threshold (e.g. 0.8m), flag an obstacle.
        - Write obstacle avoidance maneuvers (e.g. stop, steer left/right around the block, and merge back).
        - Use LIDAR side-ranges to verify distance to building/QR signs before patient pickup/hospital drop actions.
        """
        # --------------------------------------------------------------
        # FIX #2 (works together with FIX #1 above):
        # While a sign-commanded turn is actively executing (STICK/TURNING),
        # the buggy is deliberately hugging close to a track edge/curb --
        # that's expected proximity, not an obstacle. Previously this
        # function had no awareness of lane_mode at all, so it would happily
        # set self.avoid = True from ordinary turn-time proximity and steer
        # off on its own, fighting (and, per FIX #1, completely overriding)
        # the lane-stick controller. Right turns were more likely to expose
        # this since junction geometry differs left vs right.
        #
        # We simply don't run generic avoidance while a turn is in progress;
        # lane-stick's own distance-keeping (LANE_STICK_DISTANCE_PX) already
        # handles safe clearance during the maneuver.
        # --------------------------------------------------------------
        if self.lane_mode in ("STICK", "TURNING"):
            if self.avoid:
                if DEBUG_LOG:
                    self.get_logger().info(
                        "[DEBUG_LOG] lidar_callback: clearing stale avoid=True "
                        f"because lane_mode={self.lane_mode} (turn in progress)"
                    )
                self.avoid = False
            if DEBUG_LOG:
                self.get_logger().info(
                    f"[DEBUG_LOG] lidar_callback: SKIPPED (lane_mode={self.lane_mode})"
                )
            return

        # HINTS:
        n = len(message.ranges)
        right_sector = message.ranges[int(n * 7/18): int(n * 9/18)]
        left_sector = message.ranges[int(n * 9/18): int(n * 11/18)]
        sector= right_sector if min(right_sector)<min(left_sector) else left_sector
        if not self.approaching:
            if min(sector)<0.8:
                spd =min(self.target_speed,min(sector)*self.expconst/0.8 +(1-self.expconst)*self.target_speed)
                self.avoid=True
                if DEBUG_LOG:
                    self.get_logger().info(
                        "[DEBUG_LOG] lidar_callback: OBSTACLE -> avoid=True "
                        f"(min_dist={min(sector):.2f}m, "
                        f"side={'right' if sector is right_sector else 'left'})"
                    )
                if sector==right_sector :
                    lowerbound=n*7/18
                else :
                    lowerbound=n*9/18
                offset=(9*n/18-(sector.index(min(sector))+lowerbound))
                if offset!=0:
                    ang=(1-abs(offset)/(2*n/18))*(abs(offset)/offset)
                else : ang= 0.1 if lowerbound==n*7/18 else -0.1
                angle = self.expconst * ang + (1 - self.expconst) * self.target_turn

                self.rover_move_manual_mode(spd,angle)
            else:
                if self.avoid and DEBUG_LOG:
                    self.get_logger().info(
                        "[DEBUG_LOG] lidar_callback: obstacle cleared -> avoid=False"
                    )
                self.avoid=False
        else:
            left_part = message.ranges[int(n*10/18):int(n*14/18)]
            right_part = message.ranges[int(n*4/18):int(n*8/18)]
            min_left, min_right = min(left_part), min(right_part)
            if min(min_left, min_right) < 3:
                if min_left < min_right:
                    idx = int(n*10/18) + left_part.index(min_left)
                    offset = (n/2 - idx) / (4*n/18)
                    spd=self.target_speed*(1-abs(offset))
                    self.rover_move_manual_mode(spd, self.target_turn)
                else:
                    idx = int(n*4/18) + right_part.index(min_right)
                    offset = (n/2 - idx) / (4*n/18)
                    spd=self.target_speed*(1-abs(offset))
                    self.rover_move_manual_mode(spd, self.target_turn)
     

                if self.target_speed<0.15:
                    self.target_speed=0
                    self.approaching=False
                    self.reached=True

        

		
        
        pass

    def server_communication_callback(self, message):
        """
        Receives 0coordination commands from the server.
        
        GUIDELINE (Server Communication):
        - Check if the message is destined for the Buggy (`message.dest == 1`).
		- Do not forget to check for ACK messages from server
        - The server communicates mission info in the `message.msg` payload string.
        - Parse server instructions (e.g., patient pickup, target hospitals).
        - Call `self.send_server_update` to report your status when you reach a checkpoint.
        """
        
        if message.dest == 1:
            self.get_logger().info(f"Received Server Message: {message.msg}")
            self.destination=message.msg
            self.ackno=1

                
            if message.ack==1:
                self.qr_data=""
                self.patient_id=0
                self.hospital_id=0
                
        if self.reached:
            if self.patient_id in [1,2,3]:
                self.send_server_update(self.qr_data)
            if self.hospital_id in [1,2,3]:
                self.send_server_update(self.qr_data)
            if self.hospital_id==3:
                self.mission_completed=1


            

    def send_server_update(self, text_msg):
        """Sends status messages to the server. (Do not forget to send ACK messages to server)"""
        server_msg = ServerCommunication()
        server_msg.src = 1       # Source component: Buggy-1
        server_msg.dest = 2      # Destination component: Server-2
        server_msg.uid = 100     # Replace with a rolling message ID/counter
        server_msg.ack = self.ackno
        server_msg.msg = text_msg
        self.publisher_server.publish(server_msg)
        self.ackno=0

    def qr_detection_callback(self, message):
        """
        Receives QR codes scanned from the buildings.
        
        GUIDELINE (Patient/Hospital Identification):
        - Parse the decoded string payload in `message.data` (e.g. "PATIENT_A", "HOSPITAL_B").
        - If it matches your target destination, stop the vehicle close to the building (verify range using LIDAR),
          perform the action (pick patient / drop patient), and communicate the arrival to the server.
        """
        self.get_logger().info(f"Heard QR code: {message.data}")
        map1={"X":1,"Y":2,"Z":3}
        map2={"A":1,"B":2,"C":3}
        if message.data.startswith("{LOC_PATIENT_"):
            self.patient_id = int(message.data.strip("{}").split("_")[-1])
            self.get_logger().info(f"Identified Patient: {self.patient_id}")
            if self.destination in ["A","B","C"]:
                if self.patient_id==map2[self.destination]:
                    self.get_logger().info(f"Approaching target patient location: {self.patient_id}")
                    self.approaching = True
                    self.approaching = True
                    self.avoid = False
                    self.qr_data=message.data
            else: 
                self.approaching=False
                self.patient_id=0
        elif message.data.startswith("{LOC_HOSPITAL_"):
            self.hospital_id = int(message.data.strip("{}").split("_")[-1])
            self.get_logger().info(f"Identified Hospital: {self.hospital_id}")
            
            if self.destination in ["X","Y","Z"]:
                if self.hospital_id==map1[self.destination]:
                    self.get_logger().info(f"Approaching target hospital location: {self.hospital_id}")
                    self.approaching = True
                    self.qr_data=message.data
                    self.avoid = False
            else:
                self.approaching = False
                self.reached=False
                self.hospital_id=0
        pass

    def sign_board_callback(self, message):
        """
        Sign-board routing.

        The detector only stores a pending turn. It does NOT set an immediate
        steering override. This fixes the original problem where a sign seen
        from the spawn point caused an instant turn.
        """
        self.get_logger().info(f"Heard Sign Board: {message.data}")

        entries = []
        for raw in message.data.split(";"):
            raw = raw.strip()
            if not raw:
                continue

            parts = raw.split(":", 1)
            if len(parts) != 2:
                continue

            try:
                entries.append([parts[0].strip(), float(parts[1])])
            except ValueError:
                continue

        destentry = None
        for entry in entries:
            if entry[0] == self.destination:
                destentry = entry
                break

        if destentry is None:
            if DEBUG_LOG:
                self.get_logger().info(
                    f"[DEBUG_LOG] sign_board_callback: no entry matched "
                    f"destination={self.destination!r} in entries={entries}"
                )
            return

        candidates = [
            entry for entry in entries
            if entry[0] in ("Left", "Right", "Straight")
        ]
        if not candidates:
            return

        # DEBUG_LOG: if two candidates tie in distance-to-destination, min()
        # deterministically picks the FIRST one in `entries` order -- worth
        # confirming this isn't quietly biasing Left vs Right selection.
        nearest = min(
            candidates,
            key=lambda entry: abs(entry[1] - destentry[1]),
        )
        if DEBUG_LOG:
            diffs = [(c[0], abs(c[1] - destentry[1])) for c in candidates]
            self.get_logger().info(
                f"[DEBUG_LOG] sign_board_callback: destentry={destentry} "
                f"candidates(diff)={diffs} -> nearest={nearest}"
            )

        if abs(nearest[1] - destentry[1]) >= 0.1:
            if DEBUG_LOG:
                self.get_logger().info(
                    f"[DEBUG_LOG] sign_board_callback: nearest candidate too far "
                    f"(diff={abs(nearest[1] - destentry[1]):.3f} >= 0.1) -> ignored"
                )
            return

        direction = nearest[0]

        if direction == "Straight":
            self.pending_turn = None
            self.sign_candidate = None
            self.sign_candidate_count = 0
            return

        # Require repeated identical detections before arming a maneuver.
        if direction == self.sign_candidate:
            self.sign_candidate_count += 1
        else:
            self.sign_candidate = direction
            self.sign_candidate_count = 1

        if self.sign_candidate_count < 3:
            if DEBUG_LOG:
                self.get_logger().info(
                    f"[DEBUG_LOG] sign_board_callback: {direction} candidate "
                    f"count={self.sign_candidate_count}/3, not armed yet"
                )
            return

        self.pending_turn = direction
        self.pending_turn_time = (
            self.get_clock().now().nanoseconds * 1e-9
        )
        self.sign_candidate_count = 0

        self.get_logger().info(
            f"Pending {direction} turn armed. Continuing normal lane following "
            "until intersection geometry confirms the turn."
        )


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
