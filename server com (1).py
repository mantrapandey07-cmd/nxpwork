#!/usr/bin/env python3
"""
Dummy Server Node — simulates the Municipality Server for NXP CUP 2026.

Protocol (fixed):

  Buggy sends "{LOC: PATIENT_1}"  →  Server acks, then sends "HOSPITAL_2"
  Buggy sends "{LOC: HOSPITAL_2}" →  Server acks, then sends "PATIENT_2"
  Buggy sends "{LOC: PATIENT_2}"  →  Server acks, then sends "HOSPITAL_1"
  Buggy sends "{LOC: HOSPITAL_1}" →  Server acks, then sends "PATIENT_3"
  Buggy sends "{LOC: PATIENT_3}"  →  Server acks, then sends "HOSPITAL_3"
  Buggy sends "{LOC: HOSPITAL_3}" →  Server acks, then sends "MISSION_COMPLETE"
  Buggy sends "PARKED"            →  Server replies "OK" (or "INVALID")

Topic: /ServerCommunication  (Bug #16 fix: was /server_communication)

The buggy sends raw QR payloads like ``{LOC: PATIENT_1}``.
The server extracts the location name and matches it against
the mission plan.  Replies are plain location names like ``HOSPITAL_2``.
"""

import rclpy
from rclpy.node import Node

from synapse_msgs.msg import ServerCommunication

BUGGY_ID  = 1
SERVER_ID = 2

# Bug #16 fix: topic name now matches the buggy's /ServerCommunication
TOPIC_NAME = "/ServerCommunication"

# Bug #17 fix: MISSION_PLAN now expects QR-extracted location names
# (e.g. "PATIENT_1") instead of bare letters ("A").
#
# Each tuple: (expected_location_from_buggy, server_reply_to_send)
MISSION_PLAN = [
    ("PATIENT_1",  "HOSPITAL_2"),        # Patient 1 → Hospital 2
    ("HOSPITAL_2", "PATIENT_2"),          # Hospital 2 confirmed → Patient 2
    ("PATIENT_2",  "HOSPITAL_1"),        # Patient 2 → Hospital 1
    ("HOSPITAL_1", "PATIENT_3"),          # Hospital 1 confirmed → Patient 3
    ("PATIENT_3",  "HOSPITAL_3"),        # Patient 3 → Hospital 3
    ("HOSPITAL_3", "MISSION_COMPLETE"),  # Hospital 3 confirmed → Done
]

# Internal states
STATE_WAIT_QR        = "WAIT_QR"
STATE_WAIT_DEST_ACK  = "WAIT_DEST_ACK"
STATE_WAIT_PARKED    = "WAIT_PARKED"
STATE_DONE           = "DONE"


def extract_location(qr_payload):
    """
    Extract the location name from a QR payload string.

    "{LOC: PATIENT_1}" → "PATIENT_1"
    "{LOC: HOSPITAL_2}" → "HOSPITAL_2"

    Bug #17 fix: the dummy server now correctly parses the real QR
    format instead of expecting bare letters.
    """
    s = qr_payload.strip()
    if s.startswith("{") and s.endswith("}"):
        s = s[1:-1].strip()
    if ":" in s:
        s = s.split(":", 1)[1].strip()
    return s


class DummyServerNode(Node):

    def __init__(self):
        super().__init__("dummy_server_node")

        self.publisher_ = self.create_publisher(
            ServerCommunication, TOPIC_NAME, 10
        )
        self.subscription_ = self.create_subscription(
            ServerCommunication, TOPIC_NAME, self.listener_callback, 10
        )

        self.mission_index = 0
        self.state          = STATE_WAIT_QR
        self.server_uid     = 100
        self.delay_timer     = None

        self.get_logger().info("Dummy Server Node started. Waiting for buggy...")
        self.get_logger().info(f"Mission plan: {MISSION_PLAN}")

    # ── Helpers ──────────────────────────────────────────────────

    def send_message(self, uid, msg, ack=0):
        out = ServerCommunication()
        out.src  = SERVER_ID
        out.dest = BUGGY_ID
        out.uid  = uid
        out.ack  = ack
        out.msg  = msg
        self.publisher_.publish(out)
        self.get_logger().info(f"SENT  -> uid={uid} ack={ack} msg='{msg}'")

    def send_ack(self, uid):
        self.send_message(uid=uid, msg="", ack=1)

    def schedule_next_destination(self, delay_sec=1.5):
        if self.delay_timer is not None:
            self.delay_timer.cancel()

        def _fire():
            self.delay_timer.cancel()
            self.delay_timer = None
            self._send_next_destination()

        self.delay_timer = self.create_timer(delay_sec, _fire)

    def _send_next_destination(self):
        _, dest = MISSION_PLAN[self.mission_index]
        self.server_uid = (self.server_uid + 1) % 256
        self.send_message(uid=self.server_uid, msg=dest, ack=0)
        self.state = STATE_WAIT_DEST_ACK

    # ── Main state machine ──────────────────────────────────────

    def listener_callback(self, msg: ServerCommunication):
        if msg.dest != SERVER_ID or msg.src != BUGGY_ID:
            return

        self.get_logger().info(
            f"RECV  <- uid={msg.uid} ack={msg.ack} msg='{msg.msg}'"
        )

        # ── Waiting for buggy to report a scanned QR ──
        if self.state == STATE_WAIT_QR:
            expected_qr, _ = MISSION_PLAN[self.mission_index]

            if msg.ack == 0 and msg.msg:
                # Bug #17 fix: extract location from QR payload
                received = extract_location(msg.msg)
                if received == expected_qr:
                    self.send_ack(msg.uid)
                    self.schedule_next_destination()
                else:
                    self.get_logger().warn(
                        f"Expected '{expected_qr}' but got '{received}' "
                        f"(raw: '{msg.msg}') — ignoring."
                    )

        # ── Waiting for buggy to ack the destination we sent ──
        elif self.state == STATE_WAIT_DEST_ACK:
            if msg.ack == 1 and msg.uid == self.server_uid:
                self.get_logger().info(
                    f"Destination '{MISSION_PLAN[self.mission_index][1]}' "
                    f"confirmed by buggy."
                )
                self.mission_index += 1

                if self.mission_index < len(MISSION_PLAN):
                    self.state = STATE_WAIT_QR
                else:
                    self.state = STATE_WAIT_PARKED
                    self.get_logger().info(
                        "All destinations dispatched. Waiting for 'PARKED'..."
                    )
            else:
                self.get_logger().warn(
                    f"Expected ack for uid={self.server_uid}, "
                    f"got uid={msg.uid} ack={msg.ack}"
                )

        # ── Waiting for final PARKED confirmation ──
        elif self.state == STATE_WAIT_PARKED:
            if msg.ack == 0 and msg.msg.strip() == "PARKED":
                is_valid = self.validate_parking()
                reply = "OK" if is_valid else "INVALID"
                self.send_message(uid=msg.uid, msg=reply, ack=1)

                if is_valid:
                    self.get_logger().info(
                        "Buggy parked correctly. Run complete."
                    )
                    self.state = STATE_DONE
                else:
                    self.get_logger().warn(
                        "Buggy NOT parked correctly. Awaiting retry."
                    )

        elif self.state == STATE_DONE:
            self.get_logger().info(
                "Mission already complete — ignoring further messages."
            )

    # ── Parking validation stub ─────────────────────────────────

    def validate_parking(self) -> bool:
        """
        Placeholder for real parking validation logic
        (e.g. checking buggy's last known pose against a parking geofence).
        Currently always returns True for this dummy server.
        """
        return True


def main(args=None):
    rclpy.init(args=args)
    node = DummyServerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
