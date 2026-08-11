#!/usr/bin/env python3
"""
Dummy Server Node
==================
Simulates the SERVER side (src=2) of the Buggy-Server Communication Protocol.

Sequence implemented (as requested):

    Buggy sends "A"  -> Server acks A
    Server sends "X" -> Buggy acks X
    Buggy sends "B"  -> Server acks B
    Server sends "Y" -> Buggy acks Y
    Buggy sends "C"  -> Server acks C
    Server sends "Z" -> Buggy acks Z
    Buggy sends "PARKED" -> Server replies "OK" (or "INVALID")

Assumes both Buggy and Server publish/subscribe on the same topic
"/server_communication" using the ServerCommunication.msg type:

    uint8 src
    uint8 dest
    uint8 uid
    uint8 ack
    string msg

Adjust TOPIC_NAME below if your actual topic differs.
"""

import rclpy
from rclpy.node import Node

from synapse_msgs.msg import ServerCommunication

BUGGY_ID = 1
SERVER_ID = 2

TOPIC_NAME = "/server_communication"

# Ordered mission plan: (expected patient QR from buggy, next destination to send)
MISSION_PLAN = [
    ("A", "X"),
    ("B", "Y"),
    ("C", "Z"),
]

# Internal states
STATE_WAIT_QR = "WAIT_QR"                  # waiting for buggy to report a scanned QR
STATE_WAIT_DEST_ACK = "WAIT_DEST_ACK"      # waiting for buggy to ack the destination we sent
STATE_WAIT_PARKED = "WAIT_PARKED"          # all destinations sent, waiting for "PARKED"
STATE_DONE = "DONE"


class DummyServerNode(Node):

    def __init__(self):
        super().__init__("dummy_server_node")

        self.publisher_ = self.create_publisher(
            ServerCommunication, TOPIC_NAME, 10
        )
        self.subscription_ = self.create_subscription(
            ServerCommunication, TOPIC_NAME, self.listener_callback, 10
        )

        # Mission progress tracking
        self.mission_index = 0
        self.state = STATE_WAIT_QR

        # Rolling uid counter for messages the SERVER originates (destination pushes)
        self.server_uid = 100

        # Delay timer (used to mimic "brief validation delay" before sending next dest)
        self.delay_timer = None

        self.get_logger().info("Dummy Server Node started. Waiting for buggy...")
        self.get_logger().info(f"Mission plan: {MISSION_PLAN}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def send_message(self, uid: int, msg: str, ack: int = 0):
        """Publish a message from Server (src=2) to Buggy (dest=1)."""
        out = ServerCommunication()
        out.src = SERVER_ID
        out.dest = BUGGY_ID
        out.uid = uid
        out.ack = ack
        out.msg = msg
        self.publisher_.publish(out)
        self.get_logger().info(
            f"SENT  -> uid={uid} ack={ack} msg='{msg}'"
        )

    def send_ack(self, uid: int):
        """Send an empty acknowledgment for a given uid."""
        self.send_message(uid=uid, msg="", ack=1)

    def schedule_next_destination(self, delay_sec: float = 1.5):
        """Wait briefly ('validation delay') then push the next destination."""
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

    # ------------------------------------------------------------------
    # Main callback / state machine
    # ------------------------------------------------------------------

    def listener_callback(self, msg: ServerCommunication):
        # Ignore anything not addressed to the server
        if msg.dest != SERVER_ID:
            return
        # Ignore our own echoed messages if topic is shared
        if msg.src != BUGGY_ID:
            return

        self.get_logger().info(
            f"RECV  <- uid={msg.uid} ack={msg.ack} msg='{msg.msg}'"
        )

        # --- STATE: waiting for buggy to report the patient QR text ---
        if self.state == STATE_WAIT_QR:
            expected_qr, _ = MISSION_PLAN[self.mission_index]

            if msg.ack == 0 and msg.msg == expected_qr:
                # 1. Ack the QR immediately
                self.send_ack(msg.uid)
                # 2. Brief validation delay, then send next destination
                self.schedule_next_destination()
            else:
                self.get_logger().warn(
                    f"Expected QR '{expected_qr}' but got '{msg.msg}' — ignoring."
                )

        # --- STATE: waiting for buggy to ack the destination we sent ---
        elif self.state == STATE_WAIT_DEST_ACK:
            if msg.ack == 1 and msg.uid == self.server_uid:
                self.get_logger().info(
                    f"Destination '{MISSION_PLAN[self.mission_index][1]}' confirmed by buggy."
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
                    f"Expected ack for uid={self.server_uid}, got uid={msg.uid} ack={msg.ack}"
                )

        # --- STATE: waiting for final PARKED confirmation ---
        elif self.state == STATE_WAIT_PARKED:
            if msg.ack == 0 and msg.msg == "PARKED":
                is_valid = self.validate_parking()
                reply = "OK" if is_valid else "INVALID"
                self.send_message(uid=msg.uid, msg=reply, ack=1)

                if is_valid:
                    self.get_logger().info("Buggy parked correctly. Run complete.")
                    self.state = STATE_DONE
                else:
                    self.get_logger().warn("Buggy NOT parked correctly. Awaiting retry.")
                    # stays in STATE_WAIT_PARKED so buggy can resend "PARKED"

        elif self.state == STATE_DONE:
            self.get_logger().info("Mission already complete — ignoring further messages.")

    # ------------------------------------------------------------------
    # Parking validation stub — replace with real map/geofence check
    # ------------------------------------------------------------------
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