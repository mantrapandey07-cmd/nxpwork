# Copyright 2024-2026 NXP
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
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String
import cv2
import numpy as np
import os

# Bug #13 fix: YOLO (ultralytics) does NOT depend on TensorFlow.
# The old code gated model loading behind ``import tensorflow``,
# which meant the model was never loaded if TF was absent — even
# though ultralytics was installed correctly.
from ultralytics import YOLO


class ObjectRecognizer(Node):
    """
    ROS 2 Node that processes raw camera images to recognize traffic
    sign boards using a YOLO model.  Publishes detected sign labels
    on the ``/sign_board_detection`` topic.

    Bug #14 fix: model filename corrected from ``model.pt`` to ``best.pt``.
    Bug #15 fix: stale "Keras model.h5" comment removed.
    """

    def __init__(self):
        super().__init__('object_recognizer')

        # Subscription for camera images.
        self.subscription_camera = self.create_subscription(
            CompressedImage,
            '/camera/image_raw/compressed',
            self.camera_image_callback,
            10)

        # Publisher for sign board detection results.
        self.publisher_sign = self.create_publisher(
            String,
            '/sign_board_detection',
            10)

        # Load the YOLO model (best.pt) located in the same directory.
        self.model = None
        try:
            dir_path   = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(dir_path, 'best.pt')   # Bug #14 fix
            if os.path.exists(model_path):
                self.model = YOLO(model_path)
                self.get_logger().info(f"Loaded YOLO model from {model_path}")
            else:
                self.get_logger().warn(f"Model file not found at {model_path}")
        except Exception as e:
            self.get_logger().error(f"Failed to load YOLO model: {e}")

        self.get_logger().info("Object Recognizer Node started. Waiting for images...")

    def camera_image_callback(self, message):
        """Processes incoming camera frames to classify traffic signs."""
        np_arr = np.frombuffer(message.data, np.uint8)
        image  = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        sign_detected = self.classify_sign(image)

        if sign_detected is not None:
            msg = String()
            msg.data = sign_detected
            self.publisher_sign.publish(msg)
            self.get_logger().info(f"Detected Sign Board: {sign_detected}")

    def classify_sign(self, image):
        """Run YOLO inference and return semicolon-separated detections."""
        if self.model is None:
            return None

        h, w = image.shape[:2]
        edge_margin = 5

        try:
            results = self.model.predict(image, verbose=False)
        except Exception as e:
            self.get_logger().debug(f"Inference failed: {e}")
            return None

        if not results or len(results[0].boxes) == 0:
            return None

        detections = []
        for box in results[0].boxes:
            conf = float(box.conf[0])
            if conf < 0.5:
                continue

            x1, y1, x2, y2 = box.xyxy[0].tolist()

            # Reject detections clipped at image edges
            if x1 <= edge_margin or y1 <= edge_margin or \
               x2 >= (w - edge_margin) or y2 >= (h - edge_margin):
                continue

            label          = self.model.names[int(box.cls[0])]
            x_center_norm  = ((x1 + x2) / 2) / w

            detections.append(f"{label}:{x_center_norm:.3f}")

        if not detections:
            return None

        return ";".join(detections)


def main(args=None):
    rclpy.init(args=args)
    node = ObjectRecognizer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
