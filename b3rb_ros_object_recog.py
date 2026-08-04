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

import os
import json
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String
import cv2
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

class ObjectRecognizer(Node):
    """
    ROS 2 Node that processes raw camera images to recognize traffic sign
    boards using an Ultralytics YOLO model.
    It dynamically maps letters to directions and publishes the JSON layout.
    """
    def __init__(self):
        super().__init__('object_recognizer')

        self.subscription_camera = self.create_subscription(
            CompressedImage,
            '/camera/image_raw/compressed',
            self.camera_image_callback,
            10)

        self.publisher_sign = self.create_publisher(
            String,
            '/sign_board_detection',
            10)

        self.publisher_debug = self.create_publisher(
            CompressedImage,
            '/debug_images/sign_detection',
            10)

        self.model = None
        if YOLO is not None:
            try:
                dir_path = os.path.dirname(os.path.abspath(__file__))
                model_path = os.path.join(dir_path, 'best.pt')
                
                if os.path.exists(model_path):
                    self.model = YOLO(model_path)
                    self.get_logger().info(f"Loaded YOLO model from {model_path}")
                else:
                    self.get_logger().warn(f"Model file not found at {model_path}")
            except Exception as e:
                self.get_logger().error(f"Failed to load YOLO model: {e}")
        else:
            self.get_logger().warn("Ultralytics is not installed. Please install it using: pip install ultralytics")

        self.get_logger().info("Object Recognizer Node started. Waiting for images...")

    def camera_image_callback(self, message):
        """Processes incoming camera frames to classify traffic signs and publishes results."""
        np_arr = np.frombuffer(message.data, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        if image is None:
            return

        # Get the JSON map and the annotated debug image
        board_map_json, debug_img = self.classify_sign(image)

        if board_map_json is not None:
            msg = String()
            msg.data = board_map_json
            self.publisher_sign.publish(msg)
            self.get_logger().info(f"Published Board Layout: {board_map_json}")

        # Publish the annotated debug frame viewable in Foxglove
        if debug_img is not None:
            debug_msg = CompressedImage()
            _, encoded = cv2.imencode('.jpg', debug_img)
            debug_msg.format = "jpeg"
            debug_msg.data = encoded.tobytes()
            self.publisher_debug.publish(debug_msg)
            
    def classify_sign(self, image):
        """Runs YOLO inference and geometrically pairs letters with arrows."""
        if self.model is None:
            return None, image

        try:
            # Run inference directly with Ultralytics
            # imgsz=320 keeps performance optimized for the Raspberry Pi
            results = self.model.predict(source=image, imgsz=320, conf=0.55, verbose=False)
        except Exception as e:
            self.get_logger().debug(f"Inference failed: {e}")
            return None, image

        # Generate Ultralytics annotated image automatically
        annotated_img = results[0].plot() if len(results) > 0 else image

        if len(results) == 0 or len(results[0].boxes) == 0:
            return None, annotated_img

        boxes = results[0].boxes
        letters = []
        arrows = []
        
        # Parse all detected bounding boxes
        for i in range(len(boxes)):
            box = boxes[i]
            cls_id = int(box.cls[0])
            label = results[0].names[cls_id]
            
            # Get coordinates to calculate horizontal center (x-axis)
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            box_cx = (x1 + x2) / 2.0
            
            if label in ['Left', 'Right', 'Straight']:
                arrows.append((label, box_cx))
            else:
                letters.append((label, box_cx))
                
        # If we don't see both parts of the sign, we can't make a safe decision
        if not letters or not arrows:
            return None, annotated_img
            
        # Match each letter to the arrow directly underneath it (closest X-center)
        board_map = {}
        for l_label, l_cx in letters:
            closest_arrow = min(arrows, key=lambda a: abs(a[1] - l_cx))
            board_map[l_label] = closest_arrow[0]
            
        return json.dumps(board_map), annotated_img

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
    
