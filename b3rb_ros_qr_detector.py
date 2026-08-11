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

# Pyzbar is used as a fallback decoder when OpenCV's built-in detector fails.
# pip install pyzbar   (Linux also needs: sudo apt install libzbar0)
try:
    from pyzbar import pyzbar
except ImportError:
    pyzbar = None


class QRDetector(Node):
    """
    ROS 2 Node that processes raw camera images to scan for QR codes.
    It publishes the detected QR code payload on the `/qr_detection` topic.

    Detection pipeline:
      1. Generate a handful of preprocessed variants of the incoming frame
         (grayscale, contrast-enhanced, adaptive-thresholded, upscaled,
         sharpened) since a raw camera frame is often too dark, too small,
         or too noisy for a QR decoder to lock onto directly.
      2. Try OpenCV's built-in QRCodeDetector on each variant.
      3. If OpenCV comes up empty on every variant, fall back to pyzbar
         (zbar), which uses a different decoding algorithm and often
         succeeds where OpenCV fails, especially on skewed or partially
         occluded codes.
    """

    def __init__(self):
        super().__init__('qr_detector')

        # Subscription for camera images.
        self.subscription_camera = self.create_subscription(
            CompressedImage,
            '/camera/image_raw/compressed',
            self.camera_image_callback,
            10)

        # Publisher for QR code detection results.
        self.publisher_qr = self.create_publisher(
            String,
            '/qr_detection',
            10)

        self.cv_detector = cv2.QRCodeDetector()

        # Only fall back to pyzbar once OpenCV has failed this many
        # consecutive frames - pyzbar is slower, so we don't want to pay
        # for it on every frame just because of a single missed detect.
        self.pyzbar_fallback_threshold = 5
        self.consecutive_failures = 0

        if pyzbar is None:
            self.get_logger().warn(
                "pyzbar not installed - fallback decoding disabled. "
                "Install with: pip install pyzbar (and 'sudo apt install libzbar0' on Linux)"
            )

        self.get_logger().info("QR Detector Node started. Waiting for images...")

    # ------------------------------------------------------------------
    # Image callback
    # ------------------------------------------------------------------
    def camera_image_callback(self, message):
        """Processes incoming camera frames to detect QR codes."""
        np_arr = np.frombuffer(message.data, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if image is None:
            self.get_logger().debug("Failed to decode incoming compressed image.")
            return

        qr_data = self.detect_qr_code(image)

        if qr_data is not None:
            msg = String()
            msg.data = qr_data
            self.publisher_qr.publish(msg)
            self.get_logger().info(f"Published QR Data: {qr_data}")

    # ------------------------------------------------------------------
    # Preprocessing helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _to_gray(image):
        if len(image.shape) == 2:
            return image
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def _clahe(gray):
        """Contrast Limited Adaptive Histogram Equalization - helps a lot
        with uneven lighting / shadows on the QR board."""
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(gray)

    @staticmethod
    def _adaptive_threshold(gray):
        """Binarizes the image locally - useful when the code is lit
        unevenly or the background has similar brightness to the code."""
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        return cv2.adaptiveThreshold(
            blurred, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=25,
            C=7
        )

    @staticmethod
    def _sharpen(gray):
        """Simple unsharp-mask style kernel to recover edge detail lost to
        motion blur or a slightly out-of-focus camera."""
        kernel = np.array([[0, -1, 0],
                            [-1, 5, -1],
                            [0, -1, 0]])
        return cv2.filter2D(gray, -1, kernel)

    @staticmethod
    def _upscale(gray, factor=2.0):
        """QR codes that are small in-frame (far from the camera) often
        fail to decode until they're enlarged."""
        return cv2.resize(
            gray, None, fx=factor, fy=factor,
            interpolation=cv2.INTER_CUBIC
        )

    def _generate_candidates(self, image):
        """Yields a small, ordered set of preprocessed images to try.
        Ordered roughly cheapest/most-likely-to-work first so we bail
        out early on the common case instead of always paying for every
        transform."""
        gray = self._to_gray(image)

        yield image        # original color frame
        yield gray          # plain grayscale

        clahe_img = self._clahe(gray)
        yield clahe_img

        yield self._adaptive_threshold(gray)
        yield self._sharpen(clahe_img)
        yield self._upscale(gray)
        yield self._upscale(clahe_img)

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------
    def detect_qr_code(self, image):
        """
        Detect and decode a QR code in the image.

        Tries OpenCV's detector across several preprocessed variants of
        the frame first (fast path, runs every frame). Pyzbar is only
        invoked once OpenCV has failed on `pyzbar_fallback_threshold`
        consecutive frames, since it's slower and we don't want to pay
        for it just because of one missed frame here and there.
        """
        candidates = list(self._generate_candidates(image))

        # --- Method 1: OpenCV built-in QR Detector (every frame) ---
        for candidate in candidates:
            try:
                data, bbox, _ = self.cv_detector.detectAndDecode(candidate)
                if bbox is not None and data:
                    self.consecutive_failures = 0
                    return data
            except Exception as e:
                self.get_logger().debug(f"OpenCV QR detection failed on variant: {e}")

        # OpenCV found nothing on this frame.
        self.consecutive_failures += 1

        # --- Method 2: Pyzbar fallback (only after sustained failure) ---
        if pyzbar is not None and self.consecutive_failures >= self.pyzbar_fallback_threshold:
            for candidate in candidates:
                try:
                    decoded_objects = pyzbar.decode(candidate)
                    for obj in decoded_objects:
                        if obj.data:
                            self.consecutive_failures = 0
                            return obj.data.decode('utf-8')
                except Exception as e:
                    self.get_logger().debug(f"Pyzbar decoding failed on variant: {e}")

        return None


def main(args=None):
    rclpy.init(args=args)
    node = QRDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
