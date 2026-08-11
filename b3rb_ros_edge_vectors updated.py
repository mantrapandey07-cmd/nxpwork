import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
import numpy as np
import cv2
import math
from synapse_msgs.msg import EdgeVectors

QOS_PROFILE_DEFAULT = 10
PI = math.pi

RED_COLOR = (0, 0, 255)
BLUE_COLOR = (255, 0, 0)
GREEN_COLOR = (0, 255, 0)

# HINT: You can adjust what percentage of the image from the bottom is analyzed.
# Lower portions are closer to the buggy, while upper portions see further ahead.
VECTOR_IMAGE_HEIGHT_PERCENTAGE = 0.35
VECTOR_MAGNITUDE_MINIMUM = 2.25

# --- New preprocessing tunables ---
GAUSSIAN_BLUR_KERNEL = (5, 5)
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8)
ADAPTIVE_THRESH_BLOCK_SIZE = 31   # must be odd; bigger = smoother local threshold
ADAPTIVE_THRESH_C = 8             # subtracted from the local mean; higher = stricter "blackness"
MORPH_KERNEL_SIZE = (3, 3)
USE_ADAPTIVE_THRESHOLD = True     # False falls back to your original fixed-threshold behavior
FIXED_THRESHOLD_BLACK = 25


class EdgeVectorsPublisher(Node):
    """
    ROS 2 Node that processes raw camera images to detect the lane edges (left/right bounds).
    It publishes the detected boundaries as synapse_msgs/EdgeVectors.
    """
    def __init__(self):
        super().__init__('edge_vectors_publisher')

        # Subscription for camera images.
        self.subscription_camera = self.create_subscription(
            CompressedImage,
            '/camera/image_raw/compressed',
            self.camera_image_callback,
            QOS_PROFILE_DEFAULT)

        # Publisher for edge vectors.
        self.publisher_edge_vectors = self.create_publisher(
            EdgeVectors,
            '/edge_vectors',
            QOS_PROFILE_DEFAULT)

        # Publisher for thresh image (for debugging thresholding/segmentation).
        self.publisher_thresh_image = self.create_publisher(
            CompressedImage,
            "/debug_images/thresh_image",
            QOS_PROFILE_DEFAULT)

        # Publisher for vector image (for debugging vector drawing).
        self.publisher_vector_image = self.create_publisher(
            CompressedImage,
            "/debug_images/vector_image",
            QOS_PROFILE_DEFAULT)

        self.image_height = 0
        self.image_width = 0
        self.lower_image_height = 0
        self.upper_image_height = 0

        # CLAHE object is stateful/expensive to build repeatedly, so create it once.
        self.clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID_SIZE)

        # Morphology kernel reused every frame.
        self.morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, MORPH_KERNEL_SIZE)

    def publish_debug_image(self, publisher, image):
        """Helper function to publish OpenCV debug images to ROS topics."""
        message = CompressedImage()
        _, encoded_data = cv2.imencode('.jpg', image)
        message.format = "jpeg"
        message.data = encoded_data.tobytes()
        publisher.publish(message)

    def get_vector_angle_in_radians(self, vector):
        """Calculates the slope angle of a vector in radians."""
        if ((vector[0][0] - vector[1][0]) == 0):  # Prevent division by zero
            theta = PI / 2
        else:
            slope = (vector[1][1] - vector[0][1]) / (vector[0][0] - vector[1][0])
            theta = math.atan(slope)
        return theta

    def build_binary_mask(self, image_bgr):
        """
        Turns a raw BGR crop into a clean binary mask isolating the dark lane
        boundary stripes, robust to uneven lighting.

        Pipeline:
          1. LAB color space, take L (lightness) channel -- more stable than
             plain grayscale under colored ambient light / shadows.
          2. CLAHE (adaptive histogram equalization) to flatten out lighting
             gradients across the frame (e.g. one side in shadow).
          3. Gaussian blur to suppress camera sensor noise before thresholding.
          4. Adaptive thresholding (per-region threshold) instead of one global
             cutoff, so it still finds dark stripes even if part of the image
             is brighter/darker than another.
          5. Morphological opening (erode->dilate) to remove small speckle
             noise, followed by closing (dilate->erode) to bridge small gaps
             in the stripe outline so contours don't fragment.
        """
        lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
        l_channel, _, _ = cv2.split(lab)

        l_eq = self.clahe.apply(l_channel)

        blurred = cv2.GaussianBlur(l_eq, GAUSSIAN_BLUR_KERNEL, 0)

        if USE_ADAPTIVE_THRESHOLD:
            block_size = ADAPTIVE_THRESH_BLOCK_SIZE
            if block_size % 2 == 0:
                block_size += 1  # cv2 requires an odd block size
            thresh = cv2.adaptiveThreshold(
                blurred, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,
                block_size,
                ADAPTIVE_THRESH_C)
        else:
            thresh = cv2.threshold(blurred, FIXED_THRESHOLD_BLACK, 255, cv2.THRESH_BINARY_INV)[1]

        # Opening removes small isolated noise blobs.
        opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, self.morph_kernel)
        # Closing fills small gaps so a stripe's contour stays one connected shape.
        cleaned = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, self.morph_kernel)

        return cleaned

    def compute_vectors_from_image(self, image, thresh):
        """
        Analyzes the binary threshold image and extracts left and right lane edge vectors.
        This uses basic contour finding and coordinates calculations.

        You can optimize this algorithm or implement alternative methods here.
        """
        # Find contours around black edge stripes detected in the binary threshold image.
        contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]

        vectors = []
        for i in range(len(contours)):
            coordinates = contours[i][:, 0, :]

            # Get coordinates representing the boundaries of the contour
            min_y_value = np.min(coordinates[:, 1])
            max_y_value = np.max(coordinates[:, 1])

            min_y_coords = np.array(coordinates[coordinates[:, 1] == min_y_value])
            max_y_coords = np.array(coordinates[coordinates[:, 1] == max_y_value])

            min_y_coord = min_y_coords[0]
            max_y_coord = max_y_coords[0]

            # Calculate contour vector magnitude
            magnitude = np.linalg.norm(min_y_coord - max_y_coord)
            if (magnitude > VECTOR_MAGNITUDE_MINIMUM):
                # Calculate distance from the camera center at the bottom of the crop
                rover_point = [self.image_width / 2, self.lower_image_height]
                middle_point = (min_y_coord + max_y_coord) / 2
                distance = np.linalg.norm(middle_point - rover_point)

                # Correct point coordinates based on vector slope angle
                angle = self.get_vector_angle_in_radians([min_y_coord, max_y_coord])
                if angle > 0:
                    min_y_coord[0] = np.max(min_y_coords[:, 0])
                else:
                    max_y_coord[0] = np.max(max_y_coords[:, 0])

                # Store vectors with distance metadata for sorting
                vectors.append([list(min_y_coord), list(max_y_coord), distance])

            # Draw all detected raw vectors in blue on the debug image
            cv2.line(image, tuple(min_y_coord), tuple(max_y_coord), BLUE_COLOR, 2)

        return vectors, image

    def process_image_for_edge_vectors(self, image):
        """
        Applies preprocessing (LAB + CLAHE + blur + adaptive threshold + morphology)
        and extracts lane vectors.
        """
        self.image_height, self.image_width, _ = image.shape
        self.lower_image_height = int(self.image_height * VECTOR_IMAGE_HEIGHT_PERCENTAGE)
        self.upper_image_height = int(self.image_height - self.lower_image_height)

        # 1. Crop the image to focus on the lower section close to the buggy FIRST,
        #    so CLAHE/adaptive-threshold statistics are computed on the region that
        #    actually matters instead of being skewed by sky/background above it.
        image_cropped = image[self.image_height - self.lower_image_height:].copy()

        # 2. Build a clean binary mask isolating the dark lane boundary stripes.
        thresh_cropped = self.build_binary_mask(image_cropped)

        # 3. Compute vectors from the binary image contours
        vectors, debug_img = self.compute_vectors_from_image(image_cropped, thresh_cropped)

        # 4. Sort vectors based on distance from buggy (we prioritize vectors closer to us)
        vectors = sorted(vectors, key=lambda x: x[2])

        # 5. Split vectors based on left/right halves of the image
        half_width = self.image_width / 2
        vectors_left = [v for v in vectors if ((v[0][0] + v[1][0]) / 2) < half_width]
        vectors_right = [v for v in vectors if ((v[0][0] + v[1][0]) / 2) >= half_width]

        final_vectors = []
        # Select the closest vector from each side (left/right)
        for side_vectors in [vectors_left, vectors_right]:
            if len(side_vectors) > 0:
                best_vector = side_vectors[0]
                # Draw the selected key lane vector in green on the debug image
                cv2.line(debug_img, tuple(best_vector[0]), tuple(best_vector[1]), GREEN_COLOR, 2)

                # Transform coordinates back to the original uncropped image space
                best_vector[0][1] += self.upper_image_height
                best_vector[1][1] += self.upper_image_height
                final_vectors.append(best_vector[:2])

        # Publish visual debugging images (viewable in tools like Foxglove)
        self.publish_debug_image(self.publisher_thresh_image, thresh_cropped)
        self.publish_debug_image(self.publisher_vector_image, debug_img)

        return final_vectors

    def camera_image_callback(self, message):
        """Processes incoming camera frames and publishes detected EdgeVectors."""
        # Convert compressed image message to OpenCV format
        np_arr = np.frombuffer(message.data, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        vectors = self.process_image_for_edge_vectors(image)

        # Construct and publish the ROS 2 EdgeVectors message
        vectors_message = EdgeVectors()
        vectors_message.image_height = image.shape[0]
        vectors_message.image_width = image.shape[1]
        vectors_message.vector_count = 0

        # Vector 1 (usually representing Left boundary)
        if len(vectors) > 0:
            vectors_message.vector_1[0].x = float(vectors[0][0][0])
            vectors_message.vector_1[0].y = float(vectors[0][0][1])
            vectors_message.vector_1[1].x = float(vectors[0][1][0])
            vectors_message.vector_1[1].y = float(vectors[0][1][1])
            vectors_message.vector_count += 1

        # Vector 2 (usually representing Right boundary)
        if len(vectors) > 1:
            vectors_message.vector_2[0].x = float(vectors[1][0][0])
            vectors_message.vector_2[0].y = float(vectors[1][0][1])
            vectors_message.vector_2[1].x = float(vectors[1][1][0])
            vectors_message.vector_2[1].y = float(vectors[1][1][1])
            vectors_message.vector_count += 1

        self.publisher_edge_vectors.publish(vectors_message)


def main(args=None):
    rclpy.init(args=args)
    node = EdgeVectorsPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
