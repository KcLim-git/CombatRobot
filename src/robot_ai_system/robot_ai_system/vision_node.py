import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from ultralytics import YOLO
import cv2
import json
import time
import numpy as np

class VisionNode(Node):
    def __init__(self):
        super().__init__('vision_node')
        self.publisher_ = self.create_publisher(String, '/vision/detections', 10)

        self.model = YOLO('yolov5n.pt')

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.get_logger().error("❌ Could not open camera.")
            raise SystemExit

        self.timer = self.create_timer(1/10.0, self.detect_objects)  # 10 FPS

    # -------------------- COLOR DETECTORS --------------------
    def detect_blue(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_blue = np.array([90, 80, 40])
        upper_blue = np.array([130, 255, 255])
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
        count = cv2.countNonZero(mask)
        return count, mask

    def detect_red(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_red1 = np.array([0, 120, 70])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 120, 70])
        upper_red2 = np.array([180, 255, 255])
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)
        count = cv2.countNonZero(mask)
        return count, mask

    # -------------------- MAIN DETECTION --------------------
    def detect_objects(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warn("⚠️ Failed to read frame")
            return

        labels = []

        # ---------------- YOLO DETECTION ----------------
        results = self.model(frame, verbose=False)
        annotated = frame.copy()

        if len(results):
            annotated = results[0].plot()

            for c in results[0].boxes.cls:
                labels.append(self.model.names[int(c)])

        # ---------------- COLOR DETECTION: BLUE ----------------
        blue_pixels, blue_mask = self.detect_blue(frame)
        if blue_pixels > 3000:    # adjust threshold
            labels.append("my_robot")
            cv2.putText(annotated, "MY ROBOT", (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        1.1, (255, 0, 0), 3)
            blue_overlay = cv2.applyColorMap(blue_mask, cv2.COLORMAP_OCEAN)
            annotated = cv2.addWeighted(annotated, 1.0, blue_overlay, 0.4, 0)

        # ---------------- COLOR DETECTION: RED ----------------
        red_pixels, red_mask = self.detect_red(frame)
        if red_pixels > 3000:
            labels.append("enemy_robot")
            cv2.putText(annotated, "ENEMY ROBOT", (20, 80), cv2.FONT_HERSHEY_SIMPLEX,
                        1.1, (0, 0, 255), 3)
            red_overlay = cv2.applyColorMap(red_mask, cv2.COLORMAP_HOT)
            annotated = cv2.addWeighted(annotated, 1.0, red_overlay, 0.4, 0)

        # ---------------- UNIQUE LABELS ----------------
        unique_labels = sorted(list(set(labels)))

        # ROS2 publish
        msg = String()
        msg.data = json.dumps({
            "timestamp": time.time(),
            "labels": unique_labels
        })
        self.publisher_.publish(msg)

        # Logging
        self.get_logger().info(f"📷 Detected: {unique_labels}")

        # ---------------- SHOW FRAME ----------------
        cv2.imshow("Vision Output", annotated)
        cv2.waitKey(1)

    def destroy_node(self):
        self.cap.release()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = VisionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
