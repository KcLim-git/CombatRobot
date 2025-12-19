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

        # Lightweight YOLOv5 nano model
        self.model = YOLO('yolov5n.pt')

        # Camera
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.get_logger().error("❌ Could not open camera.")
            raise SystemExit

        # 10 FPS
        self.timer = self.create_timer(1 / 10.0, self.detect_objects)
        self.get_logger().info("✅ VisionNode with tracking started...")

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

        frame_h, frame_w = frame.shape[:2]
        labels = []

        # ---------------- YOLO DETECTION ----------------
        results = self.model(frame, verbose=False)
        annotated = frame.copy()

        if len(results):
            annotated = results[0].plot()  # YOLO boxes + labels
            for c in results[0].boxes.cls:
                labels.append(self.model.names[int(c)])

        # ---------------- COLOR DETECTION: BLUE (MY ROBOT) ----------------
        blue_pixels, blue_mask = self.detect_blue(frame)
        if blue_pixels > 3000:    # tune threshold based on your lighting
            labels.append("my_robot")
            cv2.putText(
                annotated,
                "MY ROBOT",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 0, 0), 2
            )
            blue_overlay = cv2.applyColorMap(blue_mask, cv2.COLORMAP_OCEAN)
            annotated = cv2.addWeighted(annotated, 1.0, blue_overlay, 0.3, 0)

        # ---------------- COLOR DETECTION: RED (ENEMY ROBOT) ----------------
        red_pixels, red_mask = self.detect_red(frame)

        enemy_info = {
            "present": False
        }

        if red_pixels > 3000:   # tune threshold based on how strong your red marker is
            labels.append("enemy_robot")

            # Find largest red blob for tracking
            contours, _ = cv2.findContours(
                red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if contours:
                largest = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest)

                # Bounding box on annotated frame
                cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 0, 255), 2)

                # Center of enemy blob
                cx = x + w / 2.0
                cy = y + h / 2.0

                # Normalized coordinates: [-1, +1]
                cx_norm = (cx - frame_w / 2.0) / (frame_w / 2.0)
                cy_norm = (frame_h / 2.0 - cy) / (frame_h / 2.0)  # +1 = top, -1 = bottom

                # Size ratio relative to frame height
                size_norm = h / float(frame_h)

                # Direction band (LEFT / CENTER / RIGHT)
                if cx < frame_w * 0.4:
                    direction = "LEFT"
                elif cx > frame_w * 0.6:
                    direction = "RIGHT"
                else:
                    direction = "CENTER"

                # Distance band (rough, based on size)
                if size_norm > 0.45:
                    distance_band = "NEAR"
                elif size_norm > 0.25:
                    distance_band = "MID"
                else:
                    distance_band = "FAR"

                # Draw crosshair + info text
                cv2.circle(annotated, (int(cx), int(cy)), 6, (0, 0, 255), -1)
                info_text = f"ENEMY ROBOT [{direction}, {distance_band}]"
                cv2.putText(
                    annotated,
                    info_text,
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2
                )

                # Overlay heatmap of red_mask for visual effect
                red_overlay = cv2.applyColorMap(red_mask, cv2.COLORMAP_HOT)
                annotated = cv2.addWeighted(annotated, 1.0, red_overlay, 0.3, 0)

                # Fill enemy_info for ROS2
                enemy_info = {
                    "present": True,
                    "cx_norm": float(cx_norm),
                    "cy_norm": float(cy_norm),
                    "size_norm": float(size_norm),
                    "direction": direction,
                    "distance_band": distance_band
                }
            else:
                # No contour even though red_pixels > threshold
                cv2.putText(
                    annotated,
                    "ENEMY ROBOT (MASK ONLY)",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2
                )

        # ---------------- UNIQUE LABELS ----------------
        unique_labels = sorted(list(set(labels)))

        # ---------------- PUBLISH TO ROS2 ----------------
        t_vision = time.time()

        msg_dict = {
        "t_vision": t_vision,
        "labels": unique_labels,
        "enemy": enemy_info
    }

        msg = String()
        msg.data = json.dumps(msg_dict)
        self.publisher_.publish(msg)

        self.get_logger().info(f"📷 Detected: {unique_labels} | enemy={enemy_info}")

        # ---------------- SHOW FRAME ----------------
        cv2.imshow("Vision Output", annotated)
        cv2.waitKey(1)

    def destroy_node(self):
        self.cap.release()
        cv2.destroyAllWindows()
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
