import json
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

import paho.mqtt.client as mqtt


MQTT_BROKER_HOST = "localhost"
MQTT_BROKER_PORT = 1883
ROBOT_ID = "robot01"

MQTT_CMD_TOPIC = f"robot/commands/{ROBOT_ID}"
MQTT_STATUS_TOPIC = f"robot/status/{ROBOT_ID}"


class ControlBridgeNode(Node):
    def __init__(self):
        super().__init__("control_bridge_node")

        # ROS subscriber: /robot/command
        self.sub_cmd = self.create_subscription(
            String, "/robot/command", self.ros_command_callback, 10
        )

        # ROS publisher: /robot/status
        self.pub_status = self.create_publisher(String, "/robot/status", 10)

        # MQTT setup
        self.mqtt_client = mqtt.Client(client_id="ros2_bridge")
        self.mqtt_client.on_connect = self.on_mqtt_connect
        self.mqtt_client.on_message = self.on_mqtt_message

        self.get_logger().info(
            f"Connecting to MQTT broker at {MQTT_BROKER_HOST}:{MQTT_BROKER_PORT}..."
        )
        self.mqtt_client.connect(MQTT_BROKER_HOST, MQTT_BROKER_PORT, keepalive=60)
        self.mqtt_client.loop_start()

        self.get_logger().info("✅ ControlBridgeNode started.")

    # --- ROS /robot/command → MQTT publish ---
    def ros_command_callback(self, msg: String):
        """Forward command JSON from ROS to ESP32 via MQTT."""
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().error(f"Invalid JSON on /robot/command: {msg.data}")
            return

        if "cmd_id" not in data:
            data["cmd_id"] = str(int(time.time()))

        payload = json.dumps(data)
        result = self.mqtt_client.publish(MQTT_CMD_TOPIC, payload)

        if result.rc == mqtt.MQTT_ERR_SUCCESS:
            self.get_logger().info(
                f"📤 Sent to MQTT [{MQTT_CMD_TOPIC}]: {payload}"
            )
        else:
            self.get_logger().error(
                f"❌ Failed to publish to MQTT (rc={result.rc})"
            )

    # --- MQTT robot/status → ROS publish ---
    def on_mqtt_connect(self, client, userdata, flags, rc):
        if rc == 0:
            self.get_logger().info("✅ MQTT bridge connected.")
            client.subscribe(MQTT_STATUS_TOPIC)
            self.get_logger().info(f"Subscribed to: {MQTT_STATUS_TOPIC}")
        else:
            self.get_logger().error(f"MQTT connect failed (rc={rc})")

    def on_mqtt_message(self, client, userdata, msg):
        payload = msg.payload.decode("utf-8")
        self.get_logger().info(
            f"📥 MQTT [{msg.topic}] → ROS /robot/status: {payload}"
        )
        ros_msg = String()
        ros_msg.data = payload
        self.pub_status.publish(ros_msg)

    def destroy_node(self):
        self.get_logger().info("Shutting down MQTT...")
        self.mqtt_client.loop_stop()
        self.mqtt_client.disconnect()
        super().destroy_node()


# ---------- MAIN MUST BE AT THE BOTTOM ----------
def main(args=None):
    rclpy.init(args=args)
    node = ControlBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

