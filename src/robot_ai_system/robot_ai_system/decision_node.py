import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import json
import time
import subprocess

class DecisionNode(Node):
    def __init__(self):
        super().__init__("decision_node")
        self.subscription = self.create_subscription(
            String,
            "/vision/detections",
            self.vision_callback,
            10,
        )

        self.publisher = self.create_publisher(
            String,
            "/robot/command",
            10
        )

        self.get_logger().info("🤖 DecisionNode started.")

    def call_llm(self, labels):
        prompt = f"""
You are a control assistant for a combat robot. 
Given the detected objects: {labels}

Output EXACTLY ONE of these tokens:
MOVE_FORWARD, MOVE_BACKWARD, ROTATE_LEFT, ROTATE_RIGHT, STOP.

Combat Rules:
- If "person" is detected → STOP (safety override).
- If "enemy_robot" is detected → MOVE_FORWARD to attack.
- If robot-like objects such as "car", "toy", "rc_car", "vehicle" appear → ROTATE_LEFT to align.
- If only obstacles like "chair", "table", "wall", "bottle" appear → ROTATE_RIGHT to reposition.
- If NO objects detected → ROTATE_RIGHT to scan and search for enemy.
- If unsure → ROTATE_RIGHT.


Reply with ONLY the token, nothing else.
""".strip()

        result = subprocess.run(
            ["ollama", "run", "llama3.2"],
            input=prompt.encode("utf-8"),
            stdout=subprocess.PIPE
        )

        action = result.stdout.decode("utf-8").strip()
        return action

    def vision_callback(self, msg):
        data = json.loads(msg.data)
        labels = data["labels"]

        self.get_logger().info(f"🧠 Detections received: {labels}")
        # Ignore our own robot so we don't target ourselves
        if "my_robot" in labels:
            labels.remove("my_robot")
            self.get_logger().info("🔵 Ignoring our own robot in decision logic.")

        # Safety override
        if "person" in labels:
            token = "STOP"

        # LLM decision
        token = self.call_llm(labels)
        self.get_logger().info(f"🧠 LLM Action: {token}")

        # Map token → command JSON
        cmd = {
            "cmd_id": f"LLM_{int(time.time())}",
            "action": token,
            "params": {
                "speed": 0.6,
                "duration": 1.0
            }
        }

        msg_out = String()
        msg_out.data = json.dumps(cmd)

        # Publish to /robot/command
        self.publisher.publish(msg_out)

        self.get_logger().info(f"📤 Published action: {msg_out.data}")


def main(args=None):
    rclpy.init(args=args)
    node = DecisionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
