import rclpy
from rclpy.node import Node
from std_msgs.msg import String

import json
import time
import subprocess


ALLOWED_ACTIONS = [
    "MOVE_FORWARD",
    "MOVE_BACKWARD",
    "ROTATE_LEFT",
    "ROTATE_RIGHT",
    "STOP",
]


class DecisionNode(Node):
    def __init__(self):
        super().__init__('decision_node')

        # Subscribe to vision detections
        self.subscription = self.create_subscription(
            String,
            '/vision/detections',
            self.vision_callback,
            10
        )

        # Publish high-level commands to control bridge
        self.publisher = self.create_publisher(
            String,
            '/robot/command',
            10
        )

        # Parameters / config
        self.declare_parameter('use_llm', True)
        self.declare_parameter('llm_model', 'llama3.1:8b')
        self.declare_parameter('default_speed', 0.6)
        self.declare_parameter('default_duration', 1.0)

        self.use_llm = self.get_parameter('use_llm').get_parameter_value().bool_value
        self.llm_model = self.get_parameter('llm_model').get_parameter_value().string_value
        self.default_speed = self.get_parameter('default_speed').get_parameter_value().double_value
        self.default_duration = self.get_parameter('default_duration').get_parameter_value().double_value

        self.get_logger().info(
            f"🧠 DecisionNode started | use_llm={self.use_llm} | model={self.llm_model}"
        )

    # -------------------------------------------------------
    # Helper: publish one action as JSON to /robot/command
    # -------------------------------------------------------
    def publish_action(self, action: str, source: str = "REFLEX"):
        if action not in ALLOWED_ACTIONS:
            self.get_logger().warn(f"⚠️ Invalid action '{action}', forcing STOP.")
            action = "STOP"

        cmd = {
            "cmd_id": f"CMD_{int(time.time())}",
            "action": action,
            "params": {
                "speed": float(self.default_speed),
                "duration": float(self.default_duration),
            }
        }

        msg = String()
        msg.data = json.dumps(cmd)

        self.publisher.publish(msg)
        self.get_logger().info(f"[{source}] → {action} | payload={msg.data}")

    # -------------------------------------------------------
    # Helper: call Ollama LLM (GPU) to refine decision
    # -------------------------------------------------------
    def query_llm(self, labels, enemy, reflex_action):
        """
        Ask LLM to possibly refine the reflex_action.
        We still *constrain* output to ALLOWED_ACTIONS.
        """
        prompt = f"""
You are the control system for a real-time combat robot.
You MUST output exactly ONE token from this list:
MOVE_FORWARD, MOVE_BACKWARD, ROTATE_LEFT, ROTATE_RIGHT, STOP

Context:
- Detected labels: {labels}
- Enemy info: {enemy}
- Reflex suggested action: {reflex_action}

Rules:
1. If a person (human) is detected in labels, STOP is always correct.
2. If enemy.present is false, scanning/ROTATE_RIGHT is usually appropriate.
3. If enemy.direction is LEFT, ROTATE_LEFT is usually appropriate.
4. If enemy.direction is RIGHT, ROTATE_RIGHT is usually appropriate.
5. If enemy.direction is CENTER and enemy.distance_band in ['FAR', 'MID'],
   MOVE_FORWARD is usually appropriate to attack.
6. If enemy.direction is CENTER and enemy.distance_band == 'NEAR',
   either STOP or MOVE_BACKWARD is appropriate.
7. You may keep the reflex_suggested action, or change it if you think
   a slightly better choice exists, but you MUST choose exactly ONE token
   from the allowed list and output ONLY that token, nothing else.
"""

        try:
            result = subprocess.run(
                ["ollama", "run", self.llm_model],
                input=prompt,
                text=True,
                capture_output=True,
                timeout=5.0
            )
            raw = result.stdout.strip()
            action = raw.split()[0].strip().upper()

            if action not in ALLOWED_ACTIONS:
                self.get_logger().warn(
                    f"LLM returned invalid token '{raw}', falling back to reflex '{reflex_action}'"
                )
                return reflex_action

            return action

        except Exception as e:
            self.get_logger().error(f"LLM error: {e}, falling back to reflex '{reflex_action}'")
            return reflex_action

    # -------------------------------------------------------
    # Main callback: process /vision/detections messages
    # -------------------------------------------------------
    def vision_callback(self, msg: String):
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError as e:
            self.get_logger().error(f"JSON parse error in /vision/detections: {e}")
            return

        labels = data.get("labels", [])
        enemy = data.get("enemy", {"present": False})

        enemy_present = enemy.get("present", False)
        enemy_direction = enemy.get("direction", "UNKNOWN")
        enemy_distance = enemy.get("distance_band", "UNKNOWN")
        enemy_cx = enemy.get("cx_norm", 0.0)
        enemy_size = enemy.get("size_norm", 0.0)

        self.get_logger().info(
            f"📥 Vision: labels={labels} | enemy_present={enemy_present} "
            f"dir={enemy_direction} dist={enemy_distance} cx={enemy_cx:.2f} size={enemy_size:.2f}"
        )

        # ---------------- REFLEX LOGIC (fast, deterministic) ----------------

        # 1) Safety first: person detected → STOP
        if "person" in labels:
            self.publish_action("STOP", source="REFLEX (PERSON SAFETY)")
            return

        # 2) No enemy detected → scan
        if not enemy_present:
            self.publish_action("ROTATE_RIGHT", source="REFLEX (SCAN)")
            return

        # 3) Enemy is clearly to the left or right
        if enemy_direction == "LEFT":
            self.publish_action("ROTATE_LEFT", source="REFLEX (TRACK LEFT)")
            return

        if enemy_direction == "RIGHT":
            self.publish_action("ROTATE_RIGHT", source="REFLEX (TRACK RIGHT)")
            return

        # 4) Enemy roughly centered → use distance to decide
        reflex_action = "STOP"

        if enemy_direction == "CENTER":
            if enemy_distance in ("FAR", "MID"):
                reflex_action = "MOVE_FORWARD"   # charge
            elif enemy_distance == "NEAR":
                # Up close: either STOP or slight retreat
                reflex_action = "STOP"
            else:
                # Unknown distance → be safe
                reflex_action = "STOP"
        else:
            # Unknown direction → slow scan
            reflex_action = "ROTATE_RIGHT"

        # ---------------- OPTIONAL LLM REFINEMENT ----------------
        final_action = reflex_action
        if self.use_llm:
            final_action = self.query_llm(labels, enemy, reflex_action)
            source = "LLM+REFLEX" if final_action != reflex_action else "REFLEX (LLM AGREED)"
        else:
            source = "REFLEX (LLM OFF)"

        self.publish_action(final_action, source=source)


# -------------------------------------------------------
# MAIN
# -------------------------------------------------------
def main(args=None):
    rclpy.init(args=args)
    node = DecisionNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
