import paho.mqtt.client as mqtt
import time
import json
import csv
import os

BROKER_HOST = "localhost"
BROKER_PORT = 1883
TOPIC =                 "robot/jitter/robot01"


CSV_PATH = os.path.expanduser("~/jitter_log.csv")


def ensure_csv_header():
    if not os.path.exists(CSV_PATH):
        with open(CSV_PATH, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "t_host",
                "t_device_ms",
                "dt_prev_s",
                "cmd_id",
                "event"
            ])


def on_connect(client, userdata, flags, rc, properties=None):
    print("Connected to MQTT with rc =", rc)
    client.subscribe(TOPIC)
    print("Subscribed to", TOPIC)


def on_message(client, userdata, msg):
    t_host = time.time()
    try:
        payload = msg.payload.decode("utf-8")
        data = json.loads(payload)
    except Exception as e:
        print("Parse error:", e)
        return

    t_device_ms = data.get("t_device_ms", None)
    dt_prev_s = data.get("dt_prev_s", None)
    cmd_id = data.get("cmd_id", "")
    event = data.get("event", "")

    print(f"[{time.strftime('%H:%M:%S')}] payload={payload}")

    try:
        with open(CSV_PATH, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                t_host,
                t_device_ms,
                dt_prev_s,
                cmd_id,
                event
            ])
    except Exception as e:
        print("CSV write error:", e)


def main():
    ensure_csv_header()
    client = mqtt.Client(client_id="esp32_jitter_logger")
    client.on_connect = on_connect
    client.on_message = on_message

    print(f"Subscribing to ESP32 logs on {BROKER_HOST}:{BROKER_PORT}, topic {TOPIC}")
    client.connect(BROKER_HOST, BROKER_PORT, keepalive=60)
    client.loop_forever()


if __name__ == "__main__":
    main()
