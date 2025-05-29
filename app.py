import os
import uuid
import sqlite3
from datetime import datetime
from pathlib import Path
from time import time
import requests
from flask import Flask, request, jsonify
from flask import redirect, url_for, session
import torch
from PIL import Image
from torchvision import transforms
import torchvision
from utils import *

"""
Early-Warn WhatsApp Bot — minimal end-to-end example (r2)
--------------------------------------------------------
Adds **service-health endpoints** so you (or a load balancer / uptime robot)
can check that the app and database are alive:
    • `GET /health`  → plain "OK" 200
    • `GET /status` → JSON uptime + event counters
All previous image-classification and reply logic is unchanged.
"""

# ---------------------------------------------------------------------------
# Configuration & tiny persistent store
# ---------------------------------------------------------------------------
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

VERIFY_TOKEN = os.getenv("VERIFY_TOKEN")
ACCESS_TOKEN = os.getenv("ACCESS_TOKEN")
PHONE_NUMBER_ID = os.getenv("PHONE_NUMBER_ID")

upload_dir = Path("static/uploads")
upload_dir.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists

DB_PATH = Path("events.db")
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
c = conn.cursor()
c.execute(
    """CREATE TABLE IF NOT EXISTS events (
           id TEXT PRIMARY KEY,
           chat_id TEXT,
           label TEXT,
           confidence REAL,
           ts DATETIME DEFAULT CURRENT_TIMESTAMP
       )"""
)
conn.commit()

# Track uptime
START_TIME = time()

# ---------------------------------------------------------------------------
# Load model (if PyTorch is available – otherwise stub-predict healthy)
# ---------------------------------------------------------------------------
# Optional: Tiny vision model (replace with your own trained/quantised weights)
if not torch.cuda.is_available():
    print("Warning: No GPU detected, running on CPU. Performance may be slow.")
    torch.set_num_threads(1)  # Limit to single thread for CPU inference


## load quantized weights if available to speed up inference
# weights_file = Path(__file__).with_name("best_model.pth")
# model = get_squeezenet_model(weights_file)
model = PlantDiseaseModel(num_classes = 38)
model.load_state_dict(torch.load('best_model.pth', weights_only=True, map_location="cpu"))
print(model)


# ---------------------------------------------------------------------------
# WhatsApp helper — send text reply
# ---------------------------------------------------------------------------

def send_message(chat_id: str, text: str):
    url = f"https://graph.facebook.com/v19.0/{PHONE_NUMBER_ID}/messages"
    headers = {"Authorization": f"Bearer {ACCESS_TOKEN}", "Content-Type": "application/json"}
    payload = {
        "messaging_product": "whatsapp",
        "to": chat_id,
        "type": "text",
        "text": {"preview_url": False, "body": text},
    }
    requests.post(url, headers=headers, json=payload, timeout=10)


# ---------------------------------------------------------------------------
# Flask application
# ---------------------------------------------------------------------------
app = Flask(__name__)
app.secret_key = os.urandom(24)

@app.route("/health", methods=["GET"])
def health():
    """Lightweight health probe for load balancers."""
    return "OK", 200


@app.route("/status", methods=["GET"])
def status():
    """Return uptime and aggregate event counts as JSON."""
    total, diseased, healthy = (0, 0, 0)
    row = c.execute(
        "SELECT COUNT(*), SUM(label='diseased'), SUM(label='healthy') FROM events"
    ).fetchone()
    if row:
        total, diseased, healthy = (r or 0 for r in row)
    uptime = int(time() - START_TIME)
    return jsonify(
        {
            "status": "ok",
            "uptime_seconds": uptime,
            "events_total": total,
            "events_healthy": healthy,
            "events_diseased": diseased,
        }
    )


@app.route("/webhook", methods=["GET"])
def verify():
    mode = request.args.get("hub.mode")
    token = request.args.get("hub.verify_token")
    challenge = request.args.get("hub.challenge")
    if mode == "subscribe" and token == VERIFY_TOKEN:
        return challenge, 200
    return "Forbidden", 403


@app.route("/webhook", methods=["POST"])
def incoming():
    data = request.get_json(force=True)
    if data.get("object") != "whatsapp_business_account":
        return "ignored", 200

    for entry in data.get("entry", []):
        for change in entry.get("changes", []):
            for msg in change.get("value", {}).get("messages", []):
                chat_id = msg.get("from")
                mtype = msg.get("type")

                # ------------------------------------------------------------
                # 1. Image ➜ classify ➜ personalised reply
                # ------------------------------------------------------------
                if mtype == "image":
                    media_id = msg["image"]["id"]
                    meta_resp = requests.get(
                        f"https://graph.facebook.com/v19.0/{media_id}",
                        headers={"Authorization": f"Bearer {ACCESS_TOKEN}"},
                        timeout=10,
                    ).json()
                    media_url = meta_resp.get("url")
                    img_data = requests.get(media_url, headers={"Authorization": f"Bearer {ACCESS_TOKEN}"}, timeout=10).content
                    tmp_path = upload_dir / f"{uuid.uuid4()}.jpg"  # Corrected path
                    tmp_path.write_bytes(img_data)

                    label, conf = classify(model, tmp_path)
                    c.execute(
                        "INSERT INTO events VALUES (?,?,?,?,?)",
                        (str(uuid.uuid4()), chat_id, label, conf, datetime.utcnow()),
                    )
                    conn.commit()

                    if label == "healthy":
                        send_message(chat_id, f"✅ Your plant looks healthy (confidence {conf:.0%}). Keep monitoring!")
                    else:
                        send_message(
                            chat_id,
                            f"⚠️ Possible disease detected (confidence {conf:.0%}). Recommend copper fungicide within 24 h and remove infected leaves.",
                        )

                # ------------------------------------------------------------
                # 2. Text commands (help, stats)
                # ------------------------------------------------------------
                elif mtype == "text":
                    body = msg["text"]["body"].lower()
                    if "help" in body:
                        send_message(chat_id, "Send a clear photo of a single leaf. I'll tell you if it looks healthy or diseased, and what to do next. Send 'stats' to see your history.")
                    elif "stats" in body:
                        row = c.execute(
                            "SELECT COUNT(*), SUM(label='diseased') FROM events WHERE chat_id=?",
                            (chat_id,),
                        ).fetchone()
                        total, diseased = row or (0, 0)
                        send_message(chat_id, f"You have sent {total} images so far; {diseased or 0} were flagged diseased.")
                    else:
                        send_message(chat_id, "Unknown command. Send 'help' for instructions.")

    return "ok", 200


@app.route("/favicon.ico", methods=["GET"])
def favicon():
    """Serve a default favicon to prevent 404 errors."""
    return "", 204  # No content response


@app.route("/upload", methods=["GET", "POST"])
def upload():
    """Simple interface to upload an image and test classification."""
    result_html = ""
    if request.method == "POST":
        if "file" not in request.files:
            return "No file uploaded", 400
        file = request.files["file"]
        if file.filename == "":
            return "No selected file", 400
        tmp_path = upload_dir / f"{uuid.uuid4()}.jpg"
        file.save(tmp_path)
        # Perform classification
        label, conf = classify(model, tmp_path)

        # Store results in session
        session['result'] = {
            'image': tmp_path.name,
            'label': label,
            'confidence': conf
        }
        return redirect(url_for('upload'))

    # Handle GET request
    result_html = ""
    if 'result' in session:
        result = session['result']
        result_html = f"""
        <h2>Prediction Result</h2>
        <img src="/static/uploads/{result['image']}" alt="Uploaded Image" style="max-width: 300px; max-height: 300px;">
        <p><strong>Label:</strong> {result['label']}</p>
        <p><strong>Confidence:</strong> {result['confidence']:.0%}</p>
        """

    return f"""
    <!doctype html>
    <title>Upload Image</title>
    <h1>Upload an image to test classification</h1>
    <form method="POST" enctype="multipart/form-data">
        <input type="file" name="file" accept="image/*">
        <input type="submit" value="Upload">
    </form>
    {result_html}
    """


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
