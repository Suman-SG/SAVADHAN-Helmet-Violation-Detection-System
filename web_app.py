print("[DEBUG] web_app.py: Starting imports...", flush=True)

import json
import os
import sqlite3
import threading
from datetime import datetime
from pathlib import Path

print("[DEBUG] web_app.py: Standard lib imports done", flush=True)

import cv2
import numpy as np
import pandas as pd
print("[DEBUG] web_app.py: CV2/numpy/pandas done", flush=True)

from flask import (
    Flask,
    abort,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    send_from_directory,
    session,
    url_for,
)
from werkzeug.utils import secure_filename
print("[DEBUG] web_app.py: Flask imports done", flush=True)

import config
print("[DEBUG] web_app.py: config imported", flush=True)

from database import TrafficDatabase
print("[DEBUG] web_app.py: database imported", flush=True)

from fine_system import FineSystem
print("[DEBUG] web_app.py: fine_system imported", flush=True)

BASE_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = BASE_DIR / "outputs"
ANNOTATED_DIR = OUTPUTS_DIR / "annotated"
EVIDENCE_DIR = OUTPUTS_DIR / "evidence"
UPLOADS_DIR = OUTPUTS_DIR / "uploads"
INVOICES_DIR = BASE_DIR / "invoices"
SETTINGS_PATH = OUTPUTS_DIR / "ui_settings.json"
DB_PATH = Path(config.DATABASE_PATH)
print("[DEBUG] web_app.py: Paths set", flush=True)

app = Flask(__name__, template_folder="templates", static_folder="static")
print("[DEBUG] web_app.py: Flask app created", flush=True)

app.secret_key = os.getenv("SECRET_KEY", "dev-secret-change-me")
app.config["MAX_CONTENT_LENGTH"] = int(os.getenv("MAX_UPLOAD_MB", "512")) * 1024 * 1024
print("[DEBUG] web_app.py: Flask config set", flush=True)

print("[DEBUG] web_app.py: Getting ADMIN_USERNAME/PASSWORD from env", flush=True)
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "2006")
print("[DEBUG] web_app.py: Forcing FAST_MODE=True to avoid heavy startup", flush=True)
try:
    config.FAST_MODE = True
except Exception:
    pass

@app.route('/health')
def _health():
    return jsonify({"status": "ok"})

@app.errorhandler(Exception)
def _handle_exception(e):
    # Log unexpected exceptions to container logs
    try:
        import traceback, sys
        traceback.print_exc()
    except Exception:
        pass
    return jsonify({"success": False, "error": str(e)}), 500
print("[DEBUG] web_app.py: ADMIN credentials set", flush=True)

print("[DEBUG] web_app.py: Creating DEFAULT_SETTINGS dict", flush=True)
DEFAULT_SETTINGS = {
    "helmet_conf": float(config.HELMET_CONF),
    "plate_conf": float(config.PLATE_CONF),
    "fine_amount": float(config.FINE_AMOUNT),
    "send_email": bool(config.SEND_EMAIL),
}
print("[DEBUG] web_app.py: DEFAULT_SETTINGS created", flush=True)

print("[DEBUG] web_app.py: Initializing global variables", flush=True)
_PIPELINE = None
_PIPELINE_LOCK = threading.Lock()
_PROCESS_LOCK = threading.Lock()
_DB = None
_DB_LOCK = threading.Lock()
_INIT_THREAD = None
_PIPELINE_READY = threading.Event()
PIPELINE_INIT_TIMEOUT = int(os.getenv("PIPELINE_INIT_TIMEOUT", "180"))
print("[DEBUG] web_app.py: Global variables initialized", flush=True)


def ensure_directories() -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    ANNOTATED_DIR.mkdir(parents=True, exist_ok=True)
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    INVOICES_DIR.mkdir(parents=True, exist_ok=True)


def load_ui_settings() -> dict:
    ensure_directories()
    if SETTINGS_PATH.exists():
        try:
            with open(SETTINGS_PATH, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            merged = dict(DEFAULT_SETTINGS)
            merged.update(payload)
            return merged
        except Exception:
            return dict(DEFAULT_SETTINGS)
    return dict(DEFAULT_SETTINGS)


def save_ui_settings(settings: dict) -> None:
    ensure_directories()
    with open(SETTINGS_PATH, "w", encoding="utf-8") as handle:
        json.dump(settings, handle, indent=2)


def apply_runtime_settings(settings: dict) -> None:
    config.HELMET_CONF = float(settings["helmet_conf"])
    config.PLATE_CONF = float(settings["plate_conf"])
    config.FINE_AMOUNT = float(settings["fine_amount"])
    config.SEND_EMAIL = bool(settings["send_email"])


def reset_pipeline() -> None:
    global _PIPELINE
    with _PIPELINE_LOCK:
        _PIPELINE = None
        global _INIT_THREAD, _PIPELINE_READY
        _INIT_THREAD = None
        _PIPELINE_READY.clear()


def _init_pipeline_background():
    from main_pipeline import ViolationPipeline

    global _PIPELINE
    try:
        _PIPELINE = ViolationPipeline(suppress_emails=False)
    except Exception as exc:
        _PIPELINE = exc
    finally:
        _PIPELINE_READY.set()


def _ensure_pipeline_thread() -> None:
    global _INIT_THREAD, _PIPELINE_READY

    with _PIPELINE_LOCK:
        if _PIPELINE is None and _INIT_THREAD is None:
            _PIPELINE_READY.clear()
            _INIT_THREAD = threading.Thread(target=_init_pipeline_background, daemon=True)
            _INIT_THREAD.start()


def get_pipeline():
    global _PIPELINE
    _ensure_pipeline_thread()

    # Wait for the first model load; this can take time on a fresh start.
    if _PIPELINE_READY.wait(timeout=PIPELINE_INIT_TIMEOUT):
        if isinstance(_PIPELINE, Exception):
            raise _PIPELINE
        return _PIPELINE
    raise RuntimeError(
        f"Model loading is still in progress. Please try again in about {PIPELINE_INIT_TIMEOUT} seconds."
    )


def start_pipeline_warmup() -> None:
    """Kick off model loading as soon as the web app starts."""
    _ensure_pipeline_thread()


def get_db() -> TrafficDatabase:
    global _DB
    with _DB_LOCK:
        if _DB is None:
            _DB = TrafficDatabase(config.DATABASE_PATH)
        return _DB


def run_query(query: str, params=None) -> pd.DataFrame:
    params = params or []
    if not DB_PATH.exists():
        return pd.DataFrame()
    with sqlite3.connect(DB_PATH) as connection:
        return pd.read_sql_query(query, connection, params=params)


def db_scalar(query: str, params=None, default=0):
    frame = run_query(query, params)
    if frame.empty:
        return default
    value = frame.iloc[0, 0]
    return default if value is None else value


def load_public_stats() -> dict:
    return {
        "today_violations": int(
            db_scalar("SELECT COUNT(*) FROM violations WHERE date(violation_date)=date('now','localtime')", default=0)
        ),
        "total_violations": int(db_scalar("SELECT COUNT(*) FROM violations", default=0)),
        "registered_vehicles": int(db_scalar("SELECT COUNT(*) FROM registered_vehicles WHERE is_active=1", default=0)),
        "emails_sent": int(db_scalar("SELECT COUNT(*) FROM violations WHERE email_sent=1", default=0)),
    }


def load_cooldowns() -> pd.DataFrame:
    query = """
        SELECT plate_number, MAX(email_sent_time) AS last_sent
        FROM violations
        WHERE email_sent_time IS NOT NULL
          AND email_sent_time > datetime('now', '-12 hours')
        GROUP BY plate_number
        ORDER BY last_sent DESC
    """
    frame = run_query(query)
    if frame.empty:
        return frame

    rows = []
    now = pd.Timestamp.now()
    for _, record in frame.iterrows():
        try:
            last_sent = pd.to_datetime(record["last_sent"], errors="coerce")
        except Exception:
            last_sent = pd.NaT
        if pd.isna(last_sent):
            continue
        remaining = (last_sent + pd.Timedelta(hours=12)) - now
        rows.append(
            {
                "plate_number": record["plate_number"],
                "last_sent": last_sent.strftime("%Y-%m-%d %H:%M:%S"),
                "hours_remaining": round(max(0.0, remaining.total_seconds()) / 3600.0, 2),
            }
        )
    return pd.DataFrame(rows)


def load_recent_violations(limit: int = 25) -> pd.DataFrame:
    query = """
        SELECT
            v.id,
            v.plate_number,
            v.violation_date,
            v.violation_type,
            v.fine_amount,
            v.status,
            v.email_sent,
            v.evidence_path,
            COALESCE(i.invoice_number, '-') AS invoice_number,
            COALESCE(r.owner_name, '-') AS owner_name,
            COALESCE(r.owner_email, '-') AS owner_email
        FROM violations v
        LEFT JOIN invoices i ON i.violation_id = v.id
        LEFT JOIN registered_vehicles r ON r.plate_number = v.plate_number
        ORDER BY v.violation_date DESC
        LIMIT ?
    """
    return run_query(query, [limit])


def load_registered_vehicles() -> pd.DataFrame:
    query = """
        SELECT plate_number, owner_name, owner_email, phone, address, vehicle_model, registration_date, is_active
        FROM registered_vehicles
        ORDER BY registration_date DESC
    """
    return run_query(query)


def path_to_artifact_url(path_value: str | None):
    if not path_value:
        return None

    path = Path(path_value).resolve()
    roots = {
        "outputs": OUTPUTS_DIR.resolve(),
        "invoices": INVOICES_DIR.resolve(),
    }

    for category, root in roots.items():
        try:
            relative = path.relative_to(root)
        except ValueError:
            continue
        return url_for("serve_artifact", category=category, filename=str(relative).replace("\\", "/"))

    return None


def normalize_record(record: dict) -> dict:
    return {
        "timestamp": record.get("timestamp", ""),
        "image_file": record.get("image_file", ""),
        "plate_text": record.get("plate_text", "NOT_DETECTED"),
        "violation_count": int(record.get("violation_count", 0) or 0),
        "ocr_conf": round(float(record.get("ocr_conf", 0.0) or 0.0), 4),
        "ocr_method": record.get("ocr_method", "none"),
        "owner_name": record.get("owner_name", ""),
        "owner_email": record.get("owner_email", ""),
        "registered": bool(record.get("registered", False)),
        "email_sent": bool(record.get("email_sent", False)),
        "status": record.get("status", ""),
        "violation_id": record.get("violation_id", ""),
        "invoice_url": path_to_artifact_url(record.get("invoice_path")),
        "evidence_url": path_to_artifact_url(record.get("evidence_path")),
    }


def serialize_image_result(result: dict | None, source_name: str) -> dict:
    if not result:
        return {
            "success": False,
            "message": "The pipeline did not return a result.",
            "source_name": source_name,
        }

    detection = result.get("detection", {}) or {}
    records = [normalize_record(item) for item in result.get("violation_records", []) or []]
    unique_plates = []
    for record in records:
        plate = record.get("plate_text")
        if plate and plate != "NOT_DETECTED" and plate not in unique_plates:
            unique_plates.append(plate)

    return {
        "success": True,
        "source_name": source_name,
        "message": "Processed successfully.",
        "summary": {
            "total_riders": int(detection.get("total_riders", 0) or 0),
            "safe_riders": int(detection.get("safe_count", 0) or 0),
            "violations": int(detection.get("violation_count", 0) or 0),
            "has_violation": bool(detection.get("has_violation", False)),
            "avg_conf": float(detection.get("avg_conf", 0.0) or 0.0),
        },
        "plates": unique_plates,
        "records": records,
        "annotated_url": path_to_artifact_url(result.get("annotated_path")),
        "annotated_path": result.get("annotated_path"),
    }


def serialize_video_result(video_name: str, results: list[dict]) -> dict:
    frames = []
    plates = []
    total_violations = 0

    for item in results:
        detection = item.get("detection", {}) or {}
        records = [normalize_record(record) for record in item.get("violation_records", []) or []]
        frame_plates = []
        for record in records:
            plate = record.get("plate_text")
            if plate and plate != "NOT_DETECTED" and plate not in plates:
                plates.append(plate)
            if plate and plate != "NOT_DETECTED" and plate not in frame_plates:
                frame_plates.append(plate)
        total_violations += int(detection.get("violation_count", 0) or 0)
        frames.append(
            {
                "frame_name": Path(item.get("image_path", "")).name,
                "violations": int(detection.get("violation_count", 0) or 0),
                "riders": int(detection.get("total_riders", 0) or 0),
                "email_sent": any(record.get("email_sent") for record in records),
                "plates": frame_plates,
                "annotated_url": path_to_artifact_url(item.get("annotated_path")),
            }
        )

    return {
        "success": True,
        "video_name": video_name,
        "processed_frames": len(results),
        "violations": total_violations,
        "plates": plates,
        "frames": frames,
    }


def save_upload(file_storage, folder_name: str) -> Path:
    ensure_directories()
    safe_name = secure_filename(file_storage.filename or folder_name)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    folder = UPLOADS_DIR / folder_name
    folder.mkdir(parents=True, exist_ok=True)
    target_path = folder / f"{stamp}_{safe_name}"
    file_storage.save(target_path)
    return target_path


def process_image_file(file_storage):
    source_path = save_upload(file_storage, "images")
    try:
        with _PROCESS_LOCK:
            pipeline = get_pipeline()
    except RuntimeError as exc:
        return {"success": False, "message": str(exc)}
    except Exception as exc:
        return {"success": False, "message": f"Pipeline initialization failed: {exc}"}

    with _PROCESS_LOCK:
        result = pipeline.process_image(str(source_path), show=False, save=True)
    return serialize_image_result(result, file_storage.filename or source_path.name)


def process_video_file(file_storage, frame_stride: int, max_frames: int):
    source_path = save_upload(file_storage, "videos")
    cap = cv2.VideoCapture(str(source_path))
    if not cap.isOpened():
        return {"success": False, "message": "The video could not be opened."}

    try:
        pipeline = get_pipeline()
    except RuntimeError as exc:
        return {"success": False, "message": str(exc)}
    except Exception as exc:
        return {"success": False, "message": f"Pipeline initialization failed: {exc}"}
    frame_dir = UPLOADS_DIR / f"video_frames_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    frame_dir.mkdir(parents=True, exist_ok=True)

    results = []
    processed = 0
    frame_index = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_index % frame_stride == 0:
            frame_path = frame_dir / f"frame_{frame_index:06d}.jpg"
            cv2.imwrite(str(frame_path), frame)
            with _PROCESS_LOCK:
                frame_result = pipeline.process_image(str(frame_path), show=False, save=True)
            if frame_result:
                results.append(frame_result)
            processed += 1
            if processed >= max_frames:
                break

        frame_index += 1

    cap.release()
    return serialize_video_result(file_storage.filename or source_path.name, results)


def require_admin():
    if session.get("role") != "admin":
        flash("Admin access required.", "error")
        return False
    return True


@app.context_processor
def inject_globals():
    settings = load_ui_settings()
    stats = load_public_stats()
    return {
        "settings": settings,
        "public_stats": stats,
        "test_email": config.TEST_EMAIL,
        "admin_username": session.get("admin_username"),
        "is_admin": session.get("role") == "admin",
        "is_user": session.get("role") == "user",
    }

print("[DEBUG] web_app.py: Context processor registered", flush=True)
print("[DEBUG] web_app.py: Registering routes...", flush=True)

@app.route("/")
def landing():
    return render_template("landing.html")


@app.route("/user")
def user_dashboard():
    return render_template("user.html")


@app.route("/admin/login", methods=["GET", "POST"])
def admin_login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
            session.clear()
            session["role"] = "admin"
            session["admin_username"] = username
            flash("Admin login successful.", "success")
            return redirect(url_for("admin_dashboard"))
        flash("Invalid admin credentials.", "error")
    return render_template("login.html")


@app.route("/admin/logout")
def admin_logout():
    session.clear()
    flash("Logged out.", "success")
    return redirect(url_for("landing"))


@app.route("/admin")
def admin_dashboard():
    if not require_admin():
        return redirect(url_for("admin_login"))

    settings = load_ui_settings()
    cooldowns = load_cooldowns()
    recent_violations = load_recent_violations()
    vehicles = load_registered_vehicles()

    return render_template(
        "admin.html",
        settings=settings,
        cooldowns=cooldowns.to_dict(orient="records") if not cooldowns.empty else [],
        recent_violations=recent_violations.to_dict(orient="records") if not recent_violations.empty else [],
        vehicles=vehicles.to_dict(orient="records") if not vehicles.empty else [],
        admin_stats=load_public_stats(),
    )


@app.route("/admin/settings", methods=["POST"])
def admin_settings():
    if not require_admin():
        return redirect(url_for("admin_login"))

    payload = {
        "helmet_conf": float(request.form.get("helmet_conf", config.HELMET_CONF)),
        "plate_conf": float(request.form.get("plate_conf", config.PLATE_CONF)),
        "fine_amount": float(request.form.get("fine_amount", config.FINE_AMOUNT)),
        "send_email": request.form.get("send_email") == "on",
    }

    save_ui_settings(payload)
    apply_runtime_settings(payload)
    reset_pipeline()
    flash("Settings saved.", "success")
    return redirect(url_for("admin_dashboard"))


@app.route("/admin/vehicles/save", methods=["POST"])
def admin_vehicle_save():
    if not require_admin():
        return redirect(url_for("admin_login"))

    plate = request.form.get("plate_number", "").strip().upper()
    owner_name = request.form.get("owner_name", "").strip()
    owner_email = request.form.get("owner_email", "").strip()
    phone = request.form.get("phone", "").strip()
    address = request.form.get("address", "").strip()
    vehicle_model = request.form.get("vehicle_model", "").strip()

    if not plate or not owner_name or not owner_email:
        flash("Plate number, owner name, and owner email are required.", "error")
        return redirect(url_for("admin_dashboard"))

    ok = get_db().register_vehicle(plate, owner_name, owner_email, phone=phone, address=address, vehicle_model=vehicle_model)
    flash("Vehicle saved." if ok else "Vehicle save failed.", "success" if ok else "error")
    return redirect(url_for("admin_dashboard"))


@app.route("/admin/vehicles/deactivate", methods=["POST"])
def admin_vehicle_deactivate():
    if not require_admin():
        return redirect(url_for("admin_login"))

    plate = request.form.get("plate_number", "").strip().upper()
    if not plate:
        flash("Enter a plate number to deactivate.", "error")
        return redirect(url_for("admin_dashboard"))

    try:
        with sqlite3.connect(DB_PATH) as connection:
            connection.execute("UPDATE registered_vehicles SET is_active=0 WHERE plate_number=?", (plate,))
            connection.commit()
        flash(f"Vehicle {plate} deactivated.", "success")
    except Exception as exc:
        flash(f"Failed to deactivate vehicle: {exc}", "error")
    return redirect(url_for("admin_dashboard"))


@app.route("/admin/smtp-test", methods=["POST"])
def admin_smtp_test():
    if not require_admin():
        return redirect(url_for("admin_login"))

    recipient = request.form.get("recipient_email", config.TEST_EMAIL).strip()
    try:
        result = FineSystem().test_send_via_both(recipient)
        if result.get("primary"):
            flash(f"Primary SMTP sent a test email to {recipient}.", "success")
        else:
            flash(f"Primary SMTP failed: {result.get('primary_error')}", "error")

        if result.get("backup"):
            flash(f"Backup SMTP sent a test email to {recipient}.", "success")
        elif result.get("backup_error"):
            flash(f"Backup SMTP: {result.get('backup_error')}", "error")
    except Exception as exc:
        flash(f"SMTP test error: {exc}", "error")

    return redirect(url_for("admin_dashboard"))


@app.route("/api/process-image", methods=["POST"])
def api_process_image():
    file_storage = request.files.get("media") or request.files.get("file")
    if not file_storage or not file_storage.filename:
        return jsonify({"success": False, "message": "Please upload an image."}), 400

    try:
        payload = process_image_file(file_storage)
    except Exception as exc:
        return jsonify({"success": False, "message": str(exc)}), 500

    return jsonify(payload)


@app.route("/api/process-video", methods=["POST"])
def api_process_video():
    file_storage = request.files.get("media") or request.files.get("file")
    if not file_storage or not file_storage.filename:
        return jsonify({"success": False, "message": "Please upload a video."}), 400

    frame_stride = int(request.form.get("frame_stride", 20))
    max_frames = int(request.form.get("max_frames", 40))
    frame_stride = max(1, min(frame_stride, 120))
    max_frames = max(1, min(max_frames, 200))

    try:
        payload = process_video_file(file_storage, frame_stride=frame_stride, max_frames=max_frames)
    except Exception as exc:
        return jsonify({"success": False, "message": str(exc)}), 500

    return jsonify(payload)


@app.route("/artifacts/<category>/<path:filename>")
def serve_artifact(category, filename):
    root_map = {
        "outputs": OUTPUTS_DIR,
        "invoices": INVOICES_DIR,
    }
    root = root_map.get(category)
    if root is None:
        abort(404)

    root_resolved = root.resolve()
    candidate = (root / filename).resolve()
    try:
        candidate.relative_to(root_resolved)
    except ValueError:
        abort(404)

    if not candidate.exists():
        abort(404)

    return send_from_directory(root, filename)


@app.route("/api/public-stats")
def api_public_stats():
    return jsonify(load_public_stats())


@app.route("/api/admin/overview")
def api_admin_overview():
    if not require_admin():
        return jsonify({"success": False, "message": "Admin access required."}), 403
    return jsonify(
        {
            "success": True,
            "stats": load_public_stats(),
            "cooldowns": load_cooldowns().to_dict(orient="records") if not load_cooldowns().empty else [],
            "recent_violations": load_recent_violations().to_dict(orient="records") if not load_recent_violations().empty else [],
        }
    )


@app.route('/admin/violation/verify', methods=['POST'])
def admin_verify_violation():
    if not require_admin():
        return jsonify({"success": False, "message": "Admin access required."}), 403

    violation_id = request.form.get('violation_id') or (request.json and request.json.get('violation_id'))
    action = (request.form.get('action') or (request.json and request.json.get('action')) or '').lower()
    if not violation_id or action not in ('confirm', 'reject'):
        return jsonify({"success": False, "message": "violation_id and action (confirm|reject) are required."}), 400

    try:
        vid = int(violation_id)
    except Exception:
        return jsonify({"success": False, "message": "Invalid violation_id."}), 400

    # Load violation row
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cur = conn.cursor()
            cur.execute("SELECT id, plate_number, violation_date, violation_type, evidence_path, fine_amount, status, email_sent FROM violations WHERE id=?", (vid,))
            row = cur.fetchone()
            if not row:
                return jsonify({"success": False, "message": "Violation not found."}), 404
            _id, plate, violation_date, violation_type, evidence_path, fine_amount, status, email_sent = row

            if action == 'reject':
                cur.execute("UPDATE violations SET status=? WHERE id=?", ('REJECTED', vid))
                conn.commit()
                return jsonify({"success": True, "message": f"Violation {vid} marked REJECTED."})

            # action == confirm: issue fine (generate pdf + attempt email) and mark as confirmed
            vehicle = get_db().get_vehicle(plate)
            if not vehicle:
                # still update status but warn
                cur.execute("UPDATE violations SET status=? WHERE id=?", ('CONFIRMED', vid))
                conn.commit()
                return jsonify({"success": False, "message": f"No registered vehicle for {plate}. Status set to CONFIRMED but email not sent."})

            # call issue_fine helper which generates PDF and attempts to send email
            try:
                from fine_system import issue_fine
                res = issue_fine(
                    violation_id=vid,
                    vehicle_info=vehicle,
                    violation_type=violation_type,
                    violation_date=violation_date,
                    evidence_path=evidence_path,
                    fine_amount=fine_amount,
                    send_email=True,
                    recipient_email=vehicle.get('owner_email'),
                    original_image_path=None,
                    test_mode=(vehicle.get('owner_email') == config.TEST_EMAIL),
                )

                # mark email_sent when reported true
                if res.get('email_sent'):
                    get_db().mark_email_sent(vid)

                cur.execute("UPDATE violations SET status=? WHERE id=?", ('CONFIRMED', vid))
                conn.commit()
                return jsonify({"success": True, "message": f"Violation {vid} confirmed.", "email_sent": bool(res.get('email_sent'))})
            except Exception as exc:
                cur.execute("UPDATE violations SET status=? WHERE id=?", ('CONFIRMED', vid))
                conn.commit()
                return jsonify({"success": False, "message": f"Error issuing fine: {exc}"}), 500

    except Exception as exc:
        return jsonify({"success": False, "message": f"Unexpected error: {exc}"}), 500


@app.errorhandler(413)
def request_entity_too_large(_error):
    return jsonify({"success": False, "message": "File is too large for this server."}), 413


@app.cli.command("init-db")
def init_db_command():
    ensure_directories()
    get_db()
    print("Database initialized.")


print("[DEBUG] web_app.py: All routes and handlers registered! Flask app is ready.", flush=True)

@app.before_request
def _log_request():
    print(f"[REQUEST] {request.method} {request.path}", flush=True)

print("[DEBUG] web_app.py: Request logging hook registered. Worker is now listening for connections.", flush=True)
import sys
sys.stdout.flush()

if __name__ == "__main__":
    ensure_directories()
    apply_runtime_settings(load_ui_settings())
    start_pipeline_warmup()
    port = int(os.getenv("PORT", "5000"))
    host = os.getenv("HOST", "0.0.0.0")
    print(f"[READY] Application startup complete. Listening on http://{host}:{port}", flush=True)
    try:
        from waitress import serve

        serve(app, host=host, port=port)
    except Exception:
        app.run(host=host, port=port, debug=False)
