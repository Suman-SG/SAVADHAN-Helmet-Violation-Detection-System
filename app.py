import io
import os
import sqlite3
import tempfile
import time
from datetime import date, datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

import config
from database import TrafficDatabase
from fine_system import FineSystem
from main_pipeline import ViolationPipeline
import sys, subprocess
try:
    import torch
except Exception:
    torch = None


BASE_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = BASE_DIR / "outputs"
EVIDENCE_DIR = OUTPUTS_DIR / "evidence"
ANNOTATED_DIR = OUTPUTS_DIR / "annotated"
UPLOADS_DIR = OUTPUTS_DIR / "uploads"
SETTINGS_PATH = OUTPUTS_DIR / "ui_settings.json"
DB_PATH = Path(config.DATABASE_PATH)


DEFAULT_SETTINGS = {
    "helmet_conf": float(config.HELMET_CONF),
    "plate_conf": float(config.PLATE_CONF),
    "fine_amount": float(config.FINE_AMOUNT),
    "send_email": bool(config.SEND_EMAIL),
}


def ensure_ui_dirs():
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    ANNOTATED_DIR.mkdir(parents=True, exist_ok=True)
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)


def load_ui_settings() -> dict:
    ensure_ui_dirs()
    if SETTINGS_PATH.exists():
        try:
            payload = pd.read_json(SETTINGS_PATH, typ="series").to_dict()
            merged = dict(DEFAULT_SETTINGS)
            merged.update(payload)
            return merged
        except Exception:
            return dict(DEFAULT_SETTINGS)
    return dict(DEFAULT_SETTINGS)


def save_ui_settings(settings: dict) -> None:
    ensure_ui_dirs()
    pd.Series(settings).to_json(SETTINGS_PATH, indent=2)


def apply_runtime_settings(settings: dict) -> None:
    config.HELMET_CONF = float(settings["helmet_conf"])
    config.PLATE_CONF = float(settings["plate_conf"])
    config.FINE_AMOUNT = float(settings["fine_amount"])
    config.SEND_EMAIL = bool(settings["send_email"])


def inject_styles() -> None:
    st.markdown(
        """
        <style>
            .stApp {
                background:
                    radial-gradient(circle at 15% 10%, rgba(12, 40, 36, 0.55), transparent 32%),
                    radial-gradient(circle at 85% 15%, rgba(42, 10, 10, 0.45), transparent 35%),
                    linear-gradient(160deg, #0b1014 0%, #131a21 60%, #0f141b 100%);
                color: #e8eef5;
            }
            .main-title {
                font-size: 2rem;
                font-weight: 700;
                margin-bottom: 0.35rem;
                color: #ecf3ff;
            }
            .sub-title {
                color: #9fb2c8;
                margin-bottom: 1rem;
            }
            .metric-card {
                padding: 1rem;
                border-radius: 14px;
                border: 1px solid rgba(255,255,255,0.12);
                background: linear-gradient(145deg, rgba(29,36,47,0.82), rgba(18,24,32,0.92));
                box-shadow: 0 8px 18px rgba(0,0,0,0.25);
                margin-bottom: 0.6rem;
                animation: riseIn 0.35s ease;
            }
            .metric-label {
                color: #b6c7db;
                font-size: 0.9rem;
                margin-bottom: 0.25rem;
            }
            .metric-value {
                font-size: 1.45rem;
                font-weight: 700;
                color: #f6fbff;
            }
            .ok-pill, .bad-pill {
                display: inline-block;
                padding: 0.25rem 0.55rem;
                border-radius: 999px;
                font-size: 0.78rem;
                font-weight: 700;
            }
            .ok-pill {
                background: rgba(35, 160, 95, 0.18);
                border: 1px solid rgba(35, 160, 95, 0.6);
                color: #7dffb8;
            }
            .bad-pill {
                background: rgba(200, 60, 60, 0.18);
                border: 1px solid rgba(220, 80, 80, 0.6);
                color: #ff9f9f;
            }
            @keyframes riseIn {
                from { transform: translateY(8px); opacity: 0; }
                to { transform: translateY(0); opacity: 1; }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def get_connection() -> sqlite3.Connection:
    ensure_ui_dirs()
    return sqlite3.connect(DB_PATH)


def run_query(query: str, params=None) -> pd.DataFrame:
    params = params or []
    if not DB_PATH.exists():
        return pd.DataFrame()
    with get_connection() as conn:
        return pd.read_sql_query(query, conn, params=params)


def db_scalar(query: str, params=None, default=0):
    df = run_query(query, params)
    if df.empty:
        return default
    value = df.iloc[0, 0]
    return default if value is None else value


def load_csv_violations() -> pd.DataFrame:
    csv_path = Path(config.CSV_PATH)
    if csv_path.exists() and csv_path.stat().st_size > 0:
        try:
            df = pd.read_csv(csv_path)
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            return df
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


def fetch_recent_detections(limit: int = 10) -> pd.DataFrame:
    df = load_csv_violations()
    if df.empty:
        return df
    if "timestamp" in df.columns:
        df = df.sort_values("timestamp", ascending=False)
    return df.head(limit)


def list_evidence_images(limit: int = 200):
    ensure_ui_dirs()
    files = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
        files.extend(EVIDENCE_DIR.glob(ext))
    files = sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)
    return files[:limit]


def fit_image_to_canvas(image: np.ndarray, canvas_w: int = 960, canvas_h: int = 560) -> np.ndarray:
    """Resize image with padding so all preview/output cards have identical size."""
    if image is None or image.size == 0:
        return np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    h, w = image.shape[:2]
    scale = min(canvas_w / max(w, 1), canvas_h / max(h, 1))
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.full((canvas_h, canvas_w, 3), 18, dtype=np.uint8)
    x0 = (canvas_w - new_w) // 2
    y0 = (canvas_h - new_h) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
    return canvas


def decode_image_bytes(image_bytes: bytes) -> np.ndarray:
    nparr = np.frombuffer(image_bytes, np.uint8)
    bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return bgr


@st.cache_resource(show_spinner=False)
def get_pipeline():
    return ViolationPipeline(suppress_emails=False)


def save_uploaded_file(uploaded_file, folder: Path) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    target = folder / f"{timestamp}_{uploaded_file.name}"
    with open(target, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return target


def process_image_with_pipeline(image_path: Path):
    pipeline = get_pipeline()
    return pipeline.process_image(str(image_path), show=False, save=True)


def process_video_with_pipeline(video_path: Path, frame_stride: int, max_frames: int):
    pipeline = get_pipeline()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []

    frame_dir = UPLOADS_DIR / f"video_frames_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    frame_dir.mkdir(parents=True, exist_ok=True)

    results = []
    processed = 0
    frame_idx = 0

    progress = st.progress(0.0)
    status = st.empty()

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    estimated = max(1, min(max_frames, (total_frames // max(1, frame_stride)) + 1))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_stride == 0:
            frame_path = frame_dir / f"frame_{frame_idx:06d}.jpg"
            cv2.imwrite(str(frame_path), frame)
            result = pipeline.process_image(str(frame_path), show=False, save=True)
            if result:
                results.append(result)
            processed += 1
            progress.progress(min(processed / estimated, 1.0))
            status.info(f"Processed frames: {processed}/{estimated}")

            if processed >= max_frames:
                break

        frame_idx += 1

    cap.release()
    progress.empty()
    status.empty()
    return results


def system_status_html() -> str:
    checks = {
        "DB": DB_PATH.exists(),
        "Helmet model": Path(config.HELMET_MODEL_PATH).exists(),
        "Plate model": Path(config.PLATE_MODEL_PATH).exists(),
        "Evidence dir": EVIDENCE_DIR.exists(),
    }
    bad = [name for name, ok in checks.items() if not ok]
    if bad:
        return f"<span class='bad-pill'>PARTIAL ({', '.join(bad)})</span>"
    return "<span class='ok-pill'>RUNNING</span>"


def render_kpi(label: str, value):
    st.markdown(
        f"""
        <div class='metric-card'>
            <div class='metric-label'>{label}</div>
            <div class='metric-value'>{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def page_dashboard():
    st.markdown("<div class='main-title'>Helmet Violation Monitoring Dashboard</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-title'>Live snapshot of detection, fines, and system health.</div>", unsafe_allow_html=True)

    today_count = db_scalar(
        "SELECT COUNT(*) FROM violations WHERE date(violation_date)=date('now','localtime')", default=0
    )
    total_plates = db_scalar("SELECT COUNT(*) FROM violations WHERE plate_number IS NOT NULL AND plate_number != ''", default=0)
    email_sent = db_scalar("SELECT COUNT(*) FROM violations WHERE email_sent=1", default=0)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_kpi("Total Violations Today", int(today_count))
    with c2:
        render_kpi("Total Plates Detected", int(total_plates))
    with c3:
        render_kpi("Emails Sent", int(email_sent))
    with c4:
        render_kpi("System Status", "")
        st.markdown(system_status_html(), unsafe_allow_html=True)

    # Show active cooldowns (plates emailed within last 12 hours)
    cooldown_query = """
        SELECT plate_number, MAX(email_sent_time) as last_sent
        FROM violations
        WHERE email_sent_time IS NOT NULL
        AND email_sent_time > datetime('now', '-12 hours')
        GROUP BY plate_number
        ORDER BY last_sent DESC
    """
    cooldown_df = run_query(cooldown_query)
    if not cooldown_df.empty:
        from datetime import datetime as _dt
        rows = []
        for _, r in cooldown_df.iterrows():
            try:
                last = _dt.fromisoformat(str(r['last_sent']))
            except Exception:
                try:
                    last = _dt.strptime(str(r['last_sent']), "%Y-%m-%d %H:%M:%S")
                except Exception:
                    last = None
            if last:
                # Compute remaining as a Timedelta then convert to hours safely
                remaining_td = (pd.Timestamp(last) + pd.Timedelta(hours=12)) - pd.Timestamp.now()
                seconds_left = max(0.0, remaining_td.total_seconds())
                hours_left = seconds_left / 3600.0
                rows.append({
                    "plate": r['plate_number'],
                    "last_sent": last.strftime("%Y-%m-%d %H:%M:%S"),
                    "hours_remaining": round(hours_left, 2)
                })
        if rows:
            st.subheader("Active Cooldowns (12h)")
            st.table(pd.DataFrame(rows))

    st.subheader("Recent Detections")
    recent_df = fetch_recent_detections(limit=12)
    if recent_df.empty:
        st.info("No detections available yet. Run an image or video from Upload page.")
    else:
        show_cols = [
            c for c in [
                "timestamp", "image_file", "plate_text", "violation_count", "ocr_conf", "owner_name", "email_sent"
            ] if c in recent_df.columns
        ]
        st.dataframe(recent_df[show_cols], width="stretch", hide_index=True)


def page_upload_image():
    st.header("Upload Image")
    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp"])
    camera = st.camera_input("Or capture from webcam (optional)")

    source_bytes = None
    source_name = None

    if uploaded is not None:
        source_bytes = uploaded.getvalue()
        source_name = uploaded.name
    elif camera is not None:
        source_bytes = camera.getvalue()
        source_name = f"camera_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"

    if source_bytes:
        preview_bgr = decode_image_bytes(source_bytes)
        preview_canvas = fit_image_to_canvas(preview_bgr)
        st.image(preview_canvas, caption="Input Preview", width="stretch")

    if st.button("Start Detection", type="primary", width="stretch"):
        if not source_bytes:
            st.warning("Please upload or capture an image first.")
            return

        temp_file = UPLOADS_DIR / source_name
        with open(temp_file, "wb") as f:
            f.write(source_bytes)

        with st.spinner("Running full pipeline..."):
            try:
                result = process_image_with_pipeline(temp_file)
            except Exception as e:
                st.error(f"Pipeline failed: {e}")
                return

        if not result:
            st.error("No result returned from pipeline.")
            return

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original Image")
            original_bgr = decode_image_bytes(source_bytes)
            st.image(fit_image_to_canvas(original_bgr), width="stretch")

        with col2:
            st.subheader("Detected Output")
            annotated = result.get("annotated")
            if annotated is not None:
                st.image(fit_image_to_canvas(annotated), width="stretch")
            else:
                st.info("Annotated output not available.")

        records = pd.DataFrame(result.get("violation_records", []))
        violation_happened = int(result.get("detection", {}).get("violation_count", 0)) > 0
        email_sent_count = 0

        if not records.empty:
            if "ocr_conf" in records.columns:
                records["ocr_conf"] = (records["ocr_conf"].fillna(0.0) * 100).round(2)
            if "email_sent" in records.columns:
                email_sent_count = int(records["email_sent"].fillna(False).astype(bool).sum())

            st.subheader("Final Status")
            s1, s2, s3 = st.columns(3)
            s1.metric("Violation Happened", "YES" if violation_happened else "NO")
            s2.metric("Email Sent Status", "YES" if email_sent_count > 0 else "NO")
            s3.metric("Violations in Image", int(result.get("detection", {}).get("violation_count", 0)))

            st.subheader("Detection Details")
            st.dataframe(records, width="stretch", hide_index=True)

            if violation_happened and email_sent_count == 0 and config.SEND_EMAIL:
                st.warning("Violation found but email not delivered. SMTP auth is failing (535 BadCredentials). Update Gmail App Password and restart Streamlit.")
        else:
            st.subheader("Final Status")
            s1, s2 = st.columns(2)
            s1.metric("Violation Happened", "NO")
            s2.metric("Email Sent Status", "NO")
            st.success("No violations found in this image.")


def page_upload_video():
    st.header("Upload Video")
    st.caption("Video processing samples frames for speed. Lower frame stride = more accurate, slower.")

    uploaded = st.file_uploader("Upload video", type=["mp4", "avi", "mov", "mkv"])
    frame_stride = st.slider("Process every Nth frame", 5, 60, 20)
    max_frames = st.slider("Max frames to process", 5, 120, 40)

    if st.button("Start Video Detection", type="primary", width="stretch"):
        if uploaded is None:
            st.warning("Please upload a video first.")
            return

        video_path = save_uploaded_file(uploaded, UPLOADS_DIR)

        with st.spinner("Analyzing video frames..."):
            try:
                results = process_video_with_pipeline(video_path, frame_stride=frame_stride, max_frames=max_frames)
            except Exception as e:
                st.error(f"Video pipeline failed: {e}")
                return

        if not results:
            st.error("No frames were processed.")
            return

        total_violations = sum(int(r["detection"].get("violation_count", 0)) for r in results if r.get("detection"))
        total_records = sum(len(r.get("violation_records", [])) for r in results)

        c1, c2 = st.columns(2)
        c1.metric("Processed Frames", len(results))
        c2.metric("Violations Detected", total_violations)

        st.subheader("Frame-Level Results")
        frame_rows = []
        for r in results:
            frame_rows.append(
                {
                    "frame": os.path.basename(r.get("image_path", "")),
                    "violations": r.get("detection", {}).get("violation_count", 0),
                    "riders": r.get("detection", {}).get("total_riders", 0),
                    "records": len(r.get("violation_records", [])),
                }
            )
        st.dataframe(pd.DataFrame(frame_rows), width="stretch", hide_index=True)
        st.caption(f"Violation records generated: {total_records}")


def page_violation_history():
    st.header("Violation History")

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
            COALESCE(i.invoice_number, '-') AS invoice_number
        FROM violations v
        LEFT JOIN invoices i ON i.violation_id = v.id
        ORDER BY v.violation_date DESC
    """
    df = run_query(query)

    if df.empty:
        st.info("No violations in database yet.")
        return

    df["violation_date"] = pd.to_datetime(df["violation_date"], errors="coerce")

    c1, c2, c3 = st.columns([2, 2, 2])
    with c1:
        search_plate = st.text_input("Search Plate", placeholder="TN72P2931")
    with c2:
        min_date = st.date_input("From Date", value=date.today().replace(day=1))
    with c3:
        max_date = st.date_input("To Date", value=date.today())

    mask = pd.Series(True, index=df.index)
    if search_plate.strip():
        mask &= df["plate_number"].astype(str).str.contains(search_plate.strip(), case=False, na=False)

    mask &= df["violation_date"].dt.date.between(min_date, max_date)
    filtered = df[mask].copy()

    st.dataframe(filtered, width="stretch", hide_index=True)

    csv_bytes = filtered.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Export CSV",
        data=csv_bytes,
        file_name=f"violation_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
    )


def page_evidence_gallery():
    st.header("Evidence Gallery")

    files = list_evidence_images(limit=300)
    if not files:
        st.info("No evidence images found yet.")
        return

    plate_filter = st.text_input("Filter by plate in filename", value="")
    if plate_filter.strip():
        files = [f for f in files if plate_filter.upper() in f.name.upper()]

    cols_per_row = st.select_slider("Images per row", options=[2, 3, 4], value=2)
    cols = st.columns(cols_per_row)
    for idx, img_path in enumerate(files):
        with cols[idx % cols_per_row]:
            raw = cv2.imread(str(img_path))
            st.image(fit_image_to_canvas(raw, canvas_w=900, canvas_h=480), caption=img_path.name, width="stretch")
            stat = img_path.stat()
            st.caption(f"Updated: {datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')}")
            with open(img_path, "rb") as fp:
                st.download_button(
                    "Download",
                    data=fp.read(),
                    file_name=img_path.name,
                    mime="image/jpeg",
                    key=f"dl_{img_path.name}_{idx}",
                    width="stretch",
                )


def page_database_view():
    st.header("Database View - Registered Vehicles")

    db = TrafficDatabase(config.DATABASE_PATH)
    vehicles = pd.DataFrame(db.get_all_vehicles())

    if vehicles.empty:
        st.info("No registered vehicles found.")
    else:
        plate_search = st.text_input("Search by plate/owner/email")
        filtered = vehicles.copy()
        if plate_search.strip():
            token = plate_search.strip()
            filtered = filtered[
                filtered.astype(str).apply(lambda col: col.str.contains(token, case=False, na=False)).any(axis=1)
            ]
        st.dataframe(filtered, width="stretch", hide_index=True)

    st.subheader("Add or Update Vehicle")
    c1, c2, c3 = st.columns(3)
    plate = c1.text_input("Plate Number").upper().strip()
    owner = c2.text_input("Owner Name").strip()
    email = c3.text_input("Owner Email").strip()

    c4, c5, c6 = st.columns(3)
    phone = c4.text_input("Phone").strip()
    address = c5.text_input("Address").strip()
    model = c6.text_input("Vehicle Model").strip()

    if st.button("Save Vehicle", type="primary"):
        if not plate or not owner or not email:
            st.warning("Plate, owner name, and email are required.")
        else:
            ok = db.register_vehicle(plate, owner, email, phone=phone, address=address, vehicle_model=model)
            if ok:
                st.success("Vehicle saved successfully.")
                st.rerun()
            else:
                st.error("Failed to save vehicle.")

    st.subheader("Delete Vehicle")
    del_plate = st.text_input("Plate Number to deactivate").upper().strip()
    if st.button("Delete Vehicle", type="secondary"):
        if not del_plate:
            st.warning("Enter plate number to delete.")
        else:
            try:
                with get_connection() as conn:
                    conn.execute(
                        "UPDATE registered_vehicles SET is_active=0 WHERE plate_number=?",
                        (del_plate,),
                    )
                    conn.commit()
                st.success(f"Vehicle {del_plate} deactivated.")
                st.rerun()
            except Exception as e:
                st.error(f"Delete failed: {e}")


def page_invoice_status():
    st.header("Email / Invoice Status")

    query = """
        SELECT
            v.plate_number,
            COALESCE(r.owner_email, '-') AS owner_email,
            COALESCE(i.invoice_number, '-') AS invoice_number,
            CASE WHEN v.email_sent = 1 THEN 'SENT' ELSE 'PENDING' END AS email_status,
            v.status,
            v.violation_date
        FROM violations v
        LEFT JOIN registered_vehicles r ON r.plate_number = v.plate_number
        LEFT JOIN invoices i ON i.violation_id = v.id
        ORDER BY v.violation_date DESC
    """
    df = run_query(query)

    if df.empty:
        st.info("No invoice/email data available.")
        return

    search_plate = st.text_input("Search Plate", "")
    if search_plate.strip():
        df = df[df["plate_number"].astype(str).str.contains(search_plate.strip(), case=False, na=False)]

    st.dataframe(df, width="stretch", hide_index=True)


def page_settings():
    st.header("Settings")
    settings = st.session_state["ui_settings"]

    c1, c2 = st.columns(2)
    with c1:
        helmet_conf = st.slider("Helmet Confidence Threshold", 0.05, 0.95, float(settings["helmet_conf"]), 0.01)
        plate_conf = st.slider("Plate Confidence Threshold", 0.05, 0.95, float(settings["plate_conf"]), 0.01)
    with c2:
        fine_amount = st.number_input("Fine Amount", min_value=100.0, max_value=50000.0, value=float(settings["fine_amount"]), step=100.0)
        send_email = st.toggle("Enable Email Sending", value=bool(settings["send_email"]))

    if st.button("Save Settings", type="primary"):
        payload = {
            "helmet_conf": float(helmet_conf),
            "plate_conf": float(plate_conf),
            "fine_amount": float(fine_amount),
            "send_email": bool(send_email),
        }
        st.session_state["ui_settings"] = payload
        apply_runtime_settings(payload)
        save_ui_settings(payload)
        get_pipeline.clear()
        st.success("Settings saved. New detections will use updated values.")

    st.subheader("SMTP Test")
    test_recipient = st.text_input("Test recipient email", value=config.TEST_EMAIL)
    if st.button("Send SMTP Test Email", type="secondary"):
        try:
            sender = FineSystem()
            res = sender.test_send_via_both(test_recipient)
            primary_ok = res.get('primary')
            backup_ok = res.get('backup')
            if primary_ok:
                st.success(f"Primary SMTP (configured) delivered test email to {test_recipient}.")
            else:
                st.warning(f"Primary SMTP failed: {res.get('primary_error')}")

            if backup_ok:
                st.success(f"Backup SMTP delivered test email to {test_recipient}.")
            else:
                st.info(f"Backup SMTP result: {res.get('backup_error')}")
        except Exception as e:
            st.error(f"SMTP test raised an error: {e}")

    st.caption("Settings are stored in outputs/ui_settings.json and applied at runtime.")


def main():
    # Print runtime GPU/Python diagnostics to help debug CUDA availability
    try:
        print(f"[Startup] Python executable: {sys.executable}")
        if torch is not None:
            print(f"[Startup] Torch: {torch.__version__}, CUDA available: {torch.cuda.is_available()}, GPU count: {torch.cuda.device_count()}")
        else:
            print("[Startup] Torch not importable in this environment")
        try:
            out = subprocess.check_output(['nvidia-smi', '-L'], stderr=subprocess.STDOUT, text=True)
            print('[Startup] nvidia-smi:\n' + out)
        except Exception as e:
            print(f"[Startup] nvidia-smi not available: {e}")
    except Exception:
        pass
    st.set_page_config(page_title="Helmet Violation Dashboard", page_icon="H", layout="wide")
    inject_styles()
    ensure_ui_dirs()

    if "ui_settings" not in st.session_state:
        st.session_state["ui_settings"] = load_ui_settings()
    apply_runtime_settings(st.session_state["ui_settings"])

    with st.sidebar:
        st.title("Navigation")
        page = st.radio(
            "Go to",
            [
                "Dashboard",
                "Upload Image",
                "Upload Video",
                "Violation History",
                "Evidence Gallery",
                "Database View",
                "Invoice Status",
                "Settings",
            ],
            index=0,
        )

    if page == "Dashboard":
        page_dashboard()
    elif page == "Upload Image":
        page_upload_image()
    elif page == "Upload Video":
        page_upload_video()
    elif page == "Violation History":
        page_violation_history()
    elif page == "Evidence Gallery":
        page_evidence_gallery()
    elif page == "Database View":
        page_database_view()
    elif page == "Invoice Status":
        page_invoice_status()
    elif page == "Settings":
        page_settings()


if __name__ == "__main__":
    main()
