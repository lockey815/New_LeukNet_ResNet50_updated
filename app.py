"""
LeukNet AML Detector - Streamlit app for Render

- Admin can upload a .keras model from their own computer (sidebar).
- The model is saved permanently:
    1) on a Render persistent disk (DATA_DIR, paid plans), and/or
    2) in a private Hugging Face model repo (HF_TOKEN + HF_MODEL_REPO, free).
  On every start the app reloads the saved model automatically.
- With no model, the app runs in DEMO mode (simulated results).
"""

import hmac
import os
import shutil
import zipfile
import zlib
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image

# ==========================================
# SETTINGS (from Render environment variables)
# ==========================================
MODEL_NAME = "best_resnet50.keras"
DATA_DIR = os.environ.get("DATA_DIR") or ("/var/data" if os.path.isdir("/var/data") else "data")
try:
    os.makedirs(DATA_DIR, exist_ok=True)
except OSError:
    DATA_DIR = "/tmp/leuknet"
    os.makedirs(DATA_DIR, exist_ok=True)

MODEL_PATH = os.path.join(DATA_DIR, MODEL_NAME)
HF_REPO = os.environ.get("HF_MODEL_REPO", "").strip()
HF_TOKEN = os.environ.get("HF_TOKEN") or None
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "")

# Same order as training: 0 = monocyte, 1 = myeloblast
CLASS_LABELS = ["Monocyte (Non-AML)", "Myeloblast (AML-positive)"]

st.set_page_config(page_title="LeukNet AML Detector", layout="centered")
st.markdown("""
    <style>
    .stButton>button { width: 100%; background-color: #1f2937; color: white; border-radius: 5px; }
    .stButton>button:hover { background-color: #111827; color: white; }
    .report-box { border-left: 4px solid #3b82f6; background-color: #eff6ff;
                  padding: 15px; border-radius: 5px; margin-top: 20px; color: #111827; }
    .report-box.red { border-left-color: #ef4444; background-color: #fef2f2; }
    </style>
""", unsafe_allow_html=True)

for key, default in [("patients", []), ("reports", []), ("last_result", None)]:
    if key not in st.session_state:
        st.session_state[key] = default

# ==========================================
# MODEL STORAGE
# ==========================================
def is_valid_file(path):
    return os.path.isfile(path) and zipfile.is_zipfile(path)


def check_model(path):
    """Return an error message, or None if the model is usable."""
    if not zipfile.is_zipfile(path):
        return "This is not a complete .keras file (the upload may be cut off)."
    try:
        m = tf.keras.models.load_model(path, compile=False)
    except Exception as e:
        return f"Keras could not load this file: {e}"
    ok = len(m.input_shape) == 4 and m.output_shape[-1] == 2
    del m
    if not ok:
        return "The model must take images and output 2 classes (monocyte, myeloblast)."
    return None


def restore_from_hub():
    if not HF_REPO:
        return False
    try:
        from huggingface_hub import hf_hub_download
        p = hf_hub_download(repo_id=HF_REPO, filename=MODEL_NAME, token=HF_TOKEN)
        shutil.copyfile(p, MODEL_PATH)
        return is_valid_file(MODEL_PATH)
    except Exception:
        return False


def push_to_hub(path):
    """Back up the model to a private Hugging Face repo. Returns an error text or None."""
    if not (HF_REPO and HF_TOKEN):
        return "No Hugging Face backup configured."
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=HF_TOKEN)
        api.create_repo(HF_REPO, repo_type="model", private=True, exist_ok=True)
        api.upload_file(path_or_fileobj=path, path_in_repo=MODEL_NAME,
                        repo_id=HF_REPO, repo_type="model")
        return None
    except Exception as e:
        return f"Backup failed: {e}"


@st.cache_resource(show_spinner="Looking for the saved model...")
def ensure_model_file():
    """Runs once per start (and after an upload). Returns where the model came from."""
    if is_valid_file(MODEL_PATH):
        return "disk"
    if restore_from_hub():
        return "Hugging Face backup"
    return None


@st.cache_resource(show_spinner="Loading model...")
def load_model(path, mtime):
    return tf.keras.models.load_model(path, compile=False)


def get_model():
    source = ensure_model_file()
    if source is None:
        return None, None
    try:
        return load_model(MODEL_PATH, os.path.getmtime(MODEL_PATH)), source
    except Exception:
        return None, "broken"


def predict(model, image, raw=b""):
    """Return (class_index, confidence %). Without a model: SIMULATED (demo)."""
    if model is None:
        rng = np.random.default_rng(zlib.crc32(raw))
        return int(rng.integers(0, 2)), float(rng.uniform(70, 99))
    size = model.input_shape[1]
    img = image.convert("RGB").resize((size, size), Image.BILINEAR)
    x = tf.keras.applications.resnet50.preprocess_input(np.asarray(img, dtype=np.float32))
    probs = model.predict(x[np.newaxis, ...], verbose=0)[0]
    idx = int(np.argmax(probs))
    return idx, float(probs[idx] * 100)

# ==========================================
# SIDEBAR + ADMIN UPLOAD
# ==========================================
st.sidebar.title("LeukNet AML")
page = st.sidebar.radio("Navigation",
                        ["Dashboard", "Patient Register", "AML Detector", "Reports Manager"])
model, source = get_model()

st.sidebar.markdown("---")
st.sidebar.subheader("Model")
if st.session_state.get("flash"):
    st.sidebar.success(st.session_state.pop("flash"))
if model is not None:
    mb = os.path.getsize(MODEL_PATH) / 1e6
    saved = datetime.fromtimestamp(os.path.getmtime(MODEL_PATH)).strftime("%b %d, %Y %H:%M")
    st.sidebar.success(f"Live model ({mb:.0f} MB)\n\nSource: {source}\n\nSaved: {saved}")
elif source == "broken":
    st.sidebar.error("Saved model file is damaged. Upload it again.")
else:
    st.sidebar.info("No model saved yet. Running in DEMO mode.")

with st.sidebar.expander("Admin: upload / replace model"):
    if not ADMIN_PASSWORD:
        st.info("Set the ADMIN_PASSWORD environment variable on Render to enable uploads.")
    else:
        pw = st.text_input("Admin password", type="password")
        if pw and not hmac.compare_digest(pw.encode(), ADMIN_PASSWORD.encode()):
            st.error("Wrong password.")
        elif pw:
            up = st.file_uploader("Choose your .keras model file", type=["keras"])
            if up and st.button("Validate and save model"):
                tmp = os.path.join(DATA_DIR, "upload.tmp.keras")
                with st.spinner("Checking and saving the model..."):
                    with open(tmp, "wb") as f:
                        f.write(up.getbuffer())
                    err = check_model(tmp)
                    if err:
                        os.remove(tmp)
                        st.error(err)
                    else:
                        if os.path.exists(MODEL_PATH):
                            shutil.copyfile(MODEL_PATH, MODEL_PATH + ".bak")
                        os.replace(tmp, MODEL_PATH)
                        backup_err = push_to_hub(MODEL_PATH)
                        st.cache_resource.clear()
                        st.session_state["flash"] = (
                            "Model saved. " + (backup_err or "Backed up to Hugging Face."))
                        st.rerun()

# ==========================================
# PAGES
# ==========================================
if page == "Dashboard":
    st.title("Dashboard")
    st.write("Welcome to the LeukNet AML Diagnostic Platform.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Patients", len(st.session_state.patients))
    c2.metric("Reports", len(st.session_state.reports))
    c3.metric("Mode", "Live model" if model is not None else "Demo")
    st.info("**AML Detector**\n\nResNet50 classifies single blood cells as monocyte or myeloblast.")

elif page == "Patient Register":
    st.title("Patient Register")
    with st.form("patient_form", clear_on_submit=True):
        name = st.text_input("Patient Name")
        col1, col2 = st.columns(2)
        with col1:
            gender = st.selectbox("Gender", ["Male", "Female"])
            age = st.number_input("Age", min_value=0, max_value=120, value=12)
        with col2:
            address = st.text_input("Address")
            contact = st.text_input("Contact")
        submit = st.form_submit_button("Add Patient")
    if submit:
        if not name.strip():
            st.error("Please enter the patient name.")
        else:
            st.session_state.patients.append({
                "Name": name.strip().upper(), "Gender": gender, "Age": age,
                "Address": address, "Contact": contact})
            st.success(f"Patient {name.strip().upper()} added.")
    if st.session_state.patients:
        st.subheader("Registered Patients")
        st.dataframe(pd.DataFrame(st.session_state.patients), use_container_width=True)

elif page == "AML Detector":
    st.title("AML Detector (ResNet50)")
    if model is None:
        st.warning("DEMO MODE: no model loaded. Results below are SIMULATED "
                   "and are not a real diagnosis.")
    names = [p["Name"] for p in st.session_state.patients]
    patient = st.selectbox("For Which Patient", names) if names else "UNREGISTERED"

    files = st.file_uploader("Upload single-cell images here",
                             type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"],
                             accept_multiple_files=True)
    if files:
        st.image([Image.open(f).convert("RGB") for f in files],
                 caption=[f"Sample {i+1}" for i in range(len(files))], width=140)
        if st.button("Process Samples"):
            with st.spinner("Processing images..."):
                rows = []
                for f in files:
                    f.seek(0)
                    raw = f.read()
                    f.seek(0)
                    idx, conf = predict(model, Image.open(f), raw)
                    rows.append({"File": f.name, "Prediction": CLASS_LABELS[idx],
                                 "Confidence (%)": round(conf, 1), "Blast": idx == 1})
                st.session_state.last_result = {
                    "patient": patient, "rows": rows,
                    "time": datetime.now().strftime("%b %d, %Y %I:%M %p")}

    res = st.session_state.last_result
    if res:
        rows = res["rows"]
        blasts = sum(r["Blast"] for r in rows)
        st.subheader("Results" + (" (DEMO - simulated)" if model is None else ""))
        st.write(f"**Patient:** {res['patient']}  |  **Samples:** {len(rows)}  |  "
                 f"**Classified as myeloblast:** {blasts}")
        st.dataframe(pd.DataFrame(rows).drop(columns="Blast"), use_container_width=True)
        if blasts:
            st.markdown(f"""<div class="report-box red"><h4>Results Report</h4>
                <p>{blasts} of {len(rows)} cell(s) classified as myeloblast. AML may be
                indicated. Please confirm with a haematologist.</p></div>""",
                        unsafe_allow_html=True)
        else:
            st.markdown("""<div class="report-box"><h4>Results Report</h4>
                <p>No myeloblasts detected in the uploaded cells.</p></div>""",
                        unsafe_allow_html=True)
        if st.button("Save Results to Patient Record"):
            st.session_state.reports.append({
                "Date": res["time"], "Patient Name": res["patient"],
                "Result": f"AML indicated ({blasts} blast(s))" if blasts else "No blasts detected",
                "Samples": len(rows)})
            st.success("Saved to Reports Manager.")
    st.caption("Research prototype. Not a certified medical device. "
               "Results must be confirmed by a qualified professional.")

elif page == "Reports Manager":
    st.title("Reports Manager")
    if st.session_state.reports:
        df = pd.DataFrame(st.session_state.reports)
        st.dataframe(df, use_container_width=True)
        st.download_button("Download Reports (CSV)", df.to_csv(index=False).encode("utf-8"),
                           file_name="aml_reports.csv", mime="text/csv")
    else:
        st.info("No saved reports yet.")
