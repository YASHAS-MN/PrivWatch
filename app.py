import streamlit as st
import tempfile
import cv2
import os
import time
from pathlib import Path
import psutil

from privwatch.privacy_engine import privacy_engine
from privwatch.raw_engine import raw_engine

WATCH_DIR = Path("./temp_monitor")
WATCH_DIR.mkdir(exist_ok=True)

def list_files():
    return set(str(p.resolve()) for p in WATCH_DIR.glob("*") if p.is_file())

def snapshot():
    return {"files": list_files(), "time": time.time()}

def diff_files(before, after):
    return sorted(list(after["files"] - before["files"]))

def io_snapshot():
    io = psutil.disk_io_counters()
    return {"read_bytes": io.read_bytes, "write_bytes": io.write_bytes}

def io_diff(before, after):
    return {
        "read_bytes": after["read_bytes"] - before["read_bytes"],
        "write_bytes": after["write_bytes"] - before["write_bytes"]
    }

st.set_page_config(page_title="PrivWatch", layout="centered")
st.title("PrivWatch Demo")

uploaded_file = st.file_uploader("Upload Video", type=["mp4", "avi"])
model_choice = st.radio("Select Model", ["RAW", "PRIVACY"])

if uploaded_file is not None:

    # SAVE FILE ONCE
    if "video_path" not in st.session_state:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tmp.write(uploaded_file.read())
        tmp.close()
        st.session_state.video_path = tmp.name

    video_path = st.session_state.video_path

    # READ VIDEO INTO MEMORY FOR DISPLAY (NO LOCK AFTER THIS)
    if "video_bytes" not in st.session_state:
        with open(video_path, "rb") as f:
            st.session_state.video_bytes = f.read()

    st.video(st.session_state.video_bytes)

    st.info("Video loaded. Ready for processing.")

    if st.button("Run Detection"):

        # Clean monitor folder
        for f in WATCH_DIR.glob("*"):
            try:
                f.unlink()
            except:
                pass

        files_before = snapshot()
        io_before = io_snapshot()

        st.info("Processing...")

        # OPEN VIDEO (ONLY HERE)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError("Failed to open video")

        # RUN MODEL
        if model_choice == "RAW":
            result, latency, stored = raw_engine(cap)
        else:
            result, latency, stored = privacy_engine(cap)

        cap.release()

        # 🔥 FORCE DELETE WITH RETRY (WINDOWS SAFE)
        for _ in range(10):
            try:
                if os.path.exists(video_path):
                    os.remove(video_path)
                    break
            except PermissionError:
                time.sleep(0.2)

        # CLEAN STATE
        if "video_path" in st.session_state:
            del st.session_state.video_path
        if "video_bytes" in st.session_state:
            del st.session_state.video_bytes

        files_after = snapshot()
        io_after = io_snapshot()

        new_files = diff_files(files_before, files_after)
        io_change = io_diff(io_before, io_after)

        st.success("Done")

        st.subheader("Result")
        st.write(result)

        st.subheader("Latency")
        st.write(f"{latency:.2f} seconds")

        st.subheader("Privacy Monitor")

        if len(new_files) == 0:
            st.success("No new files created during processing")
        else:
            st.error(f"New files created: {new_files[:5]}")

        st.write(f"Disk Write (bytes): {io_change['write_bytes']}")
        st.write(f"Disk Read (bytes): {io_change['read_bytes']}")

        if stored:
            st.error("RAW: Disk activity possible")
        else:
            if len(new_files) == 0:
                st.success("PRIVACY: RAM-only processing confirmed")
            else:
                st.error("PRIVACY: Unexpected disk activity")