import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av
import tempfile
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

# Konfigurasi halaman
st.set_page_config(page_title="YOLO Object Detection", page_icon="🎯", layout="wide")

# Load model YOLO
model_path = "best.pt"
model = YOLO(model_path)

# Header
st.markdown("""
    <h1 style='text-align: center; color: #2C3E50;'>🎯 YOLO Object Detection</h1>
    <p style='text-align: center; font-size: 16px;'>Deteksi objek secara real-time dari gambar, video, atau webcam dengan model YOLO custom.</p>
    <hr style='border: 1px solid #ccc;'/>
""", unsafe_allow_html=True)

# Sidebar pilihan mode
st.sidebar.header("📌 Pilih Input")
option = st.sidebar.radio("Metode Input:", ("📷 Gambar", "🎞️ Video", "🎥 Webcam"))

# ===== Fungsi Prediksi Gambar =====
def predict_image(image):
    results = model.predict(image, conf=0.25)
    result_img = results[0].plot()
    return result_img

# ===== Fungsi Prediksi Video =====
def predict_video(video_path):
    cap = cv2.VideoCapture(video_path)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    temp_output = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    out = cv2.VideoWriter(temp_output.name, fourcc, fps, (width, height))

    # Progress UI
    progress_text = st.empty()
    progress_bar = st.progress(0)

    frame_count = 0
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        results = model.predict(frame)
        annotated_frame = results[0].plot()
        out.write(annotated_frame)

        frame_count += 1
        if total_frames > 0:
            percent = int((frame_count / total_frames) * 100)
            progress_bar.progress(min(percent, 100))
            progress_text.text(f"📦 Memproses frame {frame_count} / {total_frames}...")
        else:
            progress_text.text(f"📦 Memproses frame {frame_count}...")

    cap.release()
    out.release()

    progress_text.text("✅ Deteksi selesai!")
    progress_bar.empty()

    return temp_output.name

# ===== Kelas Webcam =====
class YOLOProcessor(VideoProcessorBase):
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        results = model.predict(img, conf=0.1, verbose=False)
        if results and len(results[0].boxes) > 0:
            annotated = results[0].plot()
        else:
            annotated = img
        return av.VideoFrame.from_ndarray(annotated.astype(np.uint8), format="bgr24")

# ====== MODE: GAMBAR ======
if option.startswith("📷"):
    st.subheader("📷 Deteksi Objek dari Gambar")
    st.write("Unggah gambar berformat `.jpg`, `.jpeg`, atau `.png` lalu hasil deteksi akan ditampilkan.")

    uploaded_image = st.file_uploader("Upload Gambar", type=["jpg", "jpeg", "png"])
    if uploaded_image:
        image = Image.open(uploaded_image)
        st.image(image, caption="📥 Gambar yang diupload", use_container_width=True)
        with st.spinner("🔍 Mendeteksi objek..."):
            result_image = predict_image(np.array(image))
        st.image(result_image, caption="🎯 Hasil Deteksi", use_container_width=True)

# ====== MODE: VIDEO ======
elif option.startswith("🎞️"):
    st.subheader("🎞️ Deteksi Objek dari Video")
    st.write("Unggah file video berformat `.mp4` atau `.mov`, kemudian unduh hasilnya setelah diproses.")

    uploaded_video = st.file_uploader("Upload Video", type=["mp4", "mov"])
    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        tfile.flush()
        tfile.close()

        st.video(tfile.name, format="video/mp4")

        with st.spinner("⏳ Melakukan deteksi pada video..."):
            result_video_path = predict_video(tfile.name)

        with open(result_video_path, "rb") as f:
            video_bytes = f.read()

        st.success("✅ Deteksi selesai!")
        st.download_button(
            label="⬇️ Download Video Hasil Deteksi",
            data=video_bytes,
            file_name="video_deteksi_yolo.mp4",
            mime="video/mp4"
        )

# ====== MODE: WEBCAM ======
elif option.startswith("🎥"):
    st.subheader("🎥 Deteksi Objek dari Webcam (Real-time)")
    st.markdown("Klik 'Allow' saat browser meminta izin webcam.")

    rtc_config = {
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    }

    webrtc_ctx = webrtc_streamer(
        key="yolo-webcam",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=YOLOProcessor,
        media_stream_constraints={"video": True, "audio": False},
        rtc_configuration=rtc_config,
        async_processing=True,
    )

    if webrtc_ctx.video_processor:
        st.success("🟢 Webcam aktif dan YOLO berjalan!")
    elif webrtc_ctx.state.playing:
        st.info("⏳ Menginisialisasi webcam...")
    else:
        st.warning("🔴 Webcam belum aktif atau tidak terdeteksi.")
