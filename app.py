import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av
import tempfile
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

# Load model YOLO
model_path = "best.pt"  # Ganti dengan custom model jika perlu
model = YOLO(model_path)

# Judul aplikasi
st.title("YOLO Object Detection: Gambar, Video, Webcam")
st.write("Deteksi objek menggunakan model YOLO dari gambar, video, atau webcam secara real-time.")

# Sidebar: Pilihan mode input
#option = st.sidebar.radio("Pilih metode input:", ("Gambar", "Video", "Webcam"))

# ===== Fungsi Prediksi Gambar =====
def predict_image(image):
    results = model.predict(image)
    result_img = results[0].plot()
    return result_img

# ===== Fungsi Prediksi Video =====
def predict_video(video_path):
    cap = cv2.VideoCapture(video_path)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps == 0:
        fps = 25  # fallback

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    temp_output = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    out = cv2.VideoWriter(temp_output.name, fourcc, fps, (width, height))

    # Streamlit progress bar
    progress_text = st.empty()
    progress_bar = st.progress(0)

    frame_count = 0
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Proses YOLO
        results = model.predict(frame)
        annotated_frame = results[0].plot()
        out.write(annotated_frame)

        # Update progress
        frame_count += 1
        if total_frames > 0:
            progress = int((frame_count / total_frames) * 100)
            progress_bar.progress(min(progress, 100))
            progress_text.text(f"Memproses frame {frame_count} / {total_frames}...")
        else:
            progress_text.text(f"Memproses frame {frame_count}...")

        # Debug ke log terminal
        if frame_count % 10 == 0:
            print(f"[DEBUG] Frame ke-{frame_count} diproses")

    cap.release()
    out.release()

    progress_text.text("Selesai!")
    progress_bar.empty()

    return temp_output.name



# ===== Kelas Webcam (streamlit-webrtc) =====
class YOLOProcessor(VideoProcessorBase):
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        print("[DEBUG] Menerima frame dari webcam")
        results = model.predict(img, conf=0.1, verbose=False)
        print("[DEBUG] Jumlah deteksi:", len(results[0].boxes))
        if results and len(results[0].boxes) > 0:
            annotated = results[0].plot()
        else:
            annotated = img  # Jika tidak ada deteksi, tampilkan frame asli

        # Kembalikan frame
        return av.VideoFrame.from_ndarray(annotated.astype(np.uint8), format="bgr24")

if "page" not in st.session_state:
    st.session_state.page = "Gambar"

# Tangkap klik melalui query parameter (?page=...)
query_params = st.query_params
if "page" in query_params:
    st.session_state.page = query_params.get("page", "Gambar")

# ==== CSS Styling Sidebar ====
st.markdown("""
    <style>
    section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"] {
        align-items: flex-start;
        padding-top: 1rem;
    }
    .nav-title {
        font-size: 1.2rem;
        font-weight: bold;
        margin-bottom: 1rem;
        color: #1f2937;
    }
    .nav-item {
        font-size: 1rem;
        font-weight: 500;
        padding: 0.6rem 1rem;
        width: 100%;
        display: block;
        color: #374151;
        border-bottom: 1px solid #e5e7eb;
        text-decoration: none;
    }
    .nav-item:hover {
        background-color: #f3f4f6;
        color: #111827;
    }
    .nav-active {
        background-color: #e5e7eb;
        font-weight: bold;
        color: #5B2EFF;
    }
    </style>
""", unsafe_allow_html=True)

# === Sidebar Navigasi (HTML links only) ===
st.sidebar.markdown("<div class='nav-title'>🧭 Menu Navigasi</div>", unsafe_allow_html=True)

def nav_link(label, icon, page_name):
    is_active = st.session_state.page == page_name
    nav_class = "nav-item nav-active" if is_active else "nav-item"
    url = f"?page={page_name}"
    st.sidebar.markdown(f"<a href='{url}' target='_self' class='{nav_class}'>{icon} {label}</a>", unsafe_allow_html=True)

# Navigasi menu
nav_link("Gambar", "🖼️", "Gambar")
nav_link("Video", "🎞️", "Video")
nav_link("Webcam", "📷", "Webcam")

# Tombol navigasi
#if st.sidebar.button("Gambar"):
#    st.session_state.page = "Gambar"
#if st.sidebar.button("Video"):
#    st.session_state.page = "Video"
#if st.sidebar.button("Webcam"):
#    st.session_state.page = "Webcam"

# ======= KONTEN BERDASARKAN NAVIGASI =======
option = st.session_state.page

# ====== MODE: GAMBAR ======
if option == "Gambar":
    uploaded_image = st.file_uploader("Upload gambar (jpg/jpeg/png)", type=["jpg", "jpeg", "png"])
    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption="Gambar yang diupload", use_container_width=True)
        result_image = predict_image(np.array(image))
        st.image(result_image, caption="Hasil Deteksi", use_container_width=True)

# ====== MODE: VIDEO ======
elif option == "Video":
    uploaded_video = st.file_uploader("Upload video (mp4/mov)", type=["mp4", "mov"])
    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        tfile.flush()
        tfile.close()

        st.video(tfile.name)  # Menampilkan video asli (optional)

        with st.spinner("Melakukan deteksi pada video..."):
            result_video_path = predict_video(tfile.name)
            st.success("Deteksi selesai!")

        # Baca kembali file hasil
        with open(result_video_path, "rb") as f:
            video_bytes = f.read()

        st.download_button(
            label="⬇ Download Video Hasil Deteksi",
            data=video_bytes,
            file_name="video_deteksi_yolo.mp4",
            mime="video/mp4"
        )


# ====== MODE: WEBCAM ======
elif option == "Webcam":
    st.subheader("Deteksi Objek dari Webcam (Real-time)")
    st.markdown("Klik 'Allow' saat browser meminta izin webcam.")

    # Konfigurasi RTC (penting untuk deployment/non-local)
    rtc_config = {
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    }

    # Jalankan webrtc streamer
    webrtc_ctx = webrtc_streamer(
        key="yolo-webcam",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=YOLOProcessor,
        media_stream_constraints={"video": True, "audio": False},
        rtc_configuration=rtc_config,
        async_processing=True,
    )

    # Tambahkan status deteksi
    if webrtc_ctx.video_processor:
        st.success("Webcam berhasil terhubung dan model YOLO aktif!")
    elif webrtc_ctx.state.playing:
        st.info("Menginisialisasi webcam...")
    else:
        st.warning("Webcam belum aktif atau tidak terdeteksi.")
