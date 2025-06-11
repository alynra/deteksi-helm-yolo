import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av
import tempfile
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

# Load model
model = YOLO("best.pt")

# ==== CSS dan Layout ====
st.set_page_config(page_title="Deteksi Helm Pengendara Motor", layout="wide")

st.markdown("""
    <style>
        header[data-testid="stHeader"] {
            display: none;
        }
        html, body {
            margin: 0 !important;
            padding: 0 !important;
            scroll-behavior: smooth;
        }

        section.main > div.block-container {
                padding-top: 0 !important;
                margin-top: 0 !important;
                padding: 0rem;
                max-width: 100%;
                width: 100%;
            }
        div.block-container {
                padding: 0 !important;
                margin: 0 auto !important;
                width: 100% !important;
                max-width: 100% !important;
            }
        
            
        *, *::before, *::after {
            box-sizing: border-box;
        }
        

        
        .navbar, .hero {
            max-width: 100%;
            width: 100%;
            margin: 0 auto;
        }
    
        /* Pastikan tombol tidak overflow */
        .hero button {
            max-width: 90%;
            width: auto;
        }
    
        /* Responsif untuk tampilan mobile */
        @media screen and (max-width: 768px) {
            .hero h1 {
                font-size: 1.8em;
            }
            .hero p {
                font-size: 1em;
            }
            .navbar {
                flex-direction: column;
                align-items: flex-start;
            }
        }
        .hero {
            display: flex;
            flex-direction: row;
            align-items: center;
            justify-content: space-between;
            padding: 100px 60px;
            background-color: white;
            border-radius: 0;
            margin: 0;
            gap: 2rem;
        }
        
        .hero-left {
            flex: 1;
            max-width: 600px;
        }
        
        .hero-left h1 {
            font-size: 3.2em;
            font-weight: bold;
            color: #222;
            margin-bottom: 1rem;
            line-height: 1.2;
        }
        
        .hero-left p {
            font-size: 1.2em;
            color: #444;
            margin-bottom: 2rem;
        }
        
        .hero-left button {
            background-color: #c80114;
            color: white;
            padding: 0.8em 2em;
            border: none;
            font-size: 1em;
            border-radius: 10px;
            cursor: pointer;
        }
        
        .hero-left button:hover {
            background-color: #e2e8f0;
            color: #111;
        }
        
        .hero-right {
            flex: 1;
            display: flex;
            justify-content: center;
            align-items: center;
        }
        
        .hero-right img {
            width: 100%;
            max-width: 500px;
            height: auto;
        }

        .hero h1 {
            font-size: 3em;
            font-weight: bold;
        }
        .hero p {
            font-size: 1.2em;
            margin-top: 1rem;
            margin-bottom: 2rem;
        }
        .hero button {
            background-color: #c80114;
            color: white;
            padding: 0.8em 2em;
            border: none;
            font-size: 1em;
            border-radius: 10px;
            cursor: pointer;
        }
        .hero button:hover {
            background-color: #e2e8f0;
        }
        .navbar {
            position: fixed;
            top: 0;
            left: 0;
            z-index: 9999;
            width: 100%;
            background-color: #8a000d;
            padding: 1rem 2rem;
            border-bottom: 1px solid #e5e7eb;
            display: flex;
            justify-content: space-between;
            align-items: center;
            color: white;
        }
        .navbar a {
            margin: 0 1rem;
            text-decoration: none;
            color: white;
            font-weight: 500;
        }
        .navbar a:hover {
            color: #e2e8f0;
        }

        .dropdown-content .nav-item {
            color: #333;
            padding: 10px 16px;
            text-decoration: none;
            display: block;
            cursor: pointer;
        }
        
        .dropdown-content .nav-item:hover {
            background-color: #f1f1f1;
        }

        .nav-right {
            margin-right: 2rem;
            display: flex;
            align-items: center;
            gap: 1rem;
        }
        
        .dropdown {
            position: relative;
            display: inline-block;
        }
        
        .dropbtn {
            color: white;
            text-decoration: none;
            font-weight: 500;
            cursor: pointer;
        }
        
        .dropdown-content {
            display: none;
            position: absolute;
            background-color: white;
            min-width: 140px;
            box-shadow: 0px 8px 16px rgba(0,0,0,0.2);
            z-index: 10000;
            border-radius: 4px;
            overflow: hidden;
        }
        
        .dropdown-content a {
            color: #333;
            padding: 10px 16px;
            text-decoration: none;
            display: block;
        }
        
        .dropdown-content a:hover {
            background-color: #f1f1f1;
        }
        
        .dropdown:hover .dropdown-content {
            display: block;
        }
        
        div[id="prediksi-anchor"] + div {
            background-color: transparent;
            padding: 2rem;
            box-shadow: none;
            margin: 0 auto;
            max-width: 100%;
            display: flex;
            flex-direction: column;
            justify-content: center;
        }
        #prediksi {
            padding: 50px 60px;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            gap: 2rem;
            margin: 0 auto;
        }
        #prediksi h1 {
            font-size: 2em;
            font-weight: bold;
            text-align: center;
        }
        #prediksi p {
            font-size: 1.2em;
            margin-top: 1rem;
            text-align: center;
        }
    </style>

    
    <div class="navbar">
        <div>
            <img src="https://raw.githubusercontent.com/alynra/deteksi-helm-yolo/main/helmviz_2.png"
             alt="Helmviz Logo"
             style="height: 32px; width: auto; object-fit: contain;">
        </div>
        <div class="nav-right">
            <a href="#beranda">Beranda</a>
                <div class="dropdown">
                  <div class="dropbtn">Prediksi ▾</div>
                  <div class="dropdown-content">
                    <a href="?mode=Gambar#prediksi-anchor" target="_self">Gambar</a>
                    <a href="?mode=Video#prediksi-anchor" target="_self">Video</a>
                    <a href="?mode=Webcam#prediksi-anchor" target="_self">Webcam</a>
                  </div>
                </div>
        </div>
    </div>
    

    <div class="hero" id="beranda">
        <div class="hero-left">
            <h1>Deteksi Penggunaan Helm<br>Pada Pengendara Motor</h1>
            <p>AI untuk mendeteksi penggunaan helm demi keamanan berkendara dengan cepat dan akurat.</p>
            <a href="#prediksi-anchor">
              <button>Mulai Deteksi</button>
            </a>
        </div>
        <div class="hero-right">
            <img src="https://raw.githubusercontent.com/alynra/deteksi-helm-yolo/main/animasi.jpg" alt="Animasi Ilustrasi">
        </div>
    </div>

""", unsafe_allow_html=True)

if "mode" not in st.session_state:
    st.session_state["mode"] = "Gambar"

query_params = st.query_params
if "mode" in query_params:
    st.session_state["mode"] = query_params["mode"]


    
option = st.session_state["mode"]

# ===== Fungsi Prediksi Gambar =====
def predict_image(image):
    progress_text = st.empty()
    progress_bar = st.progress(0)
    
    # Simulasi proses bertahap (karena hanya 1 gambar, progress disimulasikan)
    progress_text.text("Memuat gambar...")
    progress_bar.progress(10)
    
    # Proses deteksi
    results = model.predict(image, imgsz=640)
    progress_text.text("Melakukan deteksi...")
    progress_bar.progress(70)
    
    result_img = results[0].plot()
    progress_bar.progress(100)
    progress_text.text("Selesai!")
    
    # Bersihkan progress bar
    progress_bar.empty()

    return result_img

# ===== Fungsi Prediksi Video =====
def predict_video(video_path):
    cap = cv2.VideoCapture(video_path)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps == 0:
        fps = 25

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    temp_output = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    out = cv2.VideoWriter(temp_output.name, fourcc, fps, (width, height))

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
            progress = int((frame_count / total_frames) * 100)
            progress_bar.progress(min(progress, 100))
            progress_text.text(f"Memproses frame {frame_count} / {total_frames}...")
        else:
            progress_text.text(f"Memproses frame {frame_count}...")

    cap.release()
    out.release()

    progress_text.text("Selesai!")
    progress_bar.empty()

    return temp_output.name

def resize_image(image_pil, max_size=1024):
    w, h = image_pil.size
    if max(w, h) > max_size:
        ratio = max_size / float(max(w, h))
        new_size = (int(w * ratio), int(h * ratio))
        try:
            resample_mode = Image.Resampling.LANCZOS
        except AttributeError:
            resample_mode = Image.LANCZOS  # kompatibel dengan Pillow < 10
        image_pil = image_pil.resize(new_size, resample_mode)
    return image_pil
# ======= Kelas Webcam (streamlit-webrtc) =====
class YOLOProcessor(VideoProcessorBase):
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        try:
            img = frame.to_ndarray(format="bgr24")
            print("[DEBUG] Menerima frame dari webcam")

            results = model.predict(img, conf=0.1, verbose=False)

            # Cek aman
            if results and hasattr(results[0], 'boxes') and results[0].boxes is not None:
                print("[DEBUG] Jumlah deteksi:", len(results[0].boxes))
                if len(results[0].boxes) > 0:
                    annotated = results[0].plot()
                else:
                    annotated = img
            else:
                print("[DEBUG] Tidak ada hasil deteksi")
                annotated = img

            return av.VideoFrame.from_ndarray(annotated.astype(np.uint8), format="bgr24")

        except Exception as e:
            print("❌ ERROR recv():", e)
            return frame


# ==== Bagian Prediksi ====
st.markdown("""
<div id='prediksi' style='min-height: 30vh; text-align: center; padding: 20px;'>
    <h1 style='margin-bottom: 0.5em;'>Prediksi Penggunaan Helm Pada Pengendara Motor</h1>
    <p style='margin-top: 0; font-size: 1.1rem; color: #444;'>Anda dapat memprediksi motor, pengguna helm dan non-helm dari <br>gambar, video ataupun secara real-time menggunakan webcam di sini.</p>
</div>
""", unsafe_allow_html=True)
st.markdown('<div id="prediksi-anchor"></div>', unsafe_allow_html=True)
col_left, col_center, col_right = st.columns([1, 3, 1])

with col_center:
    with st.container():
        
        #st.header("Prediksi Penggunaan Helm Pada Pengendara Motor")
        
        #option = st.radio("Pilih metode input:", ["Gambar", "Video", "Webcam"], horizontal=True)
        
        if option == "Gambar":
            uploaded_image = st.file_uploader("Upload gambar (jpg/jpeg/png)", type=["jpg", "jpeg", "png"])
            
            if uploaded_image is not None:
                image = Image.open(uploaded_image)
                image = resize_image(image, max_size=640)
                #image_np = np.array(image)
        
                st.image(image, caption="Gambar yang diupload", use_container_width=True)
        
                with st.spinner("Melakukan deteksi..."):
                    result_image = predict_image(image)
                    st.success("Deteksi selesai!")
        
                st.image(result_image, caption="Hasil Deteksi", use_container_width=True)
            st.markdown("<div style='height: 80px;'></div>", unsafe_allow_html=True)
        
        elif option == "Video":
            uploaded_video = st.file_uploader("Upload video (mp4/mov)", type=["mp4", "mov"])
            if uploaded_video is not None:
                tfile = tempfile.NamedTemporaryFile(delete=False)
                tfile.write(uploaded_video.read())
                tfile.flush()
                tfile.close()
        
                st.video(tfile.name)
        
                with st.spinner("Melakukan deteksi pada video..."):
                    result_video_path = predict_video(tfile.name)
                    st.success("Deteksi selesai!")
        
                with open(result_video_path, "rb") as f:
                    video_bytes = f.read()
        
                st.download_button(
                    label="⬇ Download Video Hasil Deteksi",
                    data=video_bytes,
                    file_name="video_deteksi_yolo.mp4",
                    mime="video/mp4"
                )
            st.markdown("<div style='height: 80px;'></div>", unsafe_allow_html=True)
        
        elif option == "Webcam":
            st.subheader("Deteksi Objek dari Webcam (Real-time)")
            st.markdown("Klik 'Allow' saat browser meminta izin webcam.")
        
            rtc_config = {
                "iceServers": [
                    {"urls": ["stun:stun.l.google.com:19302"]},
                    {
                        "urls": ["turn:openrelay.metered.ca:80"],
                        "username": "openrelayproject",
                        "credential": "openrelayproject"
                    }
                ]
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
                st.success("Webcam berhasil terhubung dan model YOLO aktif!")
            elif webrtc_ctx.state.playing:
                st.info("Menginisialisasi webcam...")
            else:
                st.warning("Webcam belum aktif atau tidak terdeteksi.")
                
            st.markdown("<div style='height: 80px;'></div>", unsafe_allow_html=True)
        #st.markdown('</div>', unsafe_allow_html=True)
