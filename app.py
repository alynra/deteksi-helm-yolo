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
            margin-top: 80px;
            background-image: url('https://images.unsplash.com/photo-1571867424485-3694642b2f73'); 
            background-size: cover;
            background-position: center;
            padding: 120px 30px;
            border-radius: 8px;
            text-align: center;
            color: black;
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
        .prediksi-container {
            max-width: 700px;
            margin: 2rem auto;
            background-color: white;
            padding: 2rem;
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.06);
        }
        div[id="prediksi-anchor"] + div {
            max-width: 700px;
            margin: 2rem auto;
            background-color: white;
            padding: 2rem;
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.06);
        }
    </style>

    <script>
        function scrollToPrediksi() {
            const el = document.getElementById("prediksi");
            if (el) el.scrollIntoView({ behavior: "smooth" });
        }
    </script>

    
    <div class="navbar">
        <div style="font-weight: bold; font-size: 1.2rem;">Deteksi Helm</div>
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
        <h1>Deteksi Penggunaan Helm Pada Pengendara Motor</h1>
        <p>AI untuk mendeteksi penggunaan helm pada pengendara motor demi keamanan bersama dengan cepat dan canggih.</p>
        <button onclick="scrollToPrediksi()">Mulai Deteksi</button>
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

# ======= Kelas Webcam (streamlit-webrtc) =====
class YOLOProcessor(VideoProcessorBase):
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        results = model.predict(img, conf=0.1, verbose=False)
        if results and len(results[0].boxes) > 0:
            annotated = results[0].plot()
        else:
            annotated = img
        return av.VideoFrame.from_ndarray(annotated.astype(np.uint8), format="bgr24")

# ==== Bagian Prediksi ====
st.markdown("<div id='prediksi'></div>", unsafe_allow_html=True)
st.markdown('<div id="prediksi-anchor"></div>', unsafe_allow_html=True)
col_left, col_center, col_right = st.columns([1, 2, 1])

with col_center:
    with st.container():
        st.markdown("""
            <div class="prediksi-container" style="background-color: white; padding: 2rem; border-radius: 12px;
                        box-shadow: 0 4px 12px rgba(0,0,0,0.06);">
        """, unsafe_allow_html=True)
        st.header("Prediksi Penggunaan Helm Pada Pengendara Motor")
        
        #option = st.radio("Pilih metode input:", ["Gambar", "Video", "Webcam"], horizontal=True)
        
        if option == "Gambar":
            uploaded_image = st.file_uploader("Upload gambar (jpg/jpeg/png)", type=["jpg", "jpeg", "png"])
            if uploaded_image is not None:
                image = Image.open(uploaded_image)
                st.image(image, caption="Gambar yang diupload", use_container_width=True)
                result_image = predict_image(np.array(image))
                st.image(result_image, caption="Hasil Deteksi", use_container_width=True)
        
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
        
        elif option == "Webcam":
            st.subheader("Deteksi Objek dari Webcam (Real-time)")
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
                st.success("Webcam berhasil terhubung dan model YOLO aktif!")
            elif webrtc_ctx.state.playing:
                st.info("Menginisialisasi webcam...")
            else:
                st.warning("Webcam belum aktif atau tidak terdeteksi.")
        
        st.markdown('</div>', unsafe_allow_html=True)
