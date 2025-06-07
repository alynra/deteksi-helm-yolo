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
option = st.sidebar.radio("Pilih metode input:", ("Gambar", "Video", "Webcam"))

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

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    temp_output = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    out = cv2.VideoWriter(temp_output.name, fourcc, fps, (width, height))

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        results = model.predict(frame)
        annotated_frame = results[0].plot()
        out.write(annotated_frame)

    cap.release()
    out.release()
    return temp_output.name

# ===== Kelas Webcam (streamlit-webrtc) =====
class YOLOProcessor(VideoProcessorBase):
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        results = model.predict(img)
        annotated = results[0].plot()
        return av.VideoFrame.from_ndarray(annotated, format="bgr24")

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
        st.video(tfile.name)

        with st.spinner("Melakukan deteksi pada video..."):
            result_video_path = predict_video(tfile.name)
            st.success("Deteksi selesai!")

        st.video(result_video_path)

# ====== MODE: WEBCAM ======
elif option == "Webcam":
    st.subheader("Deteksi Objek dari Webcam (Real-time)")
    st.markdown("Klik 'Allow' saat browser meminta izin webcam.")
    webrtc_streamer(
        key="yolo-webcam",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=YOLOProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
