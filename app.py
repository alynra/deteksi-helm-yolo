import streamlit as st
from ultralytics import YOLO
import tempfile
import cv2
import os
from PIL import Image
import numpy as np
import time

# Judul Aplikasi
st.title("YOLO Object Detection (Gambar, Video, Webcam)")
st.write("Upload gambar, video, atau gunakan webcam untuk mendeteksi objek menggunakan model YOLO.")

# Load model YOLO
model_path = "best.pt"  # Ganti ke path custom model jika diperlukan
model = YOLO(model_path)

# Sidebar untuk memilih jenis input
option = st.sidebar.radio("Pilih jenis input:", ("Gambar", "Video", "Webcam"))

# Fungsi untuk prediksi gambar
def predict_image(image):
    results = model.predict(image)
    result_img = results[0].plot()
    return result_img

# Fungsi untuk prediksi video
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

# Fungsi untuk prediksi webcam
def predict_webcam():
    stframe = st.empty()
    cap = cv2.VideoCapture(0)

    st.info("Menekan 'Stop' atau tutup tab untuk menghentikan webcam.")
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            st.warning("Tidak bisa mengakses webcam.")
            break

        # Deteksi dengan YOLO
        results = model.predict(frame)
        annotated_frame = results[0].plot()

        # Konversi ke RGB & tampilkan
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        stframe.image(annotated_frame, channels="RGB", use_container_width=True)

        # Hentikan dengan tombol
        if st.button("Stop Webcam"):
            break

    cap.release()
    cv2.destroyAllWindows()

# Untuk input Gambar
if option == "Gambar":
    uploaded_image = st.file_uploader("Upload gambar (jpg/jpeg/png)", type=["jpg", "jpeg", "png"])
    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption="Gambar yang diupload", use_container_width=True)
        result_image = predict_image(np.array(image))
        st.image(result_image, caption="Hasil Deteksi", use_container_width=True)

# Untuk input Video
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

# Untuk input Webcam
elif option == "Webcam":
    st.subheader("Webcam Live Detection")
    start = st.button("Mulai Webcam")
    if start:
        predict_webcam()
