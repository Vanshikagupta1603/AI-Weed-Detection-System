import streamlit as st
from ultralytics import YOLO
from PIL import Image
import pandas as pd
import cv2
import tempfile
import plotly.express as px
# from reportlab.pdfgen import canvas

# Load Model
model = YOLO("runs/detect/train/weights/best.pt")

st.title("🌱 AgriVision AI - Weed Detection System")
confidence_threshold = st.slider(
    "Detection Confidence Threshold",
    min_value=0.1,
    max_value=1.0,
    value=0.5,
    step=0.05
)

uploaded_file = st.file_uploader(
    "Upload Image or Video",
    type=["jpg", "jpeg", "png", "mp4", "avi", "mov"]
)

if uploaded_file:

    file_type = uploaded_file.type

    # ==========================================
    # IMAGE ANALYSIS
    # ==========================================
    if "image" in file_type:

        image = Image.open(uploaded_file)

        results = model(image,conf=confidence_threshold)

        crop_count = 0
        weed_count = 0

        for box in results[0].boxes:

            cls = int(box.cls[0])

            if cls == 0:
                crop_count += 1

            elif cls == 1:
                weed_count += 1

        detected_image = results[0].plot()

        st.image(
            detected_image,
            caption="Detection Result",
            use_container_width=True
        )

        total = crop_count + weed_count

        if total > 0:
            weed_density = (weed_count / total) * 100
        else:
            weed_density = 0

        st.subheader("Detection Summary")

        st.write(f"Crop Count: {crop_count}")
        st.write(f"Weed Count: {weed_count}")
        st.write(f"Weed Density: {weed_density:.2f}%")

        if weed_density < 10:
            recommendation = "Healthy Field"
            st.success(recommendation)

        elif weed_density < 30:
            recommendation = "Moderate Infestation"
            st.warning(recommendation)

        else:
            recommendation = "High Infestation - Immediate Action Recommended"
            st.error(recommendation)
        dashboard_df = pd.DataFrame({
            "Category": ["Crop", "Weed"],
               "Count": [crop_count, weed_count]
                })
        fig = px.bar(
            dashboard_df,
            x="Category",
            y="Count",
            title="Detection Analytics"
            )

        st.plotly_chart(fig)   

        report = pd.DataFrame({
            "Crop Count": [crop_count],
            "Weed Count": [weed_count],
            "Weed Density (%)": [round(weed_density, 2)],
            "Recommendation": [recommendation]
        })

        csv = report.to_csv(index=False)

        st.download_button(
            "Download Report",
            csv,
            "weed_report.csv",
            "text/csv"
        )

    # ==========================================
    # VIDEO ANALYSIS
    # ==========================================
    elif "video" in file_type:

        st.info("Processing video... Please wait.")

        temp_file = tempfile.NamedTemporaryFile(delete=False)

        temp_file.write(uploaded_file.read())

        cap = cv2.VideoCapture(temp_file.name)

        crop_count = 0
        weed_count = 0
        frame_count = 0

        progress_bar = st.progress(0)

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        while cap.isOpened():

            ret, frame = cap.read()

            if not ret:
                break

            frame_count += 1

            results = model(frame,conf=confidence_threshold)

            for box in results[0].boxes:

                cls = int(box.cls[0])

                if cls == 0:
                    crop_count += 1

                elif cls == 1:
                    weed_count += 1

            if total_frames > 0:
                progress_bar.progress(
                    min(frame_count / total_frames, 1.0)
                )

        cap.release()

        total = crop_count + weed_count

        if total > 0:
            weed_density = (weed_count / total) * 100
        else:
            weed_density = 0

        st.subheader("🎥 Video Analysis Summary")

        st.write(f"Frames Processed: {frame_count}")
        st.write(f"Crop Count: {crop_count}")
        st.write(f"Weed Count: {weed_count}")
        st.write(f"Weed Density: {weed_density:.2f}%")

        if weed_density < 10:
            recommendation = "Healthy Field"
            st.success(recommendation)

        elif weed_density < 30:
            recommendation = "Moderate Infestation"
            st.warning(recommendation)

        else:
            recommendation = "High Infestation - Immediate Action Recommended"
            st.error(recommendation)
        dashboard_df = pd.DataFrame({
            "Category": ["Crop", "Weed"],
               "Count": [crop_count, weed_count]
                })
        fig = px.bar(
            dashboard_df,
            x="Category",
            y="Count",
            title="Detection Analytics"
            )

        st.plotly_chart(fig)

        report = pd.DataFrame({
            "Frames Processed": [frame_count],
            "Crop Count": [crop_count],
            "Weed Count": [weed_count],
            "Weed Density (%)": [round(weed_density, 2)],
            "Recommendation": [recommendation]
        })

        csv = report.to_csv(index=False)

        st.download_button(
            "Download Video Report",
            csv,
            "video_analysis_report.csv",
            "text/csv"
        )