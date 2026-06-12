import streamlit as st
from ultralytics import YOLO
from PIL import Image
import pandas as pd

model = YOLO("runs/detect/train/weights/best.pt")

st.title("AI-Based Weed Detection System")

uploaded_file = st.file_uploader(
    "Upload Field Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:

    image = Image.open(uploaded_file)

    results = model(image)

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

    st.subheader("Detection Summary")

    st.write(f"Crop Count: {crop_count}")
    st.write(f"Weed Count: {weed_count}")

    total = crop_count + weed_count

    if total > 0:
        weed_density = (weed_count / total) * 100
    else:
        weed_density = 0

    st.write(f"Weed Density: {weed_density:.2f}%")

    if weed_density < 10:
        st.success("Low weed infestation detected.")
    elif weed_density < 30:
        st.warning("Moderate weed infestation detected.")
    else:
        st.error("High weed infestation detected.")

    report = pd.DataFrame({
        "Crop Count": [crop_count],
        "Weed Count": [weed_count],
        "Weed Density (%)": [round(weed_density, 2)]
    })

    csv = report.to_csv(index=False)

    st.download_button(
        "Download Report",
        csv,
        "weed_report.csv",
        "text/csv"
    )