from ultralytics import YOLO
import pandas as pd
model = YOLO("runs/detect/train/weights/best.pt")

results = model("C:\\Users\\HP\\OneDrive\\Desktop\\Weed_Detection\\dataset\\test\\images\\rgb-2022-10-06-17-07-14_jpg.rf.4a0cf5d07c2514c95f0f0dec47beceb0.jpg")

for result in results:
    boxes = result.boxes

    crop_count = 0
    weed_count = 0

    for box in boxes:
        cls = int(box.cls[0])

        if cls == 0:
            crop_count += 1
        elif cls == 1:
            weed_count += 1

    print(f"Crop Count: {crop_count}")
    print(f"Weed Count: {weed_count}")

    data = {
    "Crop Count": [crop_count],
    "Weed Count": [weed_count]
}

df = pd.DataFrame(data)
df.to_csv("report.csv", index=False)

print(df)