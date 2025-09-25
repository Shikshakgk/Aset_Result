from flask import Flask, render_template, request
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd   # NEW for Excel export

app = Flask(__name__)

# ------------------ Function to process one ASET image ------------------
def process_aset_image(img_path, save_path):
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((5, 5), np.uint8)
    diamond_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    coords = cv2.findNonZero(diamond_mask)
    x, y, w, h = cv2.boundingRect(coords)
    img_cropped = img_rgb[y:y + h, x:x + w]
    diamond_mask_cropped = diamond_mask[y:y + h, x:x + w]

    hsv = cv2.cvtColor(img_cropped, cv2.COLOR_RGB2HSV)

    # --- Color ranges ---
    red1 = cv2.inRange(hsv, (0, 90, 50), (10, 255, 255))
    red2 = cv2.inRange(hsv, (170, 90, 50), (179, 255, 255))
    red_mask = cv2.bitwise_or(red1, red2)
    green_mask = cv2.inRange(hsv, (35, 60, 40), (90, 255, 255))
    blue_mask = cv2.inRange(hsv, (90, 60, 40), (140, 255, 255))
    black_mask = cv2.inRange(hsv, (0, 0, 0), (180, 255, 50))
    gray_mask = cv2.inRange(hsv, (0, 0, 80), (180, 50, 200))
    white_mask = cv2.inRange(hsv, (0, 0, 200), (180, 40, 255))

    # --- Pixel counts ---
    diamond_area = np.count_nonzero(diamond_mask_cropped)
    red_count = np.count_nonzero(cv2.bitwise_and(red_mask, diamond_mask_cropped))
    green_count = np.count_nonzero(cv2.bitwise_and(green_mask, diamond_mask_cropped))
    blue_count = np.count_nonzero(cv2.bitwise_and(blue_mask, diamond_mask_cropped))
    black_count = np.count_nonzero(cv2.bitwise_and(black_mask, diamond_mask_cropped))
    gray_count = np.count_nonzero(cv2.bitwise_and(gray_mask, diamond_mask_cropped))
    white_count = np.count_nonzero(cv2.bitwise_and(white_mask, diamond_mask_cropped))

    grey_count = black_count + gray_count + white_count

    percentages = {
        "Red": 100 * red_count / diamond_area,
        "Green": 100 * green_count / diamond_area,
        "Blue": 100 * blue_count / diamond_area,
        "Others": 100 * grey_count / diamond_area
    }

    # --- Overlay masks for visualization ---
    overlay = np.zeros_like(img_cropped)
    overlay[red_mask > 0] = [255, 0, 0]
    overlay[green_mask > 0] = [0, 255, 0]
    overlay[blue_mask > 0] = [0, 0, 255]
    overlay[black_mask > 0] = [128, 128, 128]
    overlay[gray_mask > 0] = [128, 128, 128]
    overlay[white_mask > 0] = [128, 128, 128]

    blended = cv2.addWeighted(img_cropped, 0.6, overlay, 0.4, 0)

    # --- Create the 2x2 Figure ---
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    axes[0, 0].imshow(img_rgb)
    axes[0, 0].set_title("Original ASET image")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(img_cropped)
    axes[0, 1].set_title("Diamond (background removed)")
    axes[0, 1].axis("off")

    axes[1, 0].imshow(blended)
    axes[1, 0].set_title("Detected colors (R=red, G=green, B=blue, Others=black+white+grey)")
    axes[1, 0].axis("off")

    colors = ['#FF0000', '#00FF00', '#0000FF', '#808080']
    axes[1, 1].pie(percentages.values(), labels=percentages.keys(),
                   autopct='%1.1f%%', colors=colors)
    axes[1, 1].set_title("Color distribution inside diamond")

    plt.tight_layout()
    plt.savefig(save_path)   # Save full 2x2 figure
    plt.close()

    return percentages


# ------------------ Flask Routes ------------------
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        src_folder = request.form["src_folder"]
        dest_folder = request.form["dest_folder"]

        if not os.path.exists(src_folder):
            return f"❌ Source folder does not exist: {src_folder}"
        if not os.path.exists(dest_folder):
            os.makedirs(dest_folder)

        excel_data = []

        for file in os.listdir(src_folder):
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                img_path = os.path.join(src_folder, file)
                save_path = os.path.join(dest_folder, file.replace(".jpg", "_analysis.png")
                                                           .replace(".jpeg", "_analysis.png")
                                                           .replace(".png", "_analysis.png"))

                percentages = process_aset_image(img_path, save_path)

                # Add row for Excel
                excel_data.append({
                    "File": file,
                    "Red %": round(percentages["Red"], 2),
                    "Green %": round(percentages["Green"], 2),
                    "Blue %": round(percentages["Blue"], 2),
                    "Others %": round(percentages["Others"], 2)
                })

        # Save Excel file
        if excel_data:
            df = pd.DataFrame(excel_data)
            excel_path = os.path.join(dest_folder, "ASET_Analysis.xlsx")
            df.to_excel(excel_path, index=False)

        return f"✅ Processing complete! Results saved in: {dest_folder}"

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True,port=5110)


