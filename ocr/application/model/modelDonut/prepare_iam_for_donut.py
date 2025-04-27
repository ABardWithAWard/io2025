import os
import pandas as pd
from PIL import Image
from tqdm import tqdm


# Definicja folderów
IAM_WORDS_IMAGE_DIR = "datasets/iam/iam_dataset/words"  # Poprawiona ścieżka do folderu "words"
TRANSCRIPTION_FILES = ["datasets/iam/iam_dataset/linux_gt.txt", "datasets/iam/iam_dataset/train_gt.txt", "datasets/iam/iam_dataset/val_gt.txt"]
OUTPUT_IMAGE_DIR = "datasets/iam/iam_dataset/pages"  # Zapisz strony w tym katalogu
OUTPUT_LABELS_PATH = "datasets/iam/iam_dataset/labels.csv"

os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True)

# 1. Wczytaj transkrypcje z plików txt (linux_gt.txt, train_gt.txt, val_gt.txt)
def load_transcriptions(files):
    transcription_map = {}
    for file_path in files:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("#"):  # Ignoruj komentarze
                    continue
                parts = line.strip().split("\t", 1)
                if len(parts) == 2:
                    img_path = parts[0]
                    text = parts[1].replace("|", " ").strip()  # Zamień "|" na spacje
                    img_id = img_path.replace("words/", "").replace(".png", "")
                    transcription_map[img_id] = text
    return transcription_map

# 2. Grupuj obrazy w strony (np. `a01-000u`)
def group_images_by_page(image_dir):
    page_map = {}
    for folder in os.listdir(image_dir):  # iterujemy przez główne foldery (a01, a02, a03, ...)
        folder_path = os.path.join(image_dir, folder)
        if os.path.isdir(folder_path):
            for subfolder in os.listdir(folder_path):  # iterujemy przez podfoldery (a01-000u, a01-001u...)
                subfolder_path = os.path.join(folder_path, subfolder)
                if os.path.isdir(subfolder_path):
                    images = []
                    for filename in sorted(os.listdir(subfolder_path)):
                        if filename.endswith(".png"):  # Załóżmy, że obrazy mają rozszerzenie .png
                            image_id = os.path.join(folder, subfolder, filename.replace(".png", ""))
                            images.append(image_id)
                    if images:
                        page_map[subfolder] = images  # Strona = podfolder
    return page_map

# 3. Twórz połączone obrazy i zapisuj obrazy + teksty
def create_combined_pages(transcriptions, page_map, image_dir, output_dir):
    rows = []
    for page_id, image_ids in tqdm(page_map.items(), desc="Creating pages"):
        images = []
        full_text = []

        for image_id in image_ids:
            image_path = os.path.join(image_dir, image_id + ".png")
            image_path = image_path.replace("\\", "/")
            image_id = image_id.replace("\\", "/")
            # print(transcriptions)
            if os.path.exists(image_path) and image_id in transcriptions:
                img = Image.open(image_path).convert("RGB")
                images.append(img)
                full_text.append(transcriptions[image_id])
            else:
                print(f"Brak obrazu dla: {image_id} w ścieżce {image_path}")
                continue

        if images:
            total_height = sum(img.height for img in images)
            max_width = max(img.width for img in images)
            combined_img = Image.new("RGB", (max_width, total_height), (255, 255, 255))

            y_offset = 0
            for img in images:
                combined_img.paste(img, (0, y_offset))
                y_offset += img.height

            filename = f"{page_id}.png"
            combined_img.save(os.path.join(output_dir, filename))

            rows.append({"file_name": filename, "text": "\n".join(full_text)})

    return pd.DataFrame(rows)

# 4. Wykonaj wszystko:
transcriptions = load_transcriptions(TRANSCRIPTION_FILES)  # Wczytaj transkrypcje z plików
page_map = group_images_by_page(IAM_WORDS_IMAGE_DIR)  # Grupowanie obrazów w strony
df = create_combined_pages(transcriptions, page_map, IAM_WORDS_IMAGE_DIR, OUTPUT_IMAGE_DIR)  # Tworzenie stron
#
# Zapisz etykiety
df.to_csv(OUTPUT_LABELS_PATH, index=False)
print(f"\n✅ Gotowe! Zapisano {len(df)} stron do {OUTPUT_LABELS_PATH}")