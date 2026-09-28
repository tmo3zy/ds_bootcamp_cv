import os
import glob
import lmdb
import cv2
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

IMAGES_DIR = "dataset_orientation/images"
LABELS_PATH = "dataset_orientation/labels.csv"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
mdb_files = glob.glob(os.path.join(BASE_DIR, "**", "data.mdb"), recursive=True)

os.makedirs(IMAGES_DIR, exist_ok=True)

if os.path.exists(LABELS_PATH) and os.path.getsize(LABELS_PATH) > 0:
    df_existing = pd.read_csv(LABELS_PATH)
    if "group_id" in df_existing.columns:
        numeric_ids = pd.to_numeric(df_existing["group_id"], errors="coerce").dropna()
        max_val = numeric_ids.max()
        start_group_id = int(max_val) + 1 if pd.notna(max_val) else 100000
    else:
        start_group_id = 100000
    new_records = []
    print(f"Найдено существующих записей в labels.csv: {len(df_existing)}")
else:
    df_existing = pd.DataFrame()
    start_group_id = 100000
    new_records = []

if not mdb_files:
    downloads_dir = os.path.expanduser("~/Downloads")
    mdb_files = glob.glob(os.path.join(downloads_dir, "**", "data.mdb"), recursive=True)

mdb_folders = sorted(list(set(os.path.dirname(p) for p in mdb_files)))

print(f"Найдено LMDB баз: {len(mdb_folders)}")
for f in mdb_folders:
    print(" ->", f)

current_group_id = start_group_id
total_extracted = 0

for folder in mdb_folders:
    folder_name = os.path.basename(folder)
    try:
        env = lmdb.open(folder, max_readers=32, readonly=True, lock=False, readahead=False, meminit=False)
    except Exception as e:
        print(f"Пропуск {folder}: {e}")
        continue

    with env.begin(write=False) as txn:
        cursor = txn.cursor()
        for key, value in cursor:
            if key.startswith(b"image-"):
                nparr = np.frombuffer(value, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                if img is None:
                    continue

                h, w = img.shape[:2]
                if h < 8 or w < 8:
                    continue

                # 1. Оригинальная картинка (0°)
                fn_0 = f"real_{current_group_id:06d}_0.jpg"
                cv2.imwrite(os.path.join(IMAGES_DIR, fn_0), img)
                new_records.append({"filename": fn_0, "label": 0, "group_id": current_group_id})

                # 2. Повернутая картинка (180°)
                img_180 = cv2.rotate(img, cv2.ROTATE_180)
                fn_180 = f"real_{current_group_id:06d}_180.jpg"
                cv2.imwrite(os.path.join(IMAGES_DIR, fn_180), img_180)
                new_records.append({"filename": fn_180, "label": 180, "group_id": current_group_id})

                current_group_id += 1
                total_extracted += 1

    env.close()

df_new = pd.DataFrame(new_records)
df_final = pd.concat([df_existing, df_new], ignore_index=True)
df_final.to_csv(LABELS_PATH, index=False)

print(f"\nГотово!")
print(f"Успешно извлечено реальных изображений: {total_extracted}")
print(f"Сформировано пар (0° и 180°): {len(df_new)}")
print(f"Итоговый размер обучающей выборки: {len(df_final)} картинок")