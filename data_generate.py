import os
import glob
import random
import numpy as np
import cv2
import pandas as pd
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
from fontTools.ttLib import TTFont

OUTPUT_DIR = "dataset_orientation"
NUM_PAIRS = 50000
CUSTOM_FONTS_DIR = None


# Словари данных
RU_WORDS = [
    "телефон", "состояние", "оригинал", "гарантия", "доставка", "цена", "рублей",
    "новый", "купить", "скидка", "запчасти", "автомобиль", "отличное", "в наличии",
    "паспорт", "документы", "оплата", "комплект", "москва", "размер", "продажа",
    "квартира", "колеса", "шины", "диски", "часы", "ремонт", "сервис", "быстро"
]

EN_WORDS = [
    "apple", "iphone", "samsung", "pro", "max", "ultra", "sony", "original",
    "condition", "price", "sale", "new", "used", "auto", "parts", "model",
    "delivery", "black", "white", "service", "smart", "store", "xiaomi",
    "canon", "nike", "adidas", "bosch", "intel", "geforce", "edition"
]

RU_CODES = [
    "128 Гб", "256 Гб", "№ 482910", "р. 42-44", "+7 999 123-45-67",
    "15 000 ₽", "990 руб.", "ГОСТ 305-2013", "ул. Ленина", "д. 14, кв. 5"
]

EN_CODES = [
    "RTX 3060", "USB-C", "256 GB", "v1.0.4", "A-100", "i5-12400F",
    "SN: 884920", "$120", "VIN: WBA3A5C50DF", "100% cotton", "Type-C 65W"
]

# Цветовые темы оформления
COLOR_PALETTES = {
    "light": {
        "backgrounds": [(255, 255, 255), (245, 245, 245), (238, 238, 238), (255, 253, 231), (232, 240, 254)],
        "texts": [(15, 15, 15), (35, 35, 35), (20, 40, 80), (120, 20, 20), (20, 80, 40)]
    },
    "dark": {
        "backgrounds": [(18, 18, 18), (28, 28, 28), (15, 23, 42), (38, 20, 20), (20, 35, 20)],
        "texts": [(255, 255, 255), (240, 240, 240), (255, 215, 0), (0, 230, 118), (129, 212, 250)]
    },
    "badge": {
        "pairs": [
            ((229, 57, 53), (255, 255, 255)), # Красный ценник + белый текст
            ((253, 216, 53), (20, 20, 20)),   # Желтый ценник + темный текст
            ((253, 216, 53), (183, 28, 28)),  # Желтый ценник + красный текст
            ((30, 136, 229), (255, 255, 255)), # Синий стикер + белый текст
            ((67, 160, 71), (255, 255, 255)) # Зеленая плашка + белый текст
        ]
    }
}

def get_font_supported_chars(font_path):
    """Считывает таблицу cmap шрифта и возвращает множество кодов доступных глифов"""
    try:
        with TTFont(font_path) as tt:
            chars = set()
            for table in tt["cmap"].tables:
                chars.update(table.cmap.keys())
            return chars
    except Exception:
        return set()

def index_fonts(font_paths):
    """Индексирует шрифты и разделяет их по наличию кириллического диапазона"""
    font_db = []
    cyrillic_test_chars = {ord(c) for c in "абвгдеёжзийклмнопрстуфхцчшщъыьэюя"}

    for path in font_paths:
        supported = get_font_supported_chars(path)
        if not supported:
            continue
        has_cyr = cyrillic_test_chars.issubset(supported)
        font_db.append({
            "path": path,
            "supported": supported,
            "has_cyrillic": has_cyr
        })
    return font_db

def generate_text_by_category():
    """Генерирует текст строго по пропорции 60% (RU/Mix) и 40% (Latin)"""
    category = random.choices(["ru_mix", "latin"], weights=[0.60, 0.40])[0]

    if category == "ru_mix":
        submode = random.choices(["pure_ru", "mixed", "ru_code"], weights=[0.65, 0.20, 0.15])[0]
        if submode == "pure_ru":
            text = " ".join(random.choices(RU_WORDS, k=random.randint(1, 3)))
        elif submode == "mixed":
            ru = random.choice(RU_WORDS)
            en = random.choice(EN_WORDS)
            text = f"{en.capitalize()} {ru}" if random.random() > 0.5 else f"{ru.capitalize()} {en}"
        else:
            text = random.choice(RU_CODES)
        lang_group = "ru_mix"
    else:
        submode = random.choices(["pure_en", "en_code"], weights=[0.70, 0.30])[0]
        if submode == "pure_en":
            text = " ".join(random.choices(EN_WORDS, k=random.randint(1, 3)))
        else:
            text = random.choice(EN_CODES)
        lang_group = "latin"

    return text, lang_group

def pick_valid_font(font_db, text, font_size):
    """Выбирает шрифт, гарантирующий корректную отрисовку каждого символа в тексте"""
    text_chars = {ord(c) for c in text}
    has_cyrillic = any(ord("а") <= ord(c.lower()) <= ord("я") or c in "ёЁ" for c in text)

    candidates = [
        f["path"] for f in font_db
        if (not has_cyrillic or f["has_cyrillic"]) and text_chars.issubset(f["supported"])
    ]

    if not candidates:
        return None

    selected_path = random.choice(candidates)
    try:
        return ImageFont.truetype(selected_path, font_size)
    except Exception:
        return None

def get_color_scheme():
    theme = random.choices(["light", "dark", "badge"], weights=[0.60, 0.25, 0.15])[0]
    if theme == "badge":
        bg_col, text_col = random.choice(COLOR_PALETTES["badge"]["pairs"])
    else:
        bg_col = random.choice(COLOR_PALETTES[theme]["backgrounds"])
        text_col = random.choice(COLOR_PALETTES[theme]["texts"])
    return bg_col, text_col, theme

def render_crop(text, font, bg_color, text_color):
    dummy = Image.new("RGB", (1, 1))
    draw_dummy = ImageDraw.Draw(dummy)

    bbox = draw_dummy.textbbox((0, 0), text, font=font)
    t_width = bbox[2] - bbox[0]
    t_height = bbox[3] - bbox[1]

    if t_width <= 0 or t_height <= 0:
        return None

    # Поля с достаточным запасом, чтобы не срезать выносные хвостики букв
    pad_top = random.randint(6, 12)
    pad_bottom = random.randint(6, 12)
    pad_left = random.randint(8, 16)
    pad_right = random.randint(8, 16)

    img_w = t_width + pad_left + pad_right
    img_h = t_height + pad_top + pad_bottom

    img = Image.new("RGB", (img_w, img_h), color=bg_color)
    draw = ImageDraw.Draw(img)
    draw.text((pad_left - bbox[0], pad_top - bbox[1]), text, fill=text_color, font=font)

    # Небольшой наклон в пределах естественной погрешности бокса
    angle = random.uniform(-3.5, 3.5)
    img = img.rotate(angle, resample=Image.BICUBIC, expand=True, fillcolor=bg_color)
    return np.array(img)

def apply_artifacts(img_rgb):
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    if random.random() < 0.35:
        k = random.choice([3, 5])
        img_bgr = cv2.GaussianBlur(img_bgr, (k, k), 0)

    if random.random() < 0.40:
        sigma = random.uniform(2.0, 6.0)
        noise = np.random.normal(0, sigma, img_bgr.shape).astype(np.float32)
        img_bgr = np.clip(img_bgr.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    if random.random() < 0.50:
        quality = random.randint(60, 95)
        _, enc = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        img_bgr = cv2.imdecode(enc, 1)

    return img_bgr

def find_font_files():
    paths = []
    if CUSTOM_FONTS_DIR and os.path.exists(CUSTOM_FONTS_DIR):
        paths.extend(glob.glob(os.path.join(CUSTOM_FONTS_DIR, "*.[tT][tT][fF]")))
        paths.extend(glob.glob(os.path.join(CUSTOM_FONTS_DIR, "*.[oO][tT][fF]")))

    system_paths = [
        "/System/Library/Fonts/Supplemental/*.ttf",
        "/System/Library/Fonts/Supplemental/*.otf",
        "/Library/Fonts/*.ttf",
        "/Library/Fonts/*.otf"
    ]
    for p in system_paths:
        paths.extend(glob.glob(p))
    return sorted(list(set(paths)))

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    images_dir = os.path.join(OUTPUT_DIR, "images")
    os.makedirs(images_dir, exist_ok=True)

    print("Сканирование и валидация символьных таблиц шрифтов...")
    raw_fonts = find_font_files()
    font_db = index_fonts(raw_fonts)
    
    cyr_fonts_count = sum(1 for f in font_db if f["has_cyrillic"])
    print(f"Всего шрифтов: {len(font_db)} | Поддерживают кириллицу: {cyr_fonts_count}")

    records = []
    saved_pairs = 0
    pbar = tqdm(total=NUM_PAIRS, desc="Генерация датасета")

    while saved_pairs < NUM_PAIRS:
        text, lang_group = generate_text_by_category()
        font_size = random.randint(28, 44)
        
        font = pick_valid_font(font_db, text, font_size)
        if font is None:
            continue

        bg_col, text_col, theme = get_color_scheme()
        img_rgb = render_crop(text, font, bg_col, text_col)
        if img_rgb is None:
            continue

        crop_0 = apply_artifacts(img_rgb)
        if crop_0.shape[0] < 8 or crop_0.shape[1] < 8:
            continue

        group_id = f"sample_{saved_pairs:07d}"

        # 0°
        filename_0 = f"{group_id}_0.jpg"
        cv2.imwrite(os.path.join(images_dir, filename_0), crop_0)
        records.append({
            "filename": filename_0,
            "group_id": group_id,
            "label": 0,
            "text": text,
            "lang_group": lang_group,
            "theme": theme
        })

        # 180°
        crop_180 = cv2.rotate(crop_0, cv2.ROTATE_180)
        filename_180 = f"{group_id}_180.jpg"
        cv2.imwrite(os.path.join(images_dir, filename_180), crop_180)
        records.append({
            "filename": filename_180,
            "group_id": group_id,
            "label": 180,
            "text": text,
            "lang_group": lang_group,
            "theme": theme
        })

        saved_pairs += 1
        pbar.update(1)

    pbar.close()

    csv_path = os.path.join(OUTPUT_DIR, "labels.csv")
    df = pd.DataFrame(records)
    df.to_csv(csv_path, index=False)

    ru_share = (df["lang_group"] == "ru_mix").mean() * 100
    en_share = (df["lang_group"] == "latin").mean() * 100
    print(f"\nГотово! Создано картинок: {len(records)}.")
    print(f"Распределение: {ru_share:.1f}% RU/Mix vs {en_share:.1f}% Latin.")
    print(f"Файл разметки сохранен: {csv_path}")

if __name__ == "__main__":
    main()