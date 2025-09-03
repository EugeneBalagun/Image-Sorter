
# --- Необходимые библиотеки ---
# pip install numpy tqdm scipy torch torchvision Pillow faiss-cpu
# (или faiss-gpu, если у вас есть GPU и установлены CUDA toolkit)
# support ".jpg", ".jpeg", ".png", ".bmp", ".webp", ".jfif"

# --- Стандартная библиотека Python ---
import os
import shutil
import argparse
import time
import sqlite3
from sqlite3 import Error

os.environ['KMP_DUPLICATE_LIB_OK']='True' # либо для фикса import sklearn

# --- Сторонние библиотеки (ML, CV, Data) ---
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None  # Снимает ограничение на размер изображений
from tqdm.auto import tqdm      # Автоматически выбирает лучший вид progress bar

# --- Нейросетевые библиотеки (PyTorch) ---
import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode

# --- Библиотеки для графов и поиска соседей ---
import faiss
from scipy.sparse import csr_matrix, coo_matrix
from scipy.sparse.csgraph import minimum_spanning_tree, depth_first_order, connected_components



# ---------------------------------------------------------------------------- #
#                              Аргументы и Конфигурация                        #
# ---------------------------------------------------------------------------- #
parser = argparse.ArgumentParser(
    description="Сортировка картинок по визуальной близости с выбором модели"
)
parser.add_argument(
    "-m",
    type=str,
    default='regnet_y_16gf',
    choices=["mobilenet_v3_small",
             "mobilenet_v3_large",
             "convnext_small", 
             "regnet_y_16gf",
             "regnet_y_32gf",
             "regnet_y_128gf"],
    help="Какая модель будет использоваться для извлечения фичей",
)
 
parser.add_argument(
    "--more_scan",
    action='store_true',
    help="Использовать более качественный, но медленный режим извлечения признаков (анализ 6 вариантов изображения)."
)
parser.add_argument(
    "-i",
    type=str,
    default='input_images',
    help="Папка с исходными изображениями."
)
parser.add_argument(
    "-o",
    type=str,
    default='sorted_images',
    help="Папка для сохранения отсортированных изображений."
)

parser.add_argument(
    "--cpu",
    action='store_true',
    help="Использовать CPU вместо GPU."
)

args = parser.parse_args()
MODEL_NAME = args.m
MORE_SCAN = args.more_scan
SRC_FOLDER = args.i
DST_FOLDER = args.o

# --- Конфигурации нейросетей ---
if args.cpu:
    DEVICE = torch.device("cpu")
    print("задано использование только процессора")
else:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_CONFIGS = {
    "mobilenet_v3_small": {"loader": lambda: __import__("torchvision.models", fromlist=["mobilenet_v3_small"]).mobilenet_v3_small, "weights": "IMAGENET1K_V1", "hook_target": ("classifier", 3), "feat_dim": 1280},
    "mobilenet_v3_large": {"loader": lambda: __import__("torchvision.models", fromlist=["mobilenet_v3_large"]).mobilenet_v3_large, "weights": "IMAGENET1K_V1", "hook_target": ("classifier", 3), "feat_dim": 1280},
    "convnext_small": {"loader": lambda: __import__("torchvision.models", fromlist=["convnext_small"]).convnext_small, "weights": "IMAGENET1K_V1", "hook_target": ("classifier", 2), "feat_dim": 768},
    "regnet_y_16gf": {"loader": lambda: __import__("torchvision.models", fromlist=["regnet_y_16gf"]).regnet_y_16gf, "weights": "IMAGENET1K_SWAG_E2E_V1", "hook_target": ("fc", None), "feat_dim": 2592},
    "regnet_y_32gf": {"loader": lambda: __import__("torchvision.models", fromlist=["regnet_y_32gf"]).regnet_y_32gf, "weights": "IMAGENET1K_SWAG_E2E_V1", "hook_target": ("fc", None), "feat_dim": 3712},
    "regnet_y_128gf": {"loader": lambda: __import__("torchvision.models", fromlist=["regnet_y_128gf"]).regnet_y_128gf, "weights": "IMAGENET1K_SWAG_E2E_V1", "hook_target": ("fc", None), "feat_dim": 7392},


}  

# Если не тянет прям ваще, то используйте -m convnext_small
# mobilenet_v3 добавлен для сортировки миллионов картинок, помнимает цвет, форму объекта, крайне слаб, не используйте. 

# ---------------------------------------------------------------------------- #
#                             3) Загрузка модели                                 #
# ---------------------------------------------------------------------------- #
def load_model(model_name):
    cfg = MODEL_CONFIGS[model_name]
    model_constructor = cfg["loader"]()
    weights_str = cfg["weights"]

    local_weights_file = f"{model_name}.pth"
    if os.path.exists(local_weights_file):
        try:
            print(f"Найден локальный файл весов '{local_weights_file}'. Загрузка...")
            sd = torch.load(local_weights_file, map_location="cpu")
            model = model_constructor(weights=None)
            model.load_state_dict(sd)
            print("Локальные веса успешно загружены.")
        except Exception as e:
            print(f"Ошибка при загрузке локальных весов: {e}. Удаляем файл и скачиваем заново.")
            os.remove(local_weights_file)
            print(f"Загрузка предобученных весов '{weights_str}' для {model_name}...")
            model = model_constructor(weights=weights_str)
            torch.save(model.state_dict(), local_weights_file)
            print(f"Веса скачаны и сохранены в '{local_weights_file}'.")
    else:
        print(f"Локальный файл весов не найден. Загрузка предобученных весов '{weights_str}' для {model_name}...")
        model = model_constructor(weights=weights_str)
        torch.save(model.state_dict(), local_weights_file)
        print(f"Веса скачаны и сохранены в '{local_weights_file}'.")

    model.to(DEVICE).eval()

    # Вешаем hook
    hook_blob = {}
    group, idx_or_name = cfg["hook_target"]

    # Получаем родительский модуль (например, model.classifier или model.head)
    target_module_parent = getattr(model, group)
    
    # Определяем конечный модуль для hook'а
    if idx_or_name is None:
        # Если имя/индекс не указаны, значит родительский модуль и есть цель
        # (случай для swin_v2_t, где head - это и есть Linear слой)
        target_module = target_module_parent
    elif isinstance(idx_or_name, int):
        # Если это индекс, получаем элемент из родителя (случай для maxvit_t, mobilenet, efficientnet)
        target_module = target_module_parent[idx_or_name]
    else: 
        # Если это имя слоя, как 'fc' (случай для старых моделей, например, ResNet)
        target_module = getattr(target_module_parent, idx_or_name)

    def hook_fn(module, input, output):
        hook_blob["feat"] = input[0].detach().cpu().clone()
    
    target_module.register_forward_hook(hook_fn)

    return model, hook_blob

# ---------------------------------------------------------------------------- #
#                         4) Препроцессинг + извлечение                         #
# ---------------------------------------------------------------------------- #
# Стандартный препроцессор для быстрого режима
def extract_feature(path, model, hook_blob, more_scan=False):
    """
    Извлекает вектор признаков из одного изображения.
    Поддерживает стандартный и избыточный (more_scan) режимы.
    """
    # Определение pix_dim в зависимости от модели
    if MODEL_NAME == "convnext_small" or MODEL_NAME == "mobilenet_v3_small" or MODEL_NAME == "mobilenet_v3_large":
        pix_dim = 224
    elif MODEL_NAME == "regnet_y_16gf" or MODEL_NAME == "regnet_y_32gf" or MODEL_NAME == "regnet_y_128gf":
        pix_dim = 384
    else:
        raise ValueError(f"Неизвестная модель: {MODEL_NAME}")

    img = Image.open(path).convert("RGB")

    if not more_scan:
        # Стандартный режим: масштабирование с сохранением пропорций + центральный кроп до квадрата
        preproc = transforms.Compose([
            transforms.Resize(pix_dim, interpolation=InterpolationMode.BICUBIC),  # Масштабируем короткую сторону до pix_dim
            transforms.CenterCrop(pix_dim),  # Центральный кроп до квадрата pix_dim x pix_dim
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        x = preproc(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            _ = model(x)
        feat = hook_blob["feat"].numpy().ravel()
        return feat
    
    w, h = img.size
    if h == 0 or w == 0:
        raise ValueError("Изображение имеет нулевой размер")

    collected_feats = []
    final_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    def get_feat(processed_img):
        x = final_transform(processed_img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            # Предполагается, что модель изменяет hook_blob через forward hook
            _ = model(x)
        return hook_blob["feat"].numpy().ravel()
 
    # 1. Вычисляем "неквадратность" и ориентацию
    # Этот aspect_ratio всегда будет >= 1
    aspect_ratio = max(w, h) / min(w, h)
    is_horizontal = w >= h

    # 2. Определяем, какие именно области нужно вырезать
    crops_to_process = []
    if aspect_ratio < 1.3:
        # Почти квадратные изображения: 1 центральный кроп
        crops_to_process.append('center')
    elif aspect_ratio < 1.7:
        # Умеренно вытянутые: 2 кропа по краям
        if is_horizontal:
            crops_to_process.extend(['top_left', 'bottom_right'])
        else:
            crops_to_process.extend(['top_center', 'bottom_center'])
    else: 
        # Сильно вытянутые: 3 кропа (центр и края)
        crops_to_process.append('center')
        if is_horizontal:
            crops_to_process.extend(['top_left', 'bottom_right'])
        else:
            crops_to_process.extend(['top_center', 'bottom_center'])

    # 3. Готовим общие трансформации
    crop_size = min(w, h)
    resize_after_crop = transforms.Resize((pix_dim, pix_dim), interpolation=InterpolationMode.BICUBIC)

    # 4. Выполняем кропы и извлекаем признаки
    for crop_type in crops_to_process:
        if crop_type == 'center':
            cropped_img = transforms.functional.center_crop(img, (crop_size, crop_size))
        # Горизонтальные кропы
        elif crop_type == 'top_left':
            cropped_img = transforms.functional.crop(img, top=0, left=0, height=crop_size, width=crop_size)
        elif crop_type == 'bottom_right':
            cropped_img = transforms.functional.crop(img, top=h - crop_size, left=w - crop_size, height=crop_size, width=crop_size)
        # Вертикальные кропы
        elif crop_type == 'top_center':
            # Для вертикального кропа left вычисляется, чтобы он был по центру
            left = (w - crop_size) // 2
            cropped_img = transforms.functional.crop(img, top=0, left=left, height=crop_size, width=crop_size)
        elif crop_type == 'bottom_center':
            left = (w - crop_size) // 2
            cropped_img = transforms.functional.crop(img, top=h - crop_size, left=left, height=crop_size, width=crop_size)
        
        # Обрабатываем полученный кроп
        feature_vector = get_feat(resize_after_crop(cropped_img))
        collected_feats.append(feature_vector)
        
    if not collected_feats:
        raise ValueError("Не удалось извлечь признаки в режиме more_scan")
        
    final_feat = np.mean(np.array(collected_feats), axis=0)
    return final_feat


# ---------------------------------------------------------------------------- #
#                     5) Кэширование признаков в базе данных                     #
# ---------------------------------------------------------------------------- #
def process_and_cache_features(db_file, src_folder, model, hook, more_scan, batch_size=1000):
    """
    Универсальная функция для создания и обновления базы данных с признаками.

    Сканирует папку, находит файлы, отсутствующие в базе, обрабатывает их
    пакетами и сразу сохраняет в БД пакетами. Это гарантирует низкое потребление
    оперативной памяти даже при первом запуске на миллионах файлов.

    Args:
        db_file (str): Путь к файлу базы данных SQLite.
        src_folder (str): Путь к папке с исходными изображениями.
        model: Загруженная модель PyTorch.
        hook (dict): Словарь для извлечения признаков из хука модели.
        more_scan (bool): Флаг использования избыточного сканирования.
        batch_size (int): Количество файлов для обработки перед сохранением в БД.
    """
    print(f"Проверка и обновление кэша признаков в файле: {db_file}")
    conn = create_connection(db_file)
    if conn is None:
        print(" ! Не удалось создать соединение с базой данных.")
        return

    try:
        # Убедимся, что таблица существует
        create_table(conn)
        cursor = conn.cursor()

        # 1. Получаем список уже обработанных файлов из БД
        cursor.execute('SELECT filename FROM features')
        # Используем полный путь для корректного сравнения
        existing_paths_in_db = {row[0] for row in cursor.fetchall()}
        num_existing = len(existing_paths_in_db)
        if num_existing > 0:
            print(f"База данных найдена. В ней уже есть {num_existing} записей.")
        # 2. Получаем актуальный список файлов на диске
        supported_exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".jfif")
        current_paths_on_disk = {
            os.path.join(src_folder, fn) for fn in os.listdir(src_folder)
            if fn.lower().endswith(supported_exts)
        }

        # 3. Находим файлы, которые нужно обработать
        paths_to_process = sorted(list(current_paths_on_disk - existing_paths_in_db))

        if not paths_to_process:
            print("База данных актуальна. Новых файлов для обработки не найдено.")
            return

        print(f"Найдено {len(paths_to_process)} новых изображений для обработки.")
        
        # 4. Обрабатываем новые файлы пакетами
        for i in range(0, len(paths_to_process), batch_size):
            batch_paths = paths_to_process[i:i + batch_size]
            batch_feats_data = []
            
            desc_text = f"Обработка пакета {i//batch_size + 1}/{(len(paths_to_process) + batch_size - 1)//batch_size}"
            
            for full_path in tqdm(batch_paths, desc=desc_text):
                try:
                    feat = extract_feature(full_path, model, hook, more_scan=more_scan)
                    if feat is not None:
                        # Готовим данные для вставки в БД
                        safe_path = full_path.encode('utf-8', errors='replace').decode('utf-8')
                        feature_blob = sqlite3.Binary(feat.tobytes())
                        batch_feats_data.append((safe_path, feature_blob))
                except Exception as e:
                    print(f" ! Ошибка при обработке файла {full_path}: {e}")

            # 5. Сохраняем пакет в базу данных
            if batch_feats_data:
                try:
                    cursor.executemany('''
                        INSERT OR IGNORE INTO features (filename, features) VALUES (?, ?)
                    ''', batch_feats_data)
                    conn.commit()
                    print(f"Успешно сохранено {len(batch_feats_data)} новых признаков в БД.")
                except Exception as e:
                    print(f" ! Ошибка при сохранении пакета в базу данных: {e}")
                    conn.rollback()

    finally:
        if conn:
            conn.close()


def create_connection(db_file):
    """ Создает соединение с базой данных SQLite """
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        return conn
    except Error as e:
        print(e)
    return conn

def create_table(conn):
    """ Создает таблицу для хранения признаков """
    try:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS features (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT UNIQUE,
                features BLOB
            )
        ''')
    except Error as e:
        print(e)



def load_features_from_db(db_file):
    """
    Загружает признаки из базы данных.
    Проверяет, существуют ли файлы на диске, и пропускает записи об удаленных файлах.
    Если БД нет или она пуста — возвращает (None, None).
    """
    # Быстрая проверка, есть ли вообще файл БД
    if not os.path.exists(db_file):
        return None, None

    conn = create_connection(db_file)
    if conn is None:
        return None, None

    cursor = conn.cursor()
    try:
        # Извлекаем все записи из базы
        cursor.execute('SELECT filename, features FROM features')
    except sqlite3.OperationalError:
        # Таблицы 'features' может не существовать, если база создана, но пуста
        conn.close()
        return None, None

    rows = cursor.fetchall()
    conn.close()

    if not rows:
        return None, None

    # --- НОВАЯ ЛОГИКА ПРОВЕРКИ СУЩЕСТВОВАНИЯ ФАЙЛОВ ---
    
    paths = []
    feats_list = []
    deleted_count = 0

    print("Проверка актуальности записей в базе данных...")
    for filename, features_blob in rows:
        # Проверяем, что файл, указанный в базе, все еще существует
        if os.path.exists(filename):
            paths.append(filename)
            feats_list.append(np.frombuffer(features_blob, dtype=np.float32))
        else:
            # Файла нет, просто пропускаем эту запись, не трогая базу
            deleted_count += 1

    if deleted_count > 0:
        print(f"Пропущено {deleted_count} записей для удаленных файлов.")

    # Если после фильтрации ничего не осталось (например, удалили все картинки)
    if not paths:
        return None, None

    # Возвращаем только данные для существующих файлов
    return np.vstack(feats_list), paths




def copy_and_rename(paths, order, out_folder):
    """Копирует и переименовывает файлы в соответствии с вычисленным порядком."""
    # Создаем папку, если она не существует. Если существует - ничего не делаем.
    os.makedirs(out_folder, exist_ok=True)

    total_files = len(order)
    # -1, потому что индексация с нуля. Если файлов 100, то номера от 0 до 99.
    num_digits = len(str(total_files - 1)) if total_files > 0 else 1 
    fmt = f"{{:0{num_digits}d}}"

    pbar = tqdm(total=total_files, desc=f"Копирование в '{out_folder}'")
    for new_i, old_i in enumerate(order):
        src = paths[old_i]
        ext = os.path.splitext(src)[1].lower()
        original_name = os.path.splitext(os.path.basename(src))[0]
        # Формат имени: 0000_original_name.ext для удобной сортировки
        dst = os.path.join(out_folder, f"{fmt.format(new_i)}_{original_name}{ext}")
        shutil.copy2(src, dst)
        pbar.update(1)
    pbar.close()
    print(f"Копирование завершено.")


# ---------------------------------------------------------------------------- #
#                       Алгоритмы для поиска пути                              #
# ---------------------------------------------------------------------------- #

def sort_by_ann_mst(feats: np.ndarray, k: int, batch_size: int = 4096, use_gpu: bool = False, optimizer: str = '2opt', block_size: int = 100, shift: int = 90):
    """
    Улучшенная сортировка на основе MST с обработкой несвязных графов ("островов").
    Использует ЕВКЛИДОВО (L2) РАССТОЯНИЕ.

    Args:
        feats (np.ndarray): Массив признаков (N, D).
        k (int): Количество соседей для поиска в k-NN графе.
        batch_size (int): Размер батча для поиска соседей.
        use_gpu (bool): Использовать ли GPU для FAISS.

    Returns:
        np.ndarray: Отсортированный порядок индексов (путь) или None в случае ошибки.
    """
    
    
    
    if faiss is None:
        print("! ОШИБКА: Библиотека 'faiss' не установлена. Сортировка невозможна.")
        return None

    n, d = feats.shape
    total_start_time = time.time()

    print("\n" + "="*80)
    print(f"Запуск улучшенной сортировки методом ANN + MST (на основе Евклидова расстояния)") # Изменено для ясности
    print(f"  - Изображений: {n}")
    print(f"  - Размерность фичей: {d}")
    print(f"  - Соседей на точку (k): {k}")
    print(f"  - Размер батча: {batch_size}")
    print(f"  - Использовать GPU: {use_gpu}")
    print(f"  - Оптимизатор: {optimizer} (block_size={block_size}, shift={shift})")
    print("="*80)

    # Работаем с копией float32 для FAISS
    feats_copy = feats.astype('float32').copy()

    # --- Шаг 1: Индексация в FAISS ---
    step_start_time = time.time()
    print(f"\n[1/5] Шаг 1: Индексация векторов в FAISS...")
    try:
        # ИЗМЕНЕНИЕ 1: Убираем нормализацию. Она не нужна для евклидова расстояния.
        # faiss.normalize_L2(feats_copy)

        # ИЗМЕНЕНИЕ 2: Используем IndexFlatL2 для евклидова расстояния вместо IndexFlatIP.
        index = faiss.IndexFlatL2(d)

        if use_gpu and torch is not None and torch.cuda.is_available():
            print("  - Попытка использовать GPU для FAISS...")
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
            print("  - Индекс успешно перенесен на GPU.")

        index.add(feats_copy)
        print(f"  - Индекс создан. Всего векторов: {index.ntotal}.")
        print(f"  - Время на шаг 1: {time.time() - step_start_time:.2f} сек.")
    except Exception as e:
        print(f"! Ошибка на шаге 1: {e}")
        return None

    # --- Шаг 2: Поиск k-ближайших соседей ---
    step_start_time = time.time()
    print(f"\n[2/5] Шаг 2: Поиск {k} ближайших соседей...")
    try:
        all_distances = []
        all_indices = []
        for i in tqdm(range(0, n, batch_size), desc="  - Поиск k-NN (батчи)"):
            end = min(i + batch_size, n)
            # Ищем k+1 соседа, так как первый результат - это сама точка
            distances_batch, indices_batch = index.search(feats_copy[i:end], k + 1)
            all_distances.append(distances_batch)
            all_indices.append(indices_batch)
        
        # Для IndexFlatL2 возвращаются КВАДРАТЫ евклидовых расстояний.
        # Переименуем для ясности.
        distances_sq = np.vstack(all_distances)
        indices = np.vstack(all_indices)
        print(f"  - Поиск завершен. Размер матрицы индексов: {indices.shape}")
        print(f"  - Время на шаг 2: {time.time() - step_start_time:.2f} сек.")
    except Exception as e:
        print(f"! Ошибка на шаге 2: {e}")
        return None

    # --- Шаг 3: Построение симметричного разреженного графа ---
    step_start_time = time.time()
    print(f"\n[3/5] Шаг 3: Построение симметричного графа...")
    try:
        # Индексы соседей (исключаем первый столбец, т.к. это сама точка)
        cols = indices[:, 1:].flatten()
        # Индексы исходных точек, повторенные k раз
        rows = np.arange(n).repeat(k)

        # ИЗМЕНЕНИЕ 3: Расчет стоимости.
        # Квадрат расстояния уже является стоимостью (чем меньше, тем лучше).
        # Нам не нужно преобразовывать его из сходства.
        costs = distances_sq[:, 1:].flatten()
        # Проверка на отрицательные значения (хотя для L2 они маловероятны) не помешает.
        costs[costs < 0] = 0

        # Создаем асимметричный граф
        asymmetric_graph = csr_matrix((costs, (rows, cols)), shape=(n, n))

        # Делаем граф симметричным, выбирая минимальную стоимость ребра
        symmetric_graph = asymmetric_graph.minimum(asymmetric_graph.T)

        print(f"  - Граф успешно создан. Количество ребер: {symmetric_graph.nnz}.")
        print(f"  - Время на шаг 3: {time.time() - step_start_time:.2f} сек.")
    except Exception as e:
        print(f"! Ошибка на шаге 3: {e}")
        return None

    # --- Шаги 4 и 5 остаются без изменений, так как они работают с графом,
    # которому неважно, как были получены веса ребер. ---

    # --- Шаг 4: Построение Minimum Spanning Tree (MST) ---
    step_start_time = time.time()
    print(f"\n[4/5] Шаг 4: Построение Minimum Spanning Tree...")
    try:
        mst = minimum_spanning_tree(symmetric_graph)
        print(f"  - MST построено. Общая стоимость дерева: {mst.sum():.2f}.")
        print(f"  - Время на шаг 4: {time.time() - step_start_time:.2f} сек.")
    except Exception as e:
        print(f"! Ошибка на шаге 4: {e}")
        return None

    # --- Шаг 5
    from collections import defaultdict
    # Вспомогательные функции для улучшенного DFS (из предыдущего ответа)

    def get_adj_list(mst):
        adj = defaultdict(list)
        rows, cols = mst.nonzero()
        data = mst.data
        for r, c, dist in zip(rows, cols, data):
            adj[r].append((c, dist))
            adj[c].append((r, dist))
        return adj

    #def compute_greedy_walk_cost(start_node, adj, visited_in_main_dfs, depth):
    #    total_cost = 0.0
    #    current_node = start_node
    #    visited_in_walk = visited_in_main_dfs.copy()
    #    visited_in_walk.add(current_node)
    #    for _ in range(depth):
    #        neighbors = [(nb, dist) for nb, dist in adj[current_node] if nb not in visited_in_walk]
    #        if not neighbors:
    #            break
    #        next_node, dist = min(neighbors, key=lambda x: x[1])
    #        total_cost += dist
    #        current_node = next_node
    #        visited_in_walk.add(current_node)
    #    return total_cost
        
    def compute_greedy_walk_cost(start_node, adj, visited_in_main_dfs, depth):
        total_cost = 0.0
        current_node = start_node
        visited_in_walk = visited_in_main_dfs.copy()
        visited_in_walk.add(current_node)
    
        for _ in range(depth // 2):
            neighbors = [(nb, dist) for nb, dist in adj[current_node] if nb not in visited_in_walk]
    
            if not neighbors:
                break
    
            # Находим все возможные пары узлов и их суммарные стоимости
            min_pair_cost = float('inf')
            best_pair = None
    
            for nb1, dist1 in neighbors:
                # Находим соседей второго узла
                neighbors_of_nb1 = [(nb2, dist2) for nb2, dist2 in adj[nb1] if nb2 not in visited_in_walk and nb2 != current_node]
                for nb2, dist2 in neighbors_of_nb1:
                    pair_cost = dist1 + dist2
                    if pair_cost < min_pair_cost:
                        min_pair_cost = pair_cost
                        best_pair = (nb1, nb2, dist1, dist2)
    
            if best_pair is None:
                break
    
            next_node1, next_node2, dist1, dist2 = best_pair
            total_cost += dist1 + dist2
    
            # Обновляем текущий узел и добавляем посещенные узлы
            current_node = next_node2
            visited_in_walk.add(next_node1)
            visited_in_walk.add(next_node2)
    
        return total_cost


    def optimized_depth_first_order(mst, i_start, lookahead_depth, progress_callback=None):
        n = mst.shape[0]
        adj = get_adj_list(mst)
        visited = np.zeros(n, dtype=bool)
        path = []
        stack = [i_start]
        cost_cache = {}
        
        while stack:
            node = stack.pop()
            if visited[node]:
                continue
            path.append(node)
            visited[node] = True
            
            if progress_callback:
                progress_callback(len(path))
    
            unvisited_neighbors = [nb for nb, _ in adj[node] if not visited[nb]]
            if unvisited_neighbors:
                visited_nodes_set = set(np.where(visited)[0])
                neighbor_costs = []
                for nb in unvisited_neighbors:
                    if nb not in cost_cache:
                        cost_cache[nb] = compute_greedy_walk_cost(nb, adj, visited_nodes_set, depth=lookahead_depth)
                    neighbor_costs.append((nb, cost_cache[nb]))
                
                sorted_neighbors = [nb for nb, cost in sorted(neighbor_costs, key=lambda x: x[1])]
                stack.extend(sorted_neighbors[::-1])
                
        if progress_callback:
            progress_callback(len(path), final_update=True)
                
        return np.array(path), visited

    # --- Начало исправленного блока для Шага 5 ---
    step_start_time = time.time()
    print(f"\n[5/5] Шаг 5: Двухэтапная кластеризация и обход...")
    
    try:
        n = mst.shape[0]
        # Определяем все связанные компоненты в исходном MST
        n_components, labels = connected_components(csgraph=mst, directed=False, return_labels=True)
        
        # --- ЭТАП 1: ОБРАБОТКА ОСНОВНОЙ КОМПОНЕНТЫ ---
        print(f"\n  --- Этап 1: Обработка основной компоненты ---")
        if n_components > 0:
            component_sizes = np.bincount(labels)
            main_component_id = np.argmax(component_sizes)
            main_nodes_mask = (labels == main_component_id)
            main_nodes_indices = np.where(main_nodes_mask)[0]
            total_in_main_component = len(main_nodes_indices)
            
            print(f"  - Найдена основная компонента размером {total_in_main_component} узлов.")
            
            start_node = main_nodes_indices[0]
            
            LOOKAHEAD_DEPTH = 100 # Можете менять это значение, оно почти ни на что не влияет так как график строится жадно без ветвлений (например, 30, 50, 100, 0/1=выкл)
            
            # --- ГЛАВНЫЙ ПЕРЕКЛЮЧАТЕЛЬ АЛГОРИТМОВ ---
            if LOOKAHEAD_DEPTH <= 1:
                print(f"  - Глубина просмотра (depth={LOOKAHEAD_DEPTH}) <= 1. Используется стандартный быстрый DFS.")
                from scipy.sparse.csgraph import depth_first_order
                # Запускаем простой DFS на всем MST, начиная с узла из главной компоненты.
                # Он обойдет только свою компоненту связности.
                main_path_indices, _ = depth_first_order(mst, i_start=start_node, directed=False)
                main_path = main_path_indices[main_path_indices != -1] # Убираем непосещенные узлы
                print(f"    - Стандартный DFS завершен.")
            else:
                print(f"  - Глубина просмотра (depth={LOOKAHEAD_DEPTH}). Используется DFS с lookahead-оптимизацией.")
                last_reported_percent = -1
                def report_progress(processed_count, final_update=False):
                    nonlocal last_reported_percent
                    percent_done = int((processed_count / total_in_main_component) * 100)
                    if percent_done > last_reported_percent or final_update:
                        print(f"    - Прогресс: {processed_count} / {total_in_main_component} узлов ({percent_done}%)", end='\r')
                        last_reported_percent = percent_done
                
                main_path, _ = optimized_depth_first_order(
                    mst, 
                    start_node, 
                    lookahead_depth=LOOKAHEAD_DEPTH,
                    progress_callback=report_progress
                )
                print() # Перенос строки после прогресс-бара
            
            print(f"  - Основная компонента обработана. Длина основного пути: {len(main_path)}.")
            
        else:
            # Редкий случай, если граф пуст
            print("  - Не найдено ни одной компоненты. Основной путь пуст.")
            main_path = np.array([], dtype=int)
            main_nodes_mask = np.zeros(n, dtype=bool)
    
        # --- ЭТАП 2: ОБРАБОТКА ОСТАВШИХСЯ "ОСТРОВОВ" ---
        print(f"\n  --- Этап 2: Обработка оставшихся компонент ('островов') ---")
        
        island_nodes_indices = np.where(~main_nodes_mask)[0]
        num_islands = len(island_nodes_indices)
        
        secondary_path = np.array([], dtype=int) # Инициализируем вторичный путь
    
        if num_islands > 1:
            print(f"  - Найдено {num_islands} узлов в 'островах'. Запускаем для них отдельный процесс ANN+MST+DFS.")
            
            # Шаг 2.1: Извлекаем фичи только для "островов"
            island_feats = feats_copy[island_nodes_indices]
            
            # Шаг 2.2: ANN. Строим граф "все ко всем" (k = N-1)
            print(f"    - [2.1/2.4] Поиск соседей для {num_islands}...")
            island_k = min(1000, num_islands - 1) # k не может быть больше N-1
            island_index = faiss.IndexFlatL2(island_feats.shape[1])
            if use_gpu: # Предполагается, что use_gpu определена ранее
                res = faiss.StandardGpuResources()
                island_index = faiss.index_cpu_to_gpu(res, 0, island_index)
            island_index.add(island_feats)
            island_dists, island_neighbors_local_idx = island_index.search(island_feats, k=island_k)
            
            # Шаг 2.3: Построение графа и MST для "островов"
            print(f"    - [2.2/2.4] Построение графа для 'островов'...")
            
            # Индексы из FAISS - локальные (от 0 до num_islands-1).
            rows = np.arange(num_islands).repeat(island_k)
            cols = island_neighbors_local_idx.flatten()
            data = island_dists.flatten()
            
            # Убираем петли (i,i) и некорректные расстояния
            valid_mask = (rows != cols) & (data > 0)
            island_graph = coo_matrix((data[valid_mask], (rows[valid_mask], cols[valid_mask])), shape=(num_islands, num_islands))
            island_graph.eliminate_zeros()
            
            print(f"    - [2.3/2.4] Построение MST для 'островов'...")
            island_mst = minimum_spanning_tree(island_graph)
            
            # Шаг 2.4: Простой DFS для MST "островов"
            print(f"    - [2.4/2.4] Запуск простого DFS для 'островов'...")
            from scipy.sparse.csgraph import depth_first_order
            
            # Начинаем с первого узла в локальном списке "островов"
            island_path_local, _ = depth_first_order(island_mst, i_start=0, directed=False)
            island_path_local_visited = island_path_local[island_path_local != -1]
            
            # Преобразуем локальные индексы пути (0..num_islands-1) в глобальные индексы изображений
            secondary_path = island_nodes_indices[island_path_local_visited]
            print(f"  - 'Острова' обработаны. Длина вторичного пути: {len(secondary_path)}.")
            
        elif num_islands == 1:
            print("  - Найден 1 узел в 'островах', добавляем его напрямую.")
            secondary_path = island_nodes_indices
        else:
            print("  - Нет 'островов' для обработки.")
    
        # --- ЭТАП 3: ФИНАЛЬНАЯ СБОРКА И ПРОВЕРКА ---
        print(f"\n  --- Этап 3: Финальная сборка пути ---")
        
        # Собираем основной путь и путь "островов"
        path_list = list(main_path) + list(secondary_path)
        
        # Проверяем, все ли узлы были посещены. Добавляем "потерянные" в конец.
        if len(path_list) != n:
            print(f"  - ! ПРЕДУПРЕЖДЕНИЕ: Длина пути ({len(path_list)}) не равна общему числу элементов ({n}).")
            all_nodes = set(range(n))
            path_nodes = set(path_list)
            unvisited = list(all_nodes - path_nodes)
            print(f"    - Найдено {len(unvisited)} непосещенных узлов. Добавляем их в конец.")
            path_list.extend(unvisited)
        
        path = np.array(path_list)
        
        print(f"  - Сборка завершена. Получен полный путь из {len(path)} элементов.")
        print(f"  - Время на шаг 5: {time.time() - step_start_time:.2f} сек.")
    
    except Exception as e:
        print(f"\n! КРИТИЧЕСКАЯ ОШИБКА на шаге 5: {e}")
        import traceback
        traceback.print_exc()
        path = None # Возвращаем None в случае ошибки
    
   
    # --- Шаг 6: Пост-обработка пути с оптимизатором ---
    step_start_time = time.time()
    print(f"\n[6/6] Шаг 6: Пост-обработка пути с {optimizer} (блоки {block_size}, сдвиг {shift})...")
    
    # Вспомогательные функции для оптимизаторов
    def compute_distance_matrix(sub_feats):
        """Плотная матрица L2 расстояний для блока."""
        diff = sub_feats[:, None] - sub_feats[None, :]
        return np.sqrt(np.sum(diff**2, axis=-1))
    
    def two_opt(subpath, sub_feats, fixed_ends=True):
        """Простая 2-opt оптимизация для Hamiltonian path с фиксированными концами."""
        n_sub = len(subpath)
        if n_sub < 4: # Для 2-opt с фикс. концами нужно хотя бы 4 точки
            return subpath
        
        dist_matrix = compute_distance_matrix(sub_feats)
        path_indices = np.arange(n_sub) # Работаем с индексами 0..n-1
        
        improved = True
        while improved:
            improved = False
            for i in range(1, n_sub - 2):
                for j in range(i + 1, n_sub - 1):
                    # Старые ребра: (i-1 -> i) и (j -> j+1)
                    old_dist = dist_matrix[path_indices[i-1], path_indices[i]] + dist_matrix[path_indices[j], path_indices[j+1]]
                    # Новые ребра: (i-1 -> j) и (i -> j+1)
                    new_dist = dist_matrix[path_indices[i-1], path_indices[j]] + dist_matrix[path_indices[i], path_indices[j+1]]
                    
                    if new_dist < old_dist - 1e-6:
                        path_indices[i:j+1] = path_indices[i:j+1][::-1]
                        improved = True
        return subpath[path_indices]

    
    # Разбиение на overlapping блоки и оптимизация
    optimized_path = path.copy()
    num_blocks = max(1, (n - block_size) // shift + 1)
    
    for b in tqdm(range(num_blocks), desc="  - Оптимизация блоков"):
        start = b * shift
        end = min(start + block_size, n)
        if end - start < 3:
            continue
        
        subpath = optimized_path[start:end]
        sub_feats = feats[subpath]
        
        # В вашей логике концы всегда фиксированы, кроме, возможно, крайних блоков.
        # Для простоты здесь всегда считаем их фиксированными, т.к. они соединяются с остальной частью пути.
        # Если нужна особая логика для крайних блоков, ее можно добавить сюда.
        fixed_ends = True
        
        if optimizer == '2opt':
            new_subpath = two_opt(subpath, sub_feats, fixed_ends=fixed_ends)

        optimized_path[start:end] = new_subpath
    
    path = optimized_path
    print(f"  - Пост-обработка завершена. Время: {time.time() - step_start_time:.2f} сек.")

    
    # --- Финальный вывод (замените ваши оригинальные print'ы на это, чтобы учесть шаг 6) ---
    print("\n" + "="*80)
    print("Сортировка методом ANN + MST успешно завершена.")
    print(f"Общее время выполнения: {time.time() - total_start_time:.2f} сек.")
    print("="*80 + "\n")
    
    return path
    

def sort_images(feats, paths, out_folder):  
    n = feats.shape[0]
    if n < 2:
        print("Изображений слишком мало для сортировки.")
        copy_and_rename(paths, np.arange(n), out_folder)
        return
    
    print(f"\n# {'-'*76} #")
    print(f"# Сортировка изображений методом 'ANN+MST'") 
    print(f"# {'-'*76} #")
    print(f"Всего изображений: {n}")
 
    if len(paths) < 600: k_neighbors = len(paths) - 1
    else: k_neighbors = 600
    use_gpu_faiss = not args.cpu and torch.cuda.is_available()
    final_order = sort_by_ann_mst(feats, k=k_neighbors, use_gpu=use_gpu_faiss)
    
    if final_order is None or len(final_order) == 0:
        print("! Сортировка не удалась, итоговый путь пуст. Копирование отменено.")
        return
    
    copy_and_rename(paths, final_order, out_folder)
    print("Сортировка завершена.")
 

# ---------------------------------------------------------------------------- #
#                                    Main                                      #
# ---------------------------------------------------------------------------- #
if __name__ == "__main__":
    # --- ЛОГИКА СОЗДАНИЯ ИМЕНИ БАЗЫ ДАННЫХ ---
    safe_model_name = MODEL_NAME.replace("/", "_")
    scan_suffix = "_more_scan" if MORE_SCAN else ""
    DB_FILE = f"features_db_{safe_model_name}{scan_suffix}.sqlite"

    # Убедимся, что папка с исходниками есть
    if not os.path.isdir(SRC_FOLDER):
        print(f"Создайте папку {SRC_FOLDER} и положите туда картинки.")
        os.makedirs(SRC_FOLDER, exist_ok=True)
        exit(1)

    # 1. Загружаем модель (она нужна для обработки новых файлов)
    model, hook = load_model(MODEL_NAME)

    # 2. Создаем или обновляем базу данных с признаками
    # Эта функция сама найдет новые файлы и обработает только их
    process_and_cache_features(DB_FILE, SRC_FOLDER, model, hook, more_scan=MORE_SCAN)

    # 3. Загружаем ВСЕ актуальные признаки из базы для сортировки
    print("Загружаем все признаки из базы данных для начала сортировки...")
    feats, paths = load_features_from_db(DB_FILE)

    # 4. Если у нас есть фичи, запускаем сортировку
    if paths and feats is not None and len(paths) > 0:
        print(f"Всего в работе {len(paths)} изображений. Начинаем сортировку...")
        sort_images(feats, paths, DST_FOLDER)
    else:
        print(f"В папке {SRC_FOLDER} нет изображений для обработки, или не удалось загрузить признаки.")
        exit(0)

    # Удаление модели и освобождение ресурсов
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()