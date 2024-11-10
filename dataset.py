from torch.utils.data import Dataset
from torchvision import transforms
import cv2
import numpy as np
from PIL import Image
import os
import torch
import pandas as pd

label_list = [
    "нарушений нет",
    "Статья 12.16. часть 1 Несоблюдение требований, предписанных дорожными знаками или разметкой проезжей части дороги",
    "Статья 12.16 часть 2 Поворот налево или разворот в нарушение требований, предписанных дорожными знаками или разметкой проезжей части дороги",
    "Статья 12.17  часть 1.1 и 1.2. движение транспортных средств по полосе для маршрутных транспортных средств или остановка на указанной полосе в нарушение Правил дорожного движения ",
    "Статья 12.12 часть 2 1. невыполнение требования ПДД об остановке перед стоп-линией, обозначенной дорожными знаками или разметкой проезжей части дороги, при запрещающем сигнале светофора или запрещающем жесте регулировщика",
    "Статья 12.15 часть 4 Выезд в нарушение правил дорожного движения на полосу, предназначенную для встречного движения, при объезде препятствия, либо на трамвайные пути встречного направления, за исключением случаев, предусмотренных частью 3 настоящей статьи",
]


# Определение класса датасета с применением обработки кадров
import torch
import cv2
import numpy as np
from PIL import Image
import os
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

label_list = [
    "нарушений нет",
    "Статья 12.16. часть 1 Несоблюдение требований, предписанных дорожными знаками или разметкой проезжей части дороги",
    "Статья 12.16 часть 2 Поворот налево или разворот в нарушение требований, предписанных дорожными знаками или разметкой проезжей части дороги",
    "Статья 12.17  часть 1.1 и 1.2. движение транспортных средств по полосе для маршрутных транспортных средств или остановка на указанной полосе в нарушение Правил дорожного движения ",
    "Статья 12.12 часть 2 1. невыполнение требования ПДД об остановке перед стоп-линией, обозначенной дорожными знаками или разметкой проезжей части дороги, при запрещающем сигнале светофора или запрещающем жесте регулировщика",
    "Статья 12.15 часть 4 Выезд в нарушение правил дорожного движения на полосу, предназначенную для встречного движения, при объезде препятствия, либо на трамвайные пути встречного направления, за исключением случаев, предусмотренных частью 3 настоящей статьи",
]


# Определение класса датасета с применением обработки кадров и сохранением в папку
class XCLIPVideoDataset(Dataset):
    def __init__(
        self,
        dataframe,
        video_folder,
        processor,
        num_frames=8,
        apply_preprocessing=False,
        yolo_pretrained_path=None,
        yolo_custom_path=None,
        segformer_model_path=None,
    ):
        self.data_frame = dataframe.reset_index(drop=True)
        self.video_folder = video_folder
        self.processor = processor
        self.num_frames = num_frames
        self.apply_preprocessing = (
            apply_preprocessing  # Тумблер для применения предварительной обработки
        )

        # Загрузка моделей
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Загрузка модели YOLOv5 (предобученной)
        if yolo_pretrained_path:
            self.pretrained_model = (
                torch.hub.load(
                    "ultralytics/yolov5",
                    "yolov5n",
                    pretrained=True,
                )
                .to(self.device)
                .eval()
            )

        # Загрузка кастомной модели YOLOv5
        if yolo_custom_path:
            self.custom_model = (
                torch.hub.load(
                    "ultralytics/yolov5",
                    "custom",
                    path=yolo_custom_path,
                    force_reload=True,
                )
                .to(self.device)
                .eval()
            )

        # Загрузка модели SegFormer
        if segformer_model_path:
            self.extractor = SegformerImageProcessor()
            self.segformer_model = (
                SegformerForSemanticSegmentation.from_pretrained(segformer_model_path)
                .to(self.device)
                .eval()
            )

        self.video_transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ]
        )

        # Папка для сохранения кадров
        # self.save_frames_folder = "./saved_frames"
        # os.makedirs(self.save_frames_folder, exist_ok=True)

        # Параметры для обработки
        self.traffic_related_classes = ["car", "bus", "truck", "motorcycle", "bicycle"]
        self.target_class_id = 2  # Идентификатор целевого класса для SegFormer

    def extract_video_frames(self, video_path, num_frames):
        video_capture = cv2.VideoCapture(video_path)
        frames = []
        total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_interval = max(total_frames // num_frames, 1)

        for frame_idx in range(0, total_frames, frame_interval):
            video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            success, frame = video_capture.read()
            if not success:
                break

            if self.apply_preprocessing:
                frame = self.apply_models_processing(frame)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)
            frame_tensor = self.video_transform(frame_pil)
            frames.append(frame_tensor)

            # Сохранение кадра
            # frame_save_path = os.path.join(
            #     self.save_frames_folder,
            #     f"{os.path.basename(video_path).split('.')[0]}_frame_{frame_idx}.png",
            # )
            # frame_pil.save(frame_save_path)

            if len(frames) >= num_frames:
                break

        video_capture.release()

        if len(frames) == 0:
            print(f"Не удалось извлечь кадры для видео {video_path}")
            return torch.zeros((num_frames, 3, 224, 224))

        while len(frames) < num_frames:
            frames.append(
                frames[-1].clone() if len(frames) > 0 else torch.zeros(3, 224, 224)
            )

        return torch.stack(frames)

    def apply_models_processing(self, frame):
        height, width, _ = frame.shape

        # Преобразование кадра для SegFormer
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb_frame)

        # Получение результатов от обеих моделей YOLOv5
        results_pretrained = self.pretrained_model(img)
        results_custom = self.custom_model(img)

        # Объединение результатов в один DataFrame
        results_combined = pd.concat(
            [results_pretrained.pandas().xyxy[0], results_custom.pandas().xyxy[0]],
            ignore_index=True,
        )

        # Обработка кадра моделью SegFormer
        seg_map = self.predict_segformer(
            self.segformer_model, self.extractor, rgb_frame
        )

        # Создание маски для затемнения
        mask = np.zeros((height, width), dtype=np.uint8)

        # Добавление результатов YOLOv5 в маску
        for _, row in results_combined.iterrows():
            if row["name"] in self.traffic_related_classes or row["confidence"] > 0.25:
                x1 = int(max(0, row["xmin"]))
                y1 = int(max(0, row["ymin"]))
                x2 = int(min(width - 1, row["xmax"]))
                y2 = int(min(height - 1, row["ymax"]))
                mask[y1:y2, x1:x2] = 255  # Область, которую не затемняем

        # Добавление результатов SegFormer в маску
        if seg_map.shape != (height, width):
            seg_map_resized = cv2.resize(
                seg_map, (width, height), interpolation=cv2.INTER_NEAREST
            )
        else:
            seg_map_resized = seg_map
        seg_mask = np.where(seg_map_resized == self.target_class_id, 255, 0).astype(
            np.uint8
        )
        mask = cv2.bitwise_or(mask, seg_mask)

        # Создание итогового кадра с затемнением
        alpha_mask = cv2.merge((mask, mask, mask))
        frame_darkened = (frame * 0.2).astype(np.uint8)
        frame_result = np.where(alpha_mask == 255, frame, frame_darkened)

        return frame_result

    def predict_segformer(self, model, extractor, image):
        inputs = extractor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits  # Shape [batch_size, num_classes, height, width]
        segmentation = torch.argmax(logits, dim=1).squeeze(0)
        return segmentation.cpu().numpy()

    def __getitem__(self, idx):
        row = self.data_frame.iloc[idx]
        segment_id = row["id"]
        segment_name = row["segment_name"]
        label = row["violation_name"]
        video_path = os.path.join(self.video_folder, segment_name)

        video_frames_tensor = self.extract_video_frames(video_path, self.num_frames)
        label_id = label_list.index(label)

        return {
            "pixel_values": video_frames_tensor,  # [num_frames, 3, 224, 224]
            "label": torch.tensor(label_id, dtype=torch.long),
            "segment_id": segment_id,  # Добавляем идентификатор сегмента
            "segment_name": segment_name,  # Сохраняем имя сегмента для отладки, если нужно
        }

    def __len__(self):
        return len(self.data_frame)


# ======== Создание датасета для эмбеддингов ======== #
class EmbeddingsDataset(Dataset):
    def __init__(self, dataframe, embeddings_dir):
        self.data_frame = dataframe.reset_index(drop=True)
        self.embeddings_dir = embeddings_dir

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        row = self.data_frame.iloc[idx]
        segment_id = row["id"]
        label = row["violation_name"]
        label_id = label_list.index(label)

        embedding_filename = f"{segment_id}.pt"
        embedding_file = os.path.join(self.embeddings_dir, embedding_filename)

        data = torch.load(embedding_file)
        embedding = data["embedding"]  # Tensor размерности [projection_dim]
        return {"embedding": embedding, "label": label_id}
