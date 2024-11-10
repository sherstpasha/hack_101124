import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
from transformers import XCLIPModel, XCLIPProcessor
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import torch.optim as optim

from dataset import XCLIPVideoDataset, EmbeddingsDataset
from model import EmbeddingClassifier

import warnings

# Игнорировать все предупреждения
warnings.filterwarnings("ignore")


# ======== Подготовка данных ======== #
# Пути к данным
csv_path = r"C:\Users\pasha\OneDrive\Рабочий стол\dataset1011_1\videos_cut\violations_summary_p.csv"
video_folder = r"C:\Users\pasha\OneDrive\Рабочий стол\dataset1011_1\videos_cut"
processor = XCLIPProcessor.from_pretrained("microsoft/xclip-base-patch16")

# Загрузка данных
data = pd.read_csv(csv_path)

# Добавляем уникальный идентификатор для каждой строки
data.reset_index(drop=True, inplace=True)
data["id"] = data.index  # Используем индекс как уникальный идентификатор

label_list = data["violation_name"].unique().tolist()
print(label_list)


# Создание полного датасета
full_dataset = XCLIPVideoDataset(
    data,
    video_folder,
    processor,
    num_frames=8,
    apply_preprocessing=True,
    yolo_custom_path=r"C:\Users\pasha\OneDrive\Рабочий стол\best_93.pt",
    yolo_pretrained_path=r"C:\Users\pasha\OneDrive\Рабочий стол\best_93.pt",
    segformer_model_path=r"C:\Users\pasha\OneDrive\Рабочий стол\model",
)

# Создание DataLoader для вычисления эмбеддингов
dataloader = DataLoader(full_dataset, batch_size=1, shuffle=False)

# ======== Загрузка модели XCLIP и установка устройства ======== #
model_name = "microsoft/xclip-base-patch16"
model = XCLIPModel.from_pretrained(model_name)

# Устройство (CPU или GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model.to(device)
model.eval()  # Переводим модель в режим оценки

# Папка для сохранения эмбеддингов
embeddings_dir = "./embeddings"
os.makedirs(embeddings_dir, exist_ok=True)

# Вычисление и сохранение эмбеддингов
with torch.no_grad():
    for idx, sample in enumerate(tqdm(dataloader, desc="Processing")):
        pixel_values = sample["pixel_values"]  # [batch_size, num_frames, 3, 224, 224]
        label = sample["label"]  # [batch_size]
        segment_id = sample["segment_id"].item()  # Идентификатор сегмента
        segment_name = sample["segment_name"][0]  # Имя видеофайла (для отладки)

        # Создаем имя файла для эмбеддинга, используя идентификатор сегмента
        embedding_filename = f"{segment_id}.pt"
        embedding_file = os.path.join(embeddings_dir, embedding_filename)

        # Проверяем, существует ли файл эмбеддинга
        if os.path.exists(embedding_file):
            # Эмбеддинг уже существует, пропускаем вычисление
            continue

        # Перемещаем данные на устройство
        pixel_values = pixel_values.to(device)

        # Генерируем фиктивный текстовый ввод и перемещаем на устройство
        text_inputs = processor(
            text=[""] * pixel_values.size(0),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77,
        )
        input_ids = text_inputs["input_ids"].to(device)
        attention_mask = text_inputs["attention_mask"].to(device)

        # Передаем pixel_values в модель
        outputs = model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        video_embeds = outputs.video_embeds  # [batch_size, projection_dim]

        # Переносим эмбеддинги на CPU перед сохранением
        video_embeds_cpu = video_embeds.squeeze(0).cpu()

        # Сохраняем эмбеддинги и метку
        torch.save(
            {
                "embedding": video_embeds_cpu,  # [projection_dim]
                "label": label.item(),
                "segment_id": segment_id,
                "segment_name": segment_name,  # Сохраняем имя сегмента для отладки
            },
            embedding_file,
        )


# Разделяем исходный датафрейм на обучающую и валидационную выборки
train_data, val_data = train_test_split(
    data, test_size=0.3, stratify=data["violation_name"], random_state=42
)

# Создаем датасеты
train_dataset = EmbeddingsDataset(train_data, embeddings_dir)
val_dataset = EmbeddingsDataset(val_data, embeddings_dir)


# Инициализация модели
input_dim = model.config.projection_dim  # Размерность эмбеддингов
num_classes = len(label_list)
classifier_model = EmbeddingClassifier(input_dim, num_classes)

# Параметры обучения
batch_size = 8
num_epochs = 5000
initial_learning_rate = 1e-4  # Начальный learning rate

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# Определение функции потерь и оптимизатора
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(classifier_model.parameters(), lr=initial_learning_rate)

# Добавляем scheduler для изменения learning rate
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.1, patience=5, verbose=True
)

# Перемещение классификатора на устройство
classifier_model.to(device)

# Папка для сохранения моделей
models_dir = "./models"
os.makedirs(models_dir, exist_ok=True)

best_val_loss = float("inf")
best_model_path = os.path.join(models_dir, "best_model.pth")

# Добавляем параметры для ранней остановки
early_stopping_patience = 100  # Количество эпох без улучшения для остановки
epochs_without_improvement = 0  # Счетчик эпох без улучшения

# Цикл обучения
for epoch in range(num_epochs):
    classifier_model.train()
    running_loss = 0.0
    for batch in train_loader:
        embeddings = batch["embedding"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        outputs = classifier_model(embeddings)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * embeddings.size(0)

    epoch_loss = running_loss / len(train_dataset)

    # Валидация
    classifier_model.eval()
    val_running_loss = 0.0  # Добавлено для валидационного лосса
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in val_loader:
            embeddings = batch["embedding"].to(device)
            labels = batch["label"].to(device)
            outputs = classifier_model(embeddings)
            val_loss = criterion(outputs, labels)  # Вычисляем лосс на валидации
            val_running_loss += val_loss.item() * embeddings.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    val_loss_epoch = val_running_loss / len(val_dataset)  # Средний лосс на валидации
    val_accuracy = correct / total

    # Сохранение модели при улучшении валидационной ошибки
    if val_loss_epoch < best_val_loss:
        best_val_loss = val_loss_epoch
        torch.save(classifier_model.state_dict(), best_model_path)
        print(
            f"Модель сохранена на {epoch + 1} эпохе с валидационной ошибкой: {val_loss_epoch:.4f}"
        )
        epochs_without_improvement = 0  # Сброс счетчика
    else:
        epochs_without_improvement += 1

    # Шаг scheduler
    scheduler.step(val_loss_epoch)

    # Проверка на раннюю остановку
    if epochs_without_improvement >= early_stopping_patience:
        print(
            f"Ранняя остановка на {epoch + 1} эпохе. Валидационная ошибка не улучшалась {early_stopping_patience} эпох."
        )
        break

    # Вывод информации
    print(
        f"Epoch {epoch+1}/{num_epochs}, "
        f"Training Loss: {epoch_loss:.4f}, "
        f"Validation Loss: {val_loss_epoch:.4f}, "
        f"Validation Accuracy: {val_accuracy * 100:.2f}%"
    )

# Загрузка лучшей модели для оценки
best_model_path = os.path.join(models_dir, "best_model.pth")
classifier_model.load_state_dict(torch.load(best_model_path))
classifier_model.eval()

print("Лучшая модель загружена для оценки.")

# ======== Оценка модели на валидационном наборе ======== #
# Сбор всех предсказаний и истинных меток
val_all_preds = []
val_all_labels = []
with torch.no_grad():
    for batch in val_loader:
        embeddings = batch["embedding"].to(device)
        labels = batch["label"].to(device)
        outputs = classifier_model(embeddings)
        _, predicted = torch.max(outputs.data, 1)
        val_all_preds.extend(predicted.cpu().numpy())
        val_all_labels.extend(labels.cpu().numpy())

# Получаем уникальные метки, присутствующие в данных
present_labels = np.unique(val_all_labels)
present_label_names = [label_list[i] for i in present_labels]

# Вычисление матрицы ошибок для валидационного набора
cm = confusion_matrix(val_all_labels, val_all_preds, labels=present_labels)
print("Classification report for validation data:")
print(
    classification_report(
        val_all_labels,
        val_all_preds,
        labels=present_labels,
        target_names=present_label_names,
    )
)

# Отображение матрицы ошибок для валидационного набора
plt.figure(figsize=(12, 10))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=present_label_names,
    yticklabels=present_label_names,
)
plt.ylabel("Истинные метки")
plt.xlabel("Предсказанные метки")
plt.title("Матрица ошибок (валидация)")
plt.show()

# ======== Оценка модели на обучающем наборе ======== #
# Сбор всех предсказаний и истинных меток для обучающего набора
train_all_preds = []
train_all_labels = []
with torch.no_grad():
    for batch in train_loader:
        embeddings = batch["embedding"].to(device)
        labels = batch["label"].to(device)
        outputs = classifier_model(embeddings)
        _, predicted = torch.max(outputs.data, 1)
        train_all_preds.extend(predicted.cpu().numpy())
        train_all_labels.extend(labels.cpu().numpy())

# Получаем уникальные метки, присутствующие в данных
train_present_labels = np.unique(train_all_labels)
train_present_label_names = [label_list[i] for i in train_present_labels]
print(train_present_label_names)

# Вычисление матрицы ошибок для обучающего набора
train_cm = confusion_matrix(
    train_all_labels, train_all_preds, labels=train_present_labels
)
print("Classification report for training data:")
print(
    classification_report(
        train_all_labels,
        train_all_preds,
        labels=train_present_labels,
        target_names=train_present_label_names,
    )
)

# Отображение матрицы ошибок для обучающего набора
plt.figure(figsize=(12, 10))
sns.heatmap(
    train_cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=train_present_label_names,
    yticklabels=train_present_label_names,
)
plt.ylabel("Истинные метки")
plt.xlabel("Предсказанные метки")
plt.title("Матрица ошибок (обучение)")
plt.show()
