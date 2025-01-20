from ultralytics import YOLO
from collections import defaultdict, Counter
import os

model = YOLO("runs/detect/train2/weights/best.pt")
ans = model.predict("17/", verbose=True, save_txt=True)

# Функция для чтения данных из stats.txt
def read_stats(file_path):
    stats = {}
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 4:
                continue  # Пропустить некорректные строки
            filename = os.path.basename(parts[0])  # Извлекаем только имя файла
            counts = {
                't': int(parts[1].split('=')[1]),
                'c': int(parts[2].split('=')[1]),
                'q': int(parts[3].split('=')[1])
            }
            stats[filename] = counts
    return stats


# Чтение правильных значений из файла stats.txt
stats = read_stats('17/stats.txt')

# Предположим, что ans - это список результатов детекции для изображений
class_counts = defaultdict(Counter)

for result in ans:
    # Извлекаем имя файла, оставляя только имя без пути
    filename = os.path.basename(result.path)

    # Извлекаем классы из результата
    cls_labels = result.boxes.cls.to('cpu').to(int).tolist()

    # Обновляем счетчик классов для данного изображения
    class_counts[filename].update(cls_labels)

# Инициализация общих переменных
total_detected = Counter()
total_correct = Counter()

# Сравниваем результаты с правильными значениями и суммируем результаты
for filename, counts in class_counts.items():
    if filename in stats:
        correct_counts = stats[filename]

        # Обновляем общие счетчики
        total_detected.update(counts)
        total_correct.update(correct_counts)
    else:
        print(f"No stats available for {filename}. Detected counts: {counts}")

# Выводим общую статистику
print("Overall Statistics:")
print(f"Detected - Circles: {total_detected[1]}, Triangles: {total_detected[2]}, Quadrilaterals: {total_detected[0]}")
print(
    f"Correct   - Circles: {total_correct['c']}, Triangles: {total_correct['t']}, Quadrilaterals: {total_correct['q']}")
print(f"Difference - Circles: {total_detected[1] - total_correct['c']}, "
      f"Triangles: {total_detected[2] - total_correct['t']}, "
      f"Quadrilaterals: {total_detected[0] - total_correct['q']}")

# print(ans[0])
# Словарь для хранения количества объектов каждого класса
# class_counts = Counter()

# Обрабатываем результаты предсказаний
# for result in ans:
#     # Извлекаем классы из результата
#     cls_labels = result.boxes.cls.to('cpu').to(int).tolist()
#
#     # Обновляем счетчик классов
#     class_counts.update(cls_labels)
#
# # Выводим количество объектов каждого класса
# for class_id, count in class_counts.items():
#     print(f"Class ID {class_id}: {count} instances")
#     # print(clsLabels)