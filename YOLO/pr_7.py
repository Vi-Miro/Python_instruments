def count_values(filename):
    total_t = 0
    total_c = 0
    total_q = 0

    with open(filename, 'r') as file:
        for line in file:
            # Убираем лишние пробелы и разбиваем строку по пробелам
            parts = line.strip().split()
            if len(parts) < 4:  # Проверяем, что строка содержит достаточное количество элементов
                print(f"Пропущена строка: {line.strip()} (недостаточно данных)")
                continue

            try:
                # Извлекаем значения t, c и q
                t_value = int(parts[1].split('=')[1])
                c_value = int(parts[2].split('=')[1])
                q_value = int(parts[3].split('=')[1])

                # Суммируем значения
                total_t += t_value
                total_c += c_value
                total_q += q_value
            except (IndexError, ValueError) as e:
                print(f"Ошибка при обработке строки: {line.strip()} ({e})")

    print(f"Общее количество t: {total_t}")
    print(f"Общее количество c: {total_c}")
    print(f"Общее количество q: {total_q}")


count_values('17/stats.txt')