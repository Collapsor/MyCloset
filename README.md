# MyCloset
Мой шкаф
Что такое мода? Мода будет развиваться с использованием компьютеров.
С помощью машинного обучения мы изучим большое количество комбинаций стилей и разметим данные. Затем пользователи могут ввести предмет одежды, который у них уже есть в шкафу, указав, является ли он верхом или низом. Система будет ссылаться на перекрестную матрицу и определять, какие атрибуты имеет входная одежда, какие атрибуты рекомендуются (на основе эталонной одежды) и какие предметы одежды (на основе предмета гардероба) лучше всего соответствуют рекомендуемым атрибутам.

part1 Object Detection

Data Preparation
Extracting and Splitting Data.ipynb
This notebook will extract all the information from the four categories of the DeepFashion dataset and  split the data between training and testing set.

Each file outputs like the following: img_name x1 y1 x2 y2 upper_lower
The x1 y1 x2 y1 refers to the coordinate of the bbox of the item. upper_lower can either be a 1 or 2 depending on whether the clothing was upper or lower.

part2
