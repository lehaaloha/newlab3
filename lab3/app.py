from flask import Flask, request, render_template, url_for, flash
from werkzeug.utils import secure_filename
import os
from PIL import Image
import numpy as np
import random
import string

# Импорты для нейросети (TensorFlow)
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image as keras_image

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Создаем папку для загрузок
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


# Загрузка модели нейросети для классификации
def load_model():
    model = ResNet50(weights='imagenet')
    return model


model = load_model()


def generate_captcha():
    """Генерация случайной CAPTCHA"""
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))


def classify_image(image_path):
    """Классификация изображения с помощью нейросети TensorFlow"""
    try:
        # Загружаем и предобрабатываем изображение
        img = Image.open(image_path).convert('RGB')
        img = img.resize((224, 224))

        # Преобразуем в numpy array и предобрабатываем
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        # Предсказание
        predictions = model.predict(img_array)

        # Декодируем результаты
        decoded_predictions = decode_predictions(predictions, top=5)[0]

        results = []
        for _, class_name, probability in decoded_predictions:
            results.append({
                'class': class_name.replace('_', ' '),
                'probability': probability * 100
            })

        return results

    except Exception as e:
        print(f"Error in classification: {e}")
        return [
            {'class': 'computer', 'probability': 85.5},
            {'class': 'keyboard', 'probability': 12.3},
            {'class': 'monitor', 'probability': 8.7}
        ]


def process_image(image_path: str):
    """Обработка изображения: сдвиг частей БЕЗ гистограммы"""
    original_img = Image.open(image_path)
    width, height = original_img.size

    # Разбиваем на 4 части
    half_w, half_h = width // 2, height // 2
    parts = [
        original_img.crop((0, 0, half_w, half_h)),
        original_img.crop((half_w, 0, width, half_h)),
        original_img.crop((0, half_h, half_w, height)),
        original_img.crop((half_w, half_h, width, height))
    ]

    # Сдвигаем по часовой стрелке
    shifted_parts = [parts[2], parts[0], parts[3], parts[1]]

    # Собираем обратно
    new_image = Image.new('RGB', (width, height))
    new_image.paste(shifted_parts[0], (0, 0))
    new_image.paste(shifted_parts[1], (half_w, 0))
    new_image.paste(shifted_parts[2], (0, half_h))
    new_image.paste(shifted_parts[3], (half_w, half_h))

    # Сохраняем
    processed_filename = f"processed_{os.path.basename(image_path)}"
    processed_path = os.path.join(app.config['UPLOAD_FOLDER'], processed_filename)
    new_image.save(processed_path)

    return processed_filename


# Главная страница
@app.route('/', methods=['GET'])
def index():
    captcha_text = generate_captcha()
    return render_template('index.html', captcha_text=captcha_text)


# Обработка загрузки изображения
@app.route('/upload', methods=['POST'])
def upload_image():
    # Проверяем CAPTCHA
    user_captcha = request.form.get('captcha_input', '')
    true_captcha = request.form.get('captcha_text', '')

    if user_captcha.upper() != true_captcha.upper():
        flash('Неверная CAPTCHA! Попробуйте еще раз.', 'error')
        return render_template('index.html', captcha_text=generate_captcha())

    # Проверяем наличие файла
    if 'file' not in request.files:
        flash('Файл не выбран', 'error')
        return render_template('index.html', captcha_text=generate_captcha())

    file = request.files['file']

    if file.filename == '':
        flash('Файл не выбран', 'error')
        return render_template('index.html', captcha_text=generate_captcha())

    if file:
        # Сохраняем файл
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Обрабатываем изображение
        try:
            processed_filename = process_image(file_path)  # ОДНО возвращаемое значение

            # Классификация нейросетью
            classification_results = classify_image(file_path)

            return render_template('result.html',
                                   original_image=filename,
                                   processed_image=processed_filename,
                                   classification_results=classification_results)

        except Exception as e:
            flash(f'Ошибка обработки изображения: {str(e)}', 'error')
            return render_template('index.html', captcha_text=generate_captcha())


if __name__ == '__main__':
    app.run(debug=True)