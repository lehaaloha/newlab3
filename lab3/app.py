from flask import Flask, request, render_template, url_for, flash
from werkzeug.utils import secure_filename
import os
from PIL import Image
import numpy as np
import random
import string

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Создаем папку для загрузок
try:
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
except FileExistsError:
    pass


def load_neural_network():
    #легкая нейросеть
    try:
        import tensorflow as tf
        from tensorflow.keras.applications import MobileNetV2
        print("✅ MobileNetV2 загружается...")
        model = MobileNetV2(weights='imagenet')
        return model
    except Exception as e:
        print(f"❌ Ошибка загрузки нейросети: {e}")
        return None

# Загружаем модель один раз при старте
neural_model = load_neural_network()

def generate_captcha():
    """Генерация случайной CAPTCHA"""
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))

def classify_image(image_path):
    """Классификация изображения с помощью MobileNetV2"""
    if neural_model is None:
        return [{'class': 'model_loading_error', 'probability': 100.0}]
    
    try:
        import tensorflow as tf
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
        from tensorflow.keras.preprocessing import image as keras_image
        
        # Загружаем и подготавливаем изображение
        img = Image.open(image_path).convert('RGB')
        img = img.resize((224, 224))
        
        # Преобразуем в numpy array
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        
        # Предсказание нейросети
        predictions = neural_model.predict(img_array, verbose=0)
        decoded = decode_predictions(predictions, top=3)[0]
        
        # Форматируем результаты
        results = []
        for i in range(3):
            class_name = decoded[i][1].replace('_', ' ')
            probability = float(decoded[i][2]) * 100
            results.append({
                'class': class_name,
                'probability': round(probability, 2)
            })
        
        return results
        
    except Exception as e:
        print(f"❌ Ошибка классификации: {e}")
        return [
            {'class': 'classification_error', 'probability': 85.5},
            {'class': 'image_processing', 'probability': 12.3},
            {'class': 'neural_network', 'probability': 2.2}
        ]

def process_image(image_path: str):
    """Обработка изображения: сдвиг частей БЕЗ гистограммы"""
    original_img = Image.open(image_path)
    width, height = original_img.size
    
    # Разбиваем на 4 части
    half_w, half_h = width//2, height//2
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
            processed_filename = process_image(file_path)
            classification_results = classify_image(file_path)

            return render_template('result.html',
                                   original_image=filename,
                                   processed_image=processed_filename,
                                   classification_results=classification_results)

        except Exception as e:
            flash(f'Ошибка обработки изображения: {str(e)}', 'error')
            return render_template('index.html', captcha_text=generate_captcha())


if __name__ == '__main__':
    import os
    print("🚀 Запуск Flask приложения с MobileNetV2...")
    
    
    port = int(os.environ.get('PORT', 5000))
    
   
    app.run(host='0.0.0.0', port=port, debug=False)  


