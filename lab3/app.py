from flask import Flask, request, render_template, url_for, flash, redirect, send_from_directory
from werkzeug.utils import secure_filename
import os
from PIL import Image
import numpy as np
import random
import string
import requests
from datetime import datetime
import sys
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions

print("=" * 60)
print("🚀 НАЧАЛО ЗАПУСКА ПРИЛОЖЕНИЯ")
print("=" * 60)

app = Flask(__name__)

# ===== КОНФИГУРАЦИЯ =====
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-key-12345-change-me')
app.config['UPLOAD_FOLDER'] = 'uploads'  # Простая папка в корне
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

# ===== СОЗДАНИЕ ПАПКИ =====
upload_dir = app.config['UPLOAD_FOLDER']
if not os.path.exists(upload_dir):
    os.makedirs(upload_dir)
    print(f"✅ Создана папка: {upload_dir}")
else:
    print(f"✅ Папка уже существует: {upload_dir}")

# ===== GOOGLE RECAPTCHA =====
# ТЕСТОВЫЕ ключи (работают на любом домене)
RECAPTCHA_SITE_KEY = "6Lcz5iUsAAAAAGsKJ0-FI_Pfz2gbulSRcGXOfUWB"  # ключ для разработки 
RECAPTCHA_SECRET_KEY = "6Lcz5iUsAAAAALPlnt-rh-A7jH1ByaRu1AHMP_vJ"  # секретный ключ

# ===== ФУНКЦИИ =====
def verify_recaptcha(recaptcha_response):
    """Проверка Google reCAPTCHA"""
    print(f"🔍 Проверка reCAPTCHA...")
    
    # Тестовые ключи - всегда успех
    if RECAPTCHA_SECRET_KEY == "6LeIxAcTAAAAAGG-vFI1TnRWxMZNFuojJ4WifJWe":
        print("✅ Тестовая reCAPTCHA - успех")
        return True
    
    # Если нет ответа
    if not recaptcha_response:
        print("❌ Нет ответа reCAPTCHA")
        return False
    
    # Реальная проверка
    try:
        data = {
            'secret': RECAPTCHA_SECRET_KEY,
            'response': recaptcha_response
        }
        
        response = requests.post(
            'https://www.google.com/recaptcha/api/siteverify',
            data=data,
            timeout=5
        ).json()
        
        success = response.get('success', False)
        print(f"📊 reCAPTCHA результат: {success}")
        return success
        
    except Exception as e:
        print(f"⚠️ Ошибка проверки reCAPTCHA: {e}")
        return True  # В случае ошибки пропускаем

def allowed_file(filename):
    """Проверка расширения файла"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def classify_image(image_path):
    """Реальная классификация через ResNet50"""
    try:
        print(f"🧠 Начинаю классификацию изображения...")
        
        # 1. Загрузка модели ResNet50 (предобученной на ImageNet)
        print(f"📥 Загружаю модель ResNet50...")
        model = ResNet50(weights='imagenet')
        
        # 2. Загрузка и подготовка изображения
        print(f"🖼️ Обрабатываю изображение...")
        img = Image.open(image_path).convert('RGB')
        img = img.resize((224, 224))  # ResNet50 требует 224x224 пикселей
        
        # 3. Подготовка для нейросети
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)  # Добавляем batch dimension
        img_array = preprocess_input(img_array)  # Нормализация для ResNet50
        
        # 4. Предсказание нейросетью
        print(f"🤖 Нейросеть делает предсказание...")
        predictions = model.predict(img_array)
        
        # 5. Декодирование результатов (топ-5)
        decoded = decode_predictions(predictions, top=5)[0]
        
        # 6. Форматирование результатов
        results = []
        for _, class_name, probability in decoded:
            results.append({
                'class': class_name.replace('_', ' ').title(),
                'probability': round(probability * 100, 2)
            })
        
        print(f"✅ Классификация завершена. Найдено {len(results)} категорий.")
        return results
        
    except Exception as e:
        print(f"❌ Ошибка в нейросети: {e}")
        # Возвращаем пустой список в случае ошибки
        return [{'class': 'Ошибка классификации', 'probability': 0.0}]



def create_color_histogram(image_path):
    """Создает гистограмму распределения цветов RGB"""
    try:
        # 1. Открываем изображение
        img = Image.open(image_path)
        
        # 2. Преобразуем в numpy массив
        img_array = np.array(img)
        
        # 3. Разделяем на RGB каналы
        r_channel = img_array[:, :, 0].flatten()  # Красный (0-255)
        g_channel = img_array[:, :, 1].flatten()  # Зеленый (0-255)  
        b_channel = img_array[:, :, 2].flatten()  # Синий (0-255)
        
        # 4. Создаем график
        plt.figure(figsize=(10, 6))
        
        # Гистограмма для красного канала
        plt.hist(r_channel, bins=256, color='red', alpha=0.5, 
                label='Red', range=[0, 256])
        
        # Гистограмма для зеленого канала
        plt.hist(g_channel, bins=256, color='green', alpha=0.5,
                label='Green', range=[0, 256])
        
        # Гистограмма для синего канала
        plt.hist(b_channel, bins=256, color='blue', alpha=0.5,
                label='Blue', range=[0, 256])
        
        # Настройки графика
        plt.title('Распределение цветов (RGB гистограмма)', fontsize=14)
        plt.xlabel('Значение цвета (0-255)', fontsize=12)
        plt.ylabel('Количество пикселей', fontsize=12)
        plt.legend()
        plt.grid(alpha=0.3)
        
        # 5. Сохраняем график
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        histogram_name = f"histogram_{base_name}_{timestamp}.png"
        histogram_path = os.path.join(app.config['UPLOAD_FOLDER'], histogram_name)
        
        plt.savefig(histogram_path, dpi=100, bbox_inches='tight')
        plt.close()  # Закрываем график чтобы не накапливались в памяти
        
        print(f"📊 Гистограмма создана: {histogram_name}")
        return histogram_name
        
    except Exception as e:
        print(f"❌ Ошибка создания гистограммы: {e}")
        return None


def process_image(image_path):
    """Обработка изображения - сдвиг частей"""
    try:
        img = Image.open(image_path)
        width, height = img.size
        
        # Разбиваем на 4 части
        half_w, half_h = width // 2, height // 2
        parts = [
            img.crop((0, 0, half_w, half_h)),
            img.crop((half_w, 0, width, half_h)),
            img.crop((0, half_h, half_w, height)),
            img.crop((half_w, half_h, width, height))
        ]
        
        # Сдвигаем
        shifted = [parts[2], parts[0], parts[3], parts[1]]
        
        # Собираем
        new_img = Image.new('RGB', (width, height))
        new_img.paste(shifted[0], (0, 0))
        new_img.paste(shifted[1], (half_w, 0))
        new_img.paste(shifted[2], (0, half_h))
        new_img.paste(shifted[3], (half_w, half_h))
        
        # Сохраняем
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        processed_name = f"processed_{base_name}_{timestamp}.jpg"
        processed_path = os.path.join(app.config['UPLOAD_FOLDER'], processed_name)
        
        new_img.save(processed_path, 'JPEG', quality=85)
        return processed_name
        
    except Exception as e:
        print(f"❌ Ошибка обработки: {e}")
        raise

# ===== МАРШРУТЫ =====
@app.route('/')
def index():
    return render_template('index.html', 
                         site_key=RECAPTCHA_SITE_KEY,
                         max_size_mb=16)

@app.route('/upload', methods=['POST'])
def upload_image():
    try:
        print("📤 Начало загрузки файла...")
        
        # 1. Проверка reCAPTCHA
        recaptcha_response = request.form.get('g-recaptcha-response')
        if not verify_recaptcha(recaptcha_response):
            flash('❌ Пожалуйста, подтвердите что вы не робот!', 'error')
            return redirect('/')
        
        print("✅ reCAPTCHA пройдена")
        
        # 2. Проверка файла
        if 'file' not in request.files:
            flash('❌ Файл не выбран', 'error')
            return redirect('/')
        
        file = request.files['file']
        
        if file.filename == '':
            flash('❌ Файл не выбран', 'error')
            return redirect('/')
        
        if not allowed_file(file.filename):
            flash('❌ Разрешены только PNG, JPG, JPEG, GIF, BMP', 'error')
            return redirect('/')
        
        print(f"📄 Файл получен: {file.filename}")
        
        # 3. Сохранение файла
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name, ext = os.path.splitext(filename)
        unique_name = f"{name}_{timestamp}{ext}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_name)
        
        file.save(file_path)
        print(f"💾 Файл сохранен: {file_path}")
        
        # 4. Обработка
        processed_name = process_image(file_path)
        histogram_name = create_color_histogram(file_path)
        results = classify_image(file_path)
        
        print(f"✅ Обработка завершена!")
        
        return render_template('result.html',
                             original_image=unique_name,
                             processed_image=processed_name,
                             classification_results=results)
        
    except Exception as e:
        print(f"❌ Ошибка в upload: {e}")
        flash(f'❌ Ошибка: {str(e)}', 'error')
        return redirect('/')

@app.route('/uploads/<filename>')
def serve_file(filename):
    """Отдача загруженных файлов"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/health')
def health():
    return "OK", 200

# ===== ЗАПУСК =====
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    app.run(host='0.0.0.0', port=port, debug=debug)




