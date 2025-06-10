from flask import Flask, request, render_template, Response, send_from_directory
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import cv2
import pickle

# Initialize Flask app
app = Flask(__name__)

# Load the saved model
model = load_model('e_waste_classifier.h5')

# Load class indices
with open('class_indices.pkl', 'rb') as f:
    class_indices = pickle.load(f)
class_labels = list(class_indices.keys())

# Function to predict and draw bounding boxes on images
def predict_and_draw_boxes(img_path):
    original_img = cv2.imread(img_path)

    # Resize for model input
    resized_image = cv2.resize(original_img, (150, 150))
    image_array = image.img_to_array(resized_image)
    image_array = np.expand_dims(image_array, axis=0)
    image_array /= 255.0

    # Predict class
    prediction = model.predict(image_array)
    predicted_class = np.argmax(prediction, axis=1)
    label = class_labels[predicted_class[0]]

    # Draw bounding box
    color = (0, 255, 0) if label == 'E-Waste' else (0, 0, 255)
    height, width, _ = original_img.shape
    cv2.rectangle(original_img, (0, 0), (width - 1, height - 1), color, 5)
    cv2.putText(original_img, label, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 3)

    # Save result image
    dest_folder = 'e_waste' if label == 'E-Waste' else 'non_e_waste'
    os.makedirs(f'uploads/{dest_folder}', exist_ok=True)
    result_path = os.path.join(f'uploads/{dest_folder}', f'result_{os.path.basename(img_path)}')
    cv2.imwrite(result_path, original_img)

    return result_path, label

# Function for real-time detection
# Function for real-time detection with bounding boxes for both E-Waste and Non E-Waste
def detect_in_frame(frame):
    resized_image = cv2.resize(frame, (150, 150))
    image_array = image.img_to_array(resized_image)
    image_array = np.expand_dims(image_array, axis=0)
    image_array /= 255.0

    # Predict class
    prediction = model.predict(image_array)
    predicted_class = np.argmax(prediction, axis=1)
    label = class_labels[predicted_class[0]]

    # Determine box color based on classification
    if label == 'E-Waste':
        color = (0, 255, 0)  # Green for E-Waste
    else:
        color = (0, 0, 255)  # Red for Non E-Waste

    # Draw bounding box and label on the frame
    cv2.rectangle(frame, (50, 50), (550, 400), color, 2)
    cv2.putText(frame, label, (60, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    return frame


# Real-time detection generator
def generate_frames():
    camera = cv2.VideoCapture(0)
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            frame = detect_in_frame(frame)
            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    camera.release()

# Route for uploading and classifying images
@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        if 'files' not in request.files:
            return "No files uploaded.", 400
        
        files = request.files.getlist('files')
        if len(files) == 0:
            return "No files selected.", 400
        
        for file in files:
            file_path = os.path.join('uploads', file.filename)
            os.makedirs('uploads', exist_ok=True)
            file.save(file_path)
            predict_and_draw_boxes(file_path)

        return render_template('success.html')
    return render_template('upload.html')

# Route for real-time detection
@app.route('/real_time')
def real_time():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Routes for displaying categorized images
@app.route('/e_waste')
def show_e_waste():
    folder = 'e_waste'
    images = [
        (file, f'result_{file}')
        for file in os.listdir(f'uploads/{folder}')
        if file.endswith('.jpg') or file.endswith('.png')
    ]
    total_count = len(images)
    return render_template('category.html', images=images, category="E-Waste", folder=folder, total_count=total_count)

@app.route('/non_e_waste')
def show_non_e_waste():
    folder = 'non_e_waste'
    images = [
        (file, f'result_{file}')
        for file in os.listdir(f'uploads/{folder}')
        if file.endswith('.jpg') or file.endswith('.png')
    ]
    total_count = len(images)
    return render_template('category.html', images=images, category="Non E-Waste", folder=folder, total_count=total_count)

# Serve uploaded images
@app.route('/uploads/<folder>/<filename>')
def serve_file(folder, filename):
    return send_from_directory(f'uploads/{folder}', filename)

# HTML templates
os.makedirs('templates', exist_ok=True)

# Upload page
upload_page = '''
<!DOCTYPE html>
<html>
<head>
    <title>Upload Images</title>
    <style>
        body {
            background-image: url("{{ url_for('static', filename='download.jpeg') }}");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
            font-family: Arial, sans-serif;
            color: white;
            text-align: center;
            margin: 0;
            padding: 0;
        }
        h1 {
            color:#FF5F1F;
            margin-top: 50px;
        }
        form {
            margin: 20px auto;
            max-width: 90%;
            padding: 10px;
        }
        input, button {
            padding: 10px;
            font-size: 16px;
            margin: 10px 0;
        }
        a {
            display: block;
            margin-top: 20px;
            color: red;
            font-size: 18px;
            text-decoration: none;
        }
        a:hover {
            text-decoration: underline;
        }

        /* Media query for smaller screens */
        @media (max-width: 768px) {
            h1 {
                color:#FF5F1F;
                font-size: 34px;
            }
            input, button {
                font-size: 14px;
            }
            a {
                font-size: 16px;
            }
        }
    </style>
</head>
<body>
    <h1>Upload Images</h1>
    <form method="post" enctype="multipart/form-data">
        <input type="file" name="files" multiple>
        <button type="submit">Upload</button>
    </form>
    <a href="/real_time">Real-Time Detection</a>
</body>
</html>

'''
with open('templates/upload.html', 'w') as f:
    f.write(upload_page)

# Success page
success_page = '''
<!DOCTYPE html>
<html>
<head>
    <title>Upload Successful</title>
    <style>
        body {
            background-image: url("{{ url_for('static', filename='download2.jpeg') }}");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
            font-family: Arial, sans-serif;
            color: white;
            text-align: center;
            margin: 0;
            padding: 0;
        }
        h1 {
            color: #0047AB;
            margin-top: 50px;
        }
        a {
            display: block;
            margin: 20px;
            color: red;
            font-size: 18px;
            text-decoration: none;
        }
        a:hover {
            text-decoration: underline;
        }

        /* Media query for smaller screens */
        @media (max-width: 768px) {
            h1 {
                font-size: 24px;
            }
            a {
                font-size: 16px;
            }
        }
    </style>
</head>
<body>
    <h1>Upload Successful</h1>
    <a href="/e_waste">View E-Waste Images</a>
    <a href="/non_e_waste">View Non E-Waste Images</a>
</body>
</html>

'''
with open('templates/success.html', 'w') as f:
    f.write(success_page)

# Category page
category_page = '''
<!DOCTYPE html>
<html>
<head>
    <title>{{ category }} Images</title>
    <style>
        body {
            background-image: url("{{ url_for('static', filename='download1.jpeg') }}");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
            font-family: Arial, sans-serif;
            color: white;
            margin: 0;
            padding: 0;
        }
       h1, h3 {
            color: black;
            margin-top: 20px;
              text-align: center;
        }
        div {
            margin: 10px auto;
            text-align: center;
            padding: 10px;
        }
        img {
            max-width: 90%;
            height: auto;
            margin: 10px auto;
            border: 2px solid white;
            border-radius: 10px;
        }
        a {
            color: red;
            font-size: 18px;
            text-decoration: none;
            margin-top: 30px;
            display: inline-block;
        }
        a:hover {
            text-decoration: underline;
        }

        /* Media query for smaller screens */
        @media (max-width: 768px) {
            h1 {
                color:black;
                font-size: 20px;
            }
            img {
                max-width: 100%;
            }
            a {
                font-size: 14px;
            }
        }
    </style>
</head>
<body>
    <h1>{{ category }} Images</h1>
    {% for original_image, result_image in images %}
        <div>
            <h3>Uploaded Image</h3>
            <img src="{{ url_for('serve_file', folder=folder, filename=original_image) }}" alt="Uploaded Image">
        </div>
    {% endfor %}
    <h3>Total {{ category }} Images: {{ total_count }}</h3>
    <a href="/">Back to Upload</a>
</body>
</html>

'''
with open('templates/category.html', 'w') as f:
    f.write(category_page)

if __name__ == '__main__':
    os.makedirs('uploads/e_waste', exist_ok=True)
    os.makedirs('uploads/non_e_waste', exist_ok=True)
    app.run(host='0.0.0.0', port=5000, debug=True)
