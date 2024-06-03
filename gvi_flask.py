# from flask import Flask, request, jsonify, send_file
# from flask_cors import CORS
# import cv2
# import numpy as np
# import tempfile
# import os
# import json
# from PIL import Image, ExifTags
# import math

# app = Flask(__name__)
# CORS(app)

# def allowed_file(filename):
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in ['jpg', 'png', 'jpeg', 'bmp']

# def get_exif_data(img_path):
#     try:
#         img = Image.open(img_path)
#         exif_data = img.getexif()
#         return {ExifTags.TAGS[k]: v for k, v in exif_data.items() if k in ExifTags.TAGS} if exif_data else {}
#     except IOError:
#         return {}

# def ifdrational_to_float(ifdrational):
#     try:
#         return float(ifdrational.numerator) / float(ifdrational.denominator)
#     except (TypeError, AttributeError, ZeroDivisionError) as e:
#         print(f"Error converting IFDRational to float: {e}")
#         return None

# def get_dpi_from_exif(exif_data):
#     try:
#         x_dpi = ifdrational_to_float(exif_data.get("XResolution", 1))
#         y_dpi = ifdrational_to_float(exif_data.get("YResolution", 1))
#         return x_dpi, y_dpi
#     except (KeyError, ValueError, TypeError) as e:
#         print(f"Error getting DPI from EXIF: {e}")
#         return None, None

# def parse_json_data(json_data):
#     try:
#         data = json.loads(json_data)
#         return data
#     except json.JSONDecodeError:
#         return {}

# def calculate_mm_per_pixel(points, known_distance):
#     if len(points) < 2:
#         print("Not enough points for scale calculation. At least two points are required.")
#         return None

#     p1 = np.array([points[0]['x'], points[0]['y']])
#     p2 = np.array([points[1]['x'], points[1]['y']])
#     pixel_distance = np.linalg.norm(p2 - p1)
    
#     # Calculate mm per pixel using the known distance and pixel distance
#     mm_per_pixel = known_distance / pixel_distance

#     print(f"Pixel distance between first two points: {pixel_distance} pixels")
#     print(f"mm per pixel: {mm_per_pixel}")

#     return mm_per_pixel

# def calculate_physical_distances(lines, mm_per_pixel):
#     if not mm_per_pixel:
#         print("mm per pixel value is invalid.")
#         return []

#     distances = []

#     for line in lines:
#         p1 = np.array([line['start']['x'], line['start']['y']])
#         p2 = np.array([line['end']['x'], line['end']['y']])
#         pixel_distance = np.linalg.norm(p2 - p1)
#         physical_distance = pixel_distance * mm_per_pixel
#         distances.append(physical_distance)

#         print(f"Pixel Distance between points: {pixel_distance} pixels")
#         print(f"Physical Distance: {physical_distance} mm")

#     return distances

# def process_image(img_path):
#     img = cv2.imread(img_path)
#     if img is None:
#         return None, "Failed to read image"
#     mean_values = calculate_rgb_means(img)
#     hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#     green_pixel_count, green_mask = detect_green_pixels(hsv_img)
#     processed_image_path = enhance_green_pixels_with_contours(img, green_mask)
#     return mean_values + (green_pixel_count, processed_image_path)

# def calculate_rgb_means(img):
#     B, G, R = cv2.split(img)
#     return np.mean(R), np.mean(G), np.mean(B)

# def detect_green_pixels(hsv_img):
#     lower_green = np.array([25, 40, 40])
#     upper_green = np.array([100, 255, 255])
#     green_mask = cv2.inRange(hsv_img, lower_green, upper_green)
#     contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     return sum(cv2.contourArea(contour) for contour in contours), green_mask

# def enhance_green_pixels_with_contours(img, green_mask):
#     enhanced_img = cv2.bitwise_and(img, img, mask=green_mask)
#     contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     cv2.drawContours(enhanced_img, contours, -1, (0, 255, 0), 3)
#     _, temp_file_path = tempfile.mkstemp(suffix='.png')
#     cv2.imwrite(temp_file_path, enhanced_img)
#     return temp_file_path

# @app.route('/upload', methods=['POST'])
# def upload_image():
#     if 'image' not in request.files:
#         return jsonify({'error': 'No image part'}), 400
#     file = request.files['image']
#     if file.filename == '':
#         return jsonify({'error': 'No selected image'}), 400
#     if not allowed_file(file.filename):
#         return jsonify({'error': 'Unsupported file format'}), 400

#     temp_path = tempfile.mkstemp(suffix='.' + file.filename.rsplit('.', 1)[1].lower())[1]
#     file.save(temp_path)

#     # EXIF 데이터 파싱
#     exif_data = get_exif_data(temp_path)
#     x_dpi, y_dpi = get_dpi_from_exif(exif_data)
#     print(f"Extracted DPI values: x_dpi={x_dpi}, y_dpi={y_dpi}")
#     if x_dpi is None or y_dpi is None:
#         os.remove(temp_path)
#         return jsonify({'error': 'Failed to extract DPI from EXIF data'}), 400

#     # JSON 데이터 파싱 및 점들 추출
#     json_data = request.form.get('line_data', '{}')
#     data = parse_json_data(json_data)
#     lines = data.get('lines', [])
#     points = [line['start'] for line in lines] + [line['end'] for line in lines]

#     if len(points) < 2:
#         os.remove(temp_path)
#         return jsonify({'error': 'Not enough points for distance calculation'}), 400

#     # 첫 두 점 사이의 실제 거리(사용자가 입력한 값)
#     known_distance = float(request.form.get('knownDistance', '0'))
#     if known_distance <= 0:
#         os.remove(temp_path)
#         return jsonify({'error': 'Invalid known distance'}), 400

#     # 픽셀 당 mm 계산
#     mm_per_pixel = calculate_mm_per_pixel(points, known_distance)
#     if not mm_per_pixel:
#         os.remove(temp_path)
#         return jsonify({'error': 'Failed to calculate mm per pixel'}), 400

#     # 로그 추가: 수신된 점 데이터와 계산된 거리 정보 출력
#     print(f"Received lines data: {lines}")
#     distances = calculate_physical_distances(lines, mm_per_pixel)
#     print("Calculated Physical Distances:", distances)

#     processing_results = process_image(temp_path)
#     if processing_results[1] != "Failed to read image":
#         response = {
#             'mean_R': processing_results[0],
#             'mean_G': processing_results[1],
#             'mean_B': processing_results[2],
#             'green_pixel_count': processing_results[3],
#             'distances': distances,
#             'processed_image_url': request.host_url + 'image/' + os.path.basename(processing_results[4])
#         }
#         os.remove(temp_path)  # Cleanup temporary files
#         return jsonify(response)
#     else:
#         os.remove(temp_path)
#         return jsonify({'error': processing_results[1]}), 400

# @app.route('/image/<filename>')
# def get_image(filename):
#     return send_file(tempfile.gettempdir() + '/' + filename, mimetype='image/jpeg')

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', debug=True, port=8080)

# backup & old version code

#########################################################################################################################################


from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import cv2
import numpy as np
import tempfile
import os
import json
from PIL import Image, ExifTags

app = Flask(__name__)
CORS(app)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ['jpg', 'png', 'jpeg', 'bmp']

def get_exif_data(img_path):
    try:
        img = Image.open(img_path)
        exif_data = img.getexif()
        return {ExifTags.TAGS[k]: v for k, v in exif_data.items() if k in ExifTags.TAGS} if exif_data else {}
    except IOError:
        return {}

def ifdrational_to_float(ifdrational):
    try:
        return float(ifdrational.numerator) / float(ifdrational.denominator)
    except (TypeError, AttributeError, ZeroDivisionError) as e:
        print(f"Error converting IFDRational to float: {e}")
        return None

def get_dpi_from_exif(exif_data):
    try:
        x_dpi = ifdrational_to_float(exif_data.get("XResolution", 1))
        y_dpi = ifdrational_to_float(exif_data.get("YResolution", 1))
        return x_dpi, y_dpi
    except (KeyError, ValueError, TypeError) as e:
        print(f"Error getting DPI from EXIF: {e}")
        return None, None

def parse_json_data(json_data):
    try:
        data = json.loads(json_data)
        return data
    except json.JSONDecodeError:
        return {}

def calculate_mm_per_pixel(line, known_distance):
    p1 = np.array([line['start']['x'], line['start']['y']])
    p2 = np.array([line['end']['x'], line['end']['y']])
    pixel_distance = np.linalg.norm(p2 - p1)
    
    # knownDistance를 사용하여 mm per pixel 계산
    mm_per_pixel = known_distance / pixel_distance

    print(f"Pixel distance between points: {pixel_distance} pixels")
    print(f"mm per pixel: {mm_per_pixel}")

    return mm_per_pixel

def calculate_physical_distances(lines, mm_per_pixel):
    if not mm_per_pixel:
        print("mm per pixel value is invalid.")
        return []

    distances = []

    for line in lines:
        p1 = np.array([line['start']['x'], line['start']['y']])
        p2 = np.array([line['end']['x'], line['end']['y']])
        pixel_distance = np.linalg.norm(p2 - p1)
        physical_distance = pixel_distance * mm_per_pixel
        distances.append(physical_distance)

        print(f"Pixel Distance between points: {pixel_distance} pixels")
        print(f"Physical Distance: {physical_distance} mm")

    return distances

def process_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return None, "Failed to read image"
    
    # 이미지 전처리: 가우시안 블러링 적용
    blurred_img = cv2.GaussianBlur(img, (5, 5), 0)
    
    # HSV 변환 및 녹색 픽셀 검출
    hsv_img = cv2.cvtColor(blurred_img, cv2.COLOR_BGR2HSV)
    green_pixel_count, green_mask = detect_green_pixels(hsv_img)
    processed_image_path = enhance_green_pixels_with_contours(img, green_mask)
    
    # RGB 평균 계산
    mean_values = calculate_rgb_means(img)
    return mean_values + (green_pixel_count, processed_image_path)

def calculate_rgb_means(img):
    B, G, R = cv2.split(img)
    return np.mean(R), np.mean(G), np.mean(B)

def detect_green_pixels(hsv_img):
    lower_green = np.array([25, 40, 40])
    upper_green = np.array([100, 255, 255])
    green_mask = cv2.inRange(hsv_img, lower_green, upper_green)
    contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return sum(cv2.contourArea(contour) for contour in contours), green_mask

def enhance_green_pixels_with_contours(img, green_mask):
    enhanced_img = cv2.bitwise_and(img, img, mask=green_mask)
    contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(enhanced_img, contours, -1, (0, 255, 0), 3)
    _, temp_file_path = tempfile.mkstemp(suffix='.png')
    cv2.imwrite(temp_file_path, enhanced_img)
    return temp_file_path

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image part'}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected image'}), 400
    if not allowed_file(file.filename):
        return jsonify({'error': 'Unsupported file format'}), 400

    temp_path = tempfile.mkstemp(suffix='.' + file.filename.rsplit('.', 1)[1].lower())[1]
    file.save(temp_path)

    # EXIF 데이터 파싱
    exif_data = get_exif_data(temp_path)
    x_dpi, y_dpi = get_dpi_from_exif(exif_data)
    print(f"Extracted DPI values: x_dpi={x_dpi}, y_dpi={y_dpi}")
    if x_dpi is None or y_dpi is None:
        os.remove(temp_path)
        return jsonify({'error': 'Failed to extract DPI from EXIF data'}), 400

    # JSON 데이터 파싱 및 점들 추출
    json_data = request.form.get('line_data', '{}')
    data = parse_json_data(json_data)
    lines = data.get('lines', [])
    points = [line['start'] for line in lines] + [line['end'] for line in lines]

    if len(points) < 2:
        os.remove(temp_path)
        return jsonify({'error': 'Not enough points for distance calculation'}), 400

    # 첫 두 점 사이의 실제 거리(사용자가 입력한 값)
    known_distance = float(request.form.get('knownDistance', '0'))
    if known_distance <= 0:
        os.remove(temp_path)
        return jsonify({'error': 'Invalid known distance'}), 400

    # 첫 번째 라인을 기준으로 mm per pixel 계산
    mm_per_pixel = calculate_mm_per_pixel(lines[0], known_distance)
    if not mm_per_pixel:
        os.remove(temp_path)
        return jsonify({'error': 'Failed to calculate mm per pixel'}), 400

    # 로그 추가: 수신된 점 데이터와 계산된 거리 정보 출력
    print(f"Received lines data: {lines}")
    distances = calculate_physical_distances(lines, mm_per_pixel)
    print("Calculated Physical Distances:", distances)

    processing_results = process_image(temp_path)
    if processing_results[1] != "Failed to read image":
        response = {
            'mean_R': processing_results[0],
            'mean_G': processing_results[1],
            'mean_B': processing_results[2],
            'green_pixel_count': processing_results[3],
            'distances': distances,
            'processed_image_url': request.host_url + 'image/' + os.path.basename(processing_results[4])
        }
        os.remove(temp_path)  # Cleanup temporary files
        return jsonify(response)
    else:
        os.remove(temp_path)
        return jsonify({'error': processing_results[1]}), 400

@app.route('/image/<filename>')
def get_image(filename):
    return send_file(tempfile.gettempdir() + '/' + filename, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=8080)
