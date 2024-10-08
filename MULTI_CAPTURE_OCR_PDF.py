import requests
import gradio as gr
from PIL import Image, ImageDraw, ImageFont
import json
import platform
from fpdf import FPDF
import cv2
import numpy as np

def request_vision(image_path):
    endpoint = f"https://computervision06.cognitiveservices.azure.com/computervision/imageanalysis:analyze"
    api_key = "c7703f4920df4cda8965e7e170e94214"

    headers = {
        "Content-Type": "application/octet-stream",
        "Ocp-Apim-Subscription-Key": api_key
    }

    params = {
        'api-version': '2024-02-01',
        'features': 'read,caption'
    }

    with open(image_path, 'rb') as image:
        image_data = image.read()

    response = requests.post(endpoint,
                             headers=headers,
                             params=params,
                             data=image_data)
    
    response_json = response.json()
    return response_json

def change_images(image_paths):

    if not image_paths:
        return "No images uploaded."

    output_images = []
    output_data = []

    for image_path in image_paths:
        image = Image.open(image_path)
        draw = ImageDraw.Draw(image)

        font_size = 20
        if platform.system() == 'Darwin':
            font = ImageFont.truetype('AppleGothic.ttf', size=font_size)
        elif platform.system() == 'Windows':
            font = ImageFont.truetype('malgun.ttf', size=font_size)
        else:
            font = ImageFont.load_default(size=font_size)
        
        response_json = request_vision(image_path=image_path)
        
        if 'readResult' in response_json and 'blocks' in response_json['readResult']:
            block_list = response_json['readResult']['blocks']

            for block in block_list:
                line_list = block['lines']
                for line in line_list:
                    text = line['text']
                    bounding_polygon = line['boundingPolygon']
                    polygon = list(map(lambda p: (p['x'], p['y']), bounding_polygon))
                    draw.polygon(polygon, outline='red', fill=None, width=3)
                    draw.text((bounding_polygon[3]['x'], bounding_polygon[3]['y'] + 3), text=text, fill='green', font=font)
                    
                    output_data.append({
                        "text": text,
                        "bounding_polygon": bounding_polygon
                    })

        output_images.append(image)

    with open('ocr_output.json', 'w') as json_file:
        json.dump(output_data, json_file, indent=4)

    create_pdf()  

    return output_images

def create_pdf():
    with open('ocr_output.json', 'r', encoding='utf-8') as file:
        data = json.load(file)

    pdf = FPDF()
    pdf.add_page()
    pdf.add_font('NotoSans', '', 'NotoSansKR-VariableFont_wght.ttf', uni=True)
    pdf.set_font('NotoSans', '', 12)

    for item in data:
        pdf.cell(200, 10, txt=item['text'], ln=True)

    pdf.output("ocr_result.pdf")
    print("텍스트가 성공적으로 ocr_result.pdf로 변환되었습니다.")

def capture_image():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return "Could not open webcam"
    
    ret, frame = cap.read()
    if not ret:
        return "Failed to capture image"
    
    image_path = "captured_image.png"
    cv2.imwrite(image_path, frame)
    cap.release()
    cv2.destroyAllWindows()
    return image_path

def detect_faces(image):
    cascPath = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    faceCascade = cv2.CascadeClassifier(cascPath)

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

with gr.Blocks() as demo:
    with gr.Row():
        # webcam_input = gr.Image(sources='webcam', streaming=True)
        output_image = gr.Image(streaming=True)
        captured_image_display = gr.Image(label='Captured Image')
        output_images = gr.Gallery(label='Result Images', type='pil', interactive=False)
        input_files = gr.File(label='Upload Images', type='filepath', file_count='multiple')

    def process_images(image, uploaded_files):
        image_paths = []
        if uploaded_files:
            image_paths.extend(uploaded_files)
        if image is not None:
            image_path = "captured_image.png"
            cv2.imwrite(image_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            image_paths.append(image_path)
        output_images = change_images(image_paths)
        return output_images, image

    # webcam_input.stream(fn=detect_faces, inputs=webcam_input, outputs=output_image)
    output_image.change(fn=process_images, inputs=[output_image, input_files], outputs=[output_images, captured_image_display])
    input_files.change(fn=process_images, inputs=[output_image, input_files], outputs=[output_images, captured_image_display])

demo.launch()
