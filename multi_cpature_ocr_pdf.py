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

def live_feed():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield frame

def get_live_feed():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    if ret:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame
    else:
        return None

with gr.Blocks() as demo:

    input_files = gr.File(label='이미지 선택', type='filepath', file_count='multiple')
    capture_button = gr.Button("사진 찍기")
    live_feed_display = gr.Image(label='실시간 카메라 피드')
    captured_image_display = gr.Image(label='캡처된 이미지')
    output_images = gr.Gallery(label='결과 이미지', type='pil', interactive=False)

    def process_images(uploaded_files, capture):
        image_paths = []
        captured_image_path = None
        if uploaded_files:
            image_paths.extend(uploaded_files)
        if capture:
            captured_image_path = capture_image()
            image_paths.append(captured_image_path)
        output_images = change_images(image_paths)
        return output_images, captured_image_path

    input_files.change(fn=process_images, inputs=[input_files, capture_button], outputs=[output_images, captured_image_display])
    capture_button.click(fn=process_images, inputs=[input_files, capture_button], outputs=[output_images, captured_image_display])

    demo.load(get_live_feed, inputs=None, outputs=live_feed_display, every=0)

demo.launch()
