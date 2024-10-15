import requests
import gradio as gr
from PIL import Image, ImageDraw, ImageFont
import json
import platform
from pptx import Presentation
from pptx.util import Inches, Pt
from fpdf import FPDF


# OCR API request
def request_vision(image_path):
    endpoint = f"https://computervision06.cognitiveservices.azure.com/computervision/imageanalysis:analyze"
    api_key = "c7703f4920df4cda8965e7e170e94214"

    headers = {
        "Content-Type": "application/octet-stream",
        "Ocp-Apim-Subscription-Key": api_key,
    }

    params = {"api-version": "2024-02-01", "features": "read,caption"}

    with open(image_path, "rb") as image:
        image_data = image.read()

    response = requests.post(endpoint, headers=headers, params=params, data=image_data)

    response_json = response.json()
    return response_json


# Process image and perform OCR
def process_image(image_path):
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)

    font_size = 20
    if platform.system() == "Darwin":
        font = ImageFont.truetype("AppleGothic.ttf", size=font_size)
    elif platform.system() == "Windows":
        font = ImageFont.truetype("malgun.ttf", size=font_size)
    else:
        font = ImageFont.load_default(size=font_size)

    response_json = request_vision(image_path=image_path)
    output_data = []

    if "readResult" in response_json and "blocks" in response_json["readResult"]:
        block_list = response_json["readResult"]["blocks"]
        for block in block_list:
            line_list = block["lines"]
            for line in line_list:
                text = line["text"]
                bounding_polygon = line["boundingPolygon"]
                polygon = list(map(lambda p: (p["x"], p["y"]), bounding_polygon))
                draw.polygon(polygon, outline="red", fill=None, width=3)
                draw.text(
                    (bounding_polygon[3]["x"], bounding_polygon[3]["y"] + 3),
                    text=text,
                    fill="green",
                    font=font,
                )
                output_data.append(
                    {
                        "text": text,
                        "bounding_polygon": bounding_polygon,
                        "font_size": font_size,  # Store font size information
                    }
                )

    with open("ocr_output.json", "w") as json_file:
        json.dump(output_data, json_file, indent=4)

    return image, output_data


# Generate PPT from OCR data
def pixels_to_inches(pixels, dpi=96):
    return pixels / dpi


def generate_ppt(ocr_data, output_path="ocr_result.pptx"):
    prs = Presentation()
    slide_layout = prs.slide_layouts[5]  # Blank slide layout

    # Set slide size to a user-friendly aspect ratio
    prs.slide_width = Inches(13.33)  # 16:9 ratio
    prs.slide_height = Inches(7.5)

    max_x = max(point["x"] for item in ocr_data for point in item["bounding_polygon"])
    max_y = max(point["y"] for item in ocr_data for point in item["bounding_polygon"])

    scale_x = prs.slide_width.inches / pixels_to_inches(max_x)
    scale_y = prs.slide_height.inches / pixels_to_inches(max_y)

    def add_text_to_slide(slide, items, y_offset=0):
        for item in items:
            text = item["text"]
            polygon = item["bounding_polygon"]
            font_size = item["font_size"]

            x_min = min(point["x"] for point in polygon)
            y_min = min(point["y"] for point in polygon) - y_offset
            x_max = max(point["x"] for point in polygon)
            y_max = max(point["y"] for point in polygon) - y_offset

            left = pixels_to_inches(x_min) * scale_x
            top = pixels_to_inches(y_min) * scale_y
            width = pixels_to_inches(x_max - x_min) * scale_x
            height = pixels_to_inches(y_max - y_min) * scale_y

            textbox = slide.shapes.add_textbox(
                Inches(left), Inches(top), Inches(width), Inches(height)
            )
            text_frame = textbox.text_frame
            text_frame.text = text

            for paragraph in text_frame.paragraphs:
                for run in paragraph.runs:
                    run.font.size = Pt(font_size)
                    while textbox.height > height or textbox.width > width:
                        run.font.size -= Pt(1)
                        if run.font.size <= Pt(10):  # Minimum font size is 10pt
                            run.font.size = Pt(10)
                            break

    current_slide = prs.slides.add_slide(slide_layout)
    current_height = 0
    max_height = prs.slide_height.inches
    slide_count = 0
    y_offset = 0  # Initialize y_offset

    for item in ocr_data:
        x_min = min(point["x"] for point in item["bounding_polygon"])
        y_min = min(point["y"] for point in item["bounding_polygon"])
        x_max = max(point["x"] for point in item["bounding_polygon"])
        y_max = max(point["y"] for point in item["bounding_polygon"])

        height = pixels_to_inches(y_max - y_min) * scale_y
        width = pixels_to_inches(x_max - x_min) * scale_x

        if current_height + height > max_height:
            slide_count += 1
            current_slide = prs.slides.add_slide(slide_layout)
            current_height = 0  # Reset the current height for the new slide
            y_offset = (
                slide_count * max_height * 96 / scale_y
            )  # Adjust y-coordinates for the new slide

        add_text_to_slide(current_slide, [item], y_offset)
        current_height += height

    prs.save(output_path)
    return output_path


# Generate PDF from OCR data
def generate_pdf(ocr_data, output_path="ocr_result.pdf"):
    pdf = FPDF()
    pdf.add_page()
    pdf.add_font("NotoSans", "", "NotoSansKR-VariableFont_wght.ttf", uni=True)
    pdf.set_font("NotoSans", "", 12)

    for item in ocr_data:
        pdf.cell(200, 10, txt=item["text"], ln=True)

    pdf.output(output_path)
    return output_path


# Gradio interface
def ocr_interface(image_path):
    processed_image, ocr_data = process_image(image_path)
    ppt_path = generate_ppt(ocr_data)
    pdf_path = generate_pdf(ocr_data)
    return processed_image, ppt_path, pdf_path


with gr.Blocks() as demo:
    input_image = gr.Image(label="Select Image", type="filepath", width=800)
    output_image = gr.Image(
        label="Result Image", type="pil", interactive=False, width=800
    )
    download_ppt_button = gr.File(label="Download PPT", file_count="single")
    download_pdf_button = gr.File(label="Download PDF", file_count="single")

    def update_output(image_path):
        processed_image, ppt_path, pdf_path = ocr_interface(image_path)
        return processed_image, ppt_path, pdf_path

    input_image.change(
        fn=update_output,
        inputs=[input_image],
        outputs=[output_image, download_ppt_button, download_pdf_button],
    )

demo.launch()
