import gradio as gr
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def generate_caption(image):
    if image is None:
        return "Please upload an image to generate a caption."
    inputs = processor(images=image, return_tensors="pt")
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

iface = gr.Interface(
    fn=generate_caption, 
    inputs=gr.Image(type="pil", label="Upload Image"), 
    outputs="text", 
    live=True,
    title="Image Captioning App",
    description="Upload an image and get a description of what the image contains.",
    allow_flagging="never"
)

iface.launch()
