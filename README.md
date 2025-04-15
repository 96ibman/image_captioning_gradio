
# Describe Any Image Using AI

🚀 Try it live on [Hugging Face Spaces](https://huggingface.co/spaces/Ibrahimnasser/image_captioning)


This is a simple Gradio-powered app that uses a pre-trained vision-language model to describe the content of images. Upload any image and see how AI interprets the scene.

---
## What is Image Captioning?

**Image captioning** is a task where a deep learning model generates a textual description of an image. It combines computer vision and natural language processing in one pipeline.

---

## Packages

```python
import gradio as gr
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
```

## 🤗 BlipProcessor
I used `BlipProcessor` which is a class from 🤗 `transformers` library. It is designed for Bootstrapping Language-Image Pretraining (BLIP) models. It basically combines image and text preprocessing
```python
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
```

## `generate_caption` Function
```python
def generate_caption(image):
    if image is None:
        return "Please upload an image to generate a caption."
    inputs = processor(images=image, return_tensors="pt")
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption
```

## Gradio Interface
```python
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
```

---
## Run Locally
```bash
git clone https://github.com/96ibman/image_captioning_gradio.git
```

```bash
cd image_captioning_gradio
```

```bash
python -m venv venv
```

```bash
venv/Scripts/activate
```
```bash
pip install -r requirements.txt
```
```bash
python app.py
```
---

## About me
[Website](https://96ibman.github.io/ibrahim-nasser/)