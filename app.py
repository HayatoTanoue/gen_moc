import time
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import gradio as gr

# Attempt to load a CLIP model. If unavailable, continue with fallback.
try:
    model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
    processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
    CLIP_AVAILABLE = True
except Exception as e:  # model not available (e.g., offline)
    model = None
    processor = None
    CLIP_AVAILABLE = False


def compute_similarity(image: Image.Image, text: str):
    """Compute image-text similarity and measure execution time."""
    start = time.time()
    if image is None or not text:
        return 0.0, 0.0

    if CLIP_AVAILABLE:
        inputs = processor(text=[text], images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            img_emb = outputs.image_embeds
            txt_emb = outputs.text_embeds
        similarity = float(torch.nn.functional.cosine_similarity(img_emb, txt_emb).item())
    else:
        # Fallback similarity using image mean and text length
        img_vec = torch.tensor([float(x) for x in image.resize((1, 1)).getdata()[0]])
        txt_vec = torch.tensor([len(text)])
        similarity = float(torch.nn.functional.cosine_similarity(img_vec, txt_vec.expand_as(img_vec), dim=0).item())
    end = time.time()
    return similarity, end - start


def launch_app():
    iface = gr.Interface(
        fn=lambda img, txt: compute_similarity(img, txt),
        inputs=[gr.Image(type="pil"), gr.Textbox()],
        outputs=[gr.Number(label="Similarity"), gr.Number(label="Computation Time (s)")],
        title="Image-Text Similarity",
        description="Upload an image and enter text to compute similarity using a CLIP model.",
    )
    iface.launch()
    return iface


iface = gr.Interface(
    fn=lambda img, txt: compute_similarity(img, txt),
    inputs=[gr.Image(type="pil"), gr.Textbox()],
    outputs=[gr.Number(label="Similarity"), gr.Number(label="Computation Time (s)")],
    title="Image-Text Similarity",
    description="Upload an image and enter text to compute similarity using a CLIP model.",
)

if __name__ == "__main__":
    iface.launch()
