import gradio as gr
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel

# Load the CLIP model and processor
model_name = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_name)
processor = CLIPProcessor.from_pretrained(model_name)

def compute_similarity(image: Image.Image, text: str) -> float:
    """Compute cosine similarity between an image and text using CLIP."""
    if image is None or text is None or text.strip() == "":
        return 0.0
    inputs = processor(text=[text], images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    # CLIP outputs the similarity logits scaled by a temperature parameter
    similarity = outputs.logits_per_image.squeeze().item()
    return float(similarity)

iface = gr.Interface(
    fn=compute_similarity,
    inputs=[gr.Image(type="pil"), gr.Textbox(label="Input Text")],
    outputs=gr.Number(label="Similarity"),
    title="Image-Text Similarity Demo",
    description="Upload an image and enter text to compute similarity using CLIP",
)

if __name__ == "__main__":
    iface.launch()

