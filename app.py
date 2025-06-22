import gradio as gr
import torch
import clip
from PIL import Image
import numpy as np
import time
from typing import Tuple, Optional

class MoCApp:
    def __init__(self):
        """Initialize the MoC (Multimodal Comparison) application."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.preprocess = None
        self.load_model()
    
    def load_model(self):
        """Load CLIP model for image-text similarity."""
        try:
            self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
            print(f"Model loaded successfully on {self.device}")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def calculate_similarity(self, image: Image.Image, text: str) -> Tuple[float, float]:
        """
        Calculate similarity between image and text.
        
        Args:
            image: PIL Image object
            text: Text string for comparison
            
        Returns:
            Tuple of (similarity_score, calculation_time)
        """
        if not image or not text.strip():
            return 0.0, 0.0
        
        start_time = time.time()
        
        try:
            image_input = self.preprocess(image).unsqueeze(0).to(self.device)
            
            text_input = clip.tokenize([text]).to(self.device)
            
            with torch.no_grad():
                image_features = self.model.encode_image(image_input)
                text_features = self.model.encode_text(text_input)
                
                image_features /= image_features.norm(dim=-1, keepdim=True)
                text_features /= text_features.norm(dim=-1, keepdim=True)
                
                similarity = torch.matmul(image_features, text_features.T).item()
            
            calculation_time = time.time() - start_time
            
            return similarity, calculation_time
            
        except Exception as e:
            print(f"Error calculating similarity: {e}")
            return 0.0, 0.0
    
    def process_similarity(self, image: Optional[Image.Image], text: str) -> Tuple[str, str]:
        """
        Process similarity calculation and return formatted results.
        
        Args:
            image: Uploaded image
            text: Input text
            
        Returns:
            Tuple of (similarity_result, performance_info)
        """
        if image is None:
            return "画像をアップロードしてください。", ""
        
        if not text.strip():
            return "テキストを入力してください。", ""
        
        similarity, calc_time = self.calculate_similarity(image, text)
        
        similarity_percentage = similarity * 100
        
        similarity_result = f"類似度: {similarity_percentage:.2f}%"
        performance_info = f"計算時間: {calc_time:.4f}秒"
        
        return similarity_result, performance_info

def create_interface():
    """Create and configure the Gradio interface."""
    app = MoCApp()
    
    with gr.Blocks(title="MoC - Multimodal Comparison App") as interface:
        gr.Markdown("# MoC - 画像とテキストの類似度検索アプリ")
        gr.Markdown("画像をアップロードし、テキストを入力して類似度を計算します。")
        
        with gr.Row():
            with gr.Column():
                image_input = gr.Image(
                    type="pil",
                    label="画像をアップロード"
                )
                text_input = gr.Textbox(
                    label="テキストを入力",
                    placeholder="画像の内容を説明するテキストを入力してください..."
                )
                calculate_btn = gr.Button("類似度を計算", variant="primary")
            
            with gr.Column():
                similarity_output = gr.Textbox(
                    label="類似度結果",
                    interactive=False
                )
                performance_output = gr.Textbox(
                    label="性能情報",
                    interactive=False
                )
        
        calculate_btn.click(
            fn=app.process_similarity,
            inputs=[image_input, text_input],
            outputs=[similarity_output, performance_output]
        )
        
        image_input.change(
            fn=app.process_similarity,
            inputs=[image_input, text_input],
            outputs=[similarity_output, performance_output]
        )
        
        text_input.change(
            fn=app.process_similarity,
            inputs=[image_input, text_input],
            outputs=[similarity_output, performance_output]
        )
    
    return interface

if __name__ == "__main__":
    interface = create_interface()
    interface.launch(share=False, server_name="0.0.0.0", server_port=7860)
