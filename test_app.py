import pytest
import torch
from PIL import Image
import numpy as np
from app import MoCApp
import tempfile
import os

class TestMoCApp:
    """Test suite for MoC application."""
    
    @pytest.fixture
    def app(self):
        """Create MoCApp instance for testing."""
        return MoCApp()
    
    @pytest.fixture
    def sample_image(self):
        """Create a sample image for testing."""
        image_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        return Image.fromarray(image_array)
    
    def test_model_loading(self, app):
        """Test that the CLIP model loads successfully."""
        assert app.model is not None
        assert app.preprocess is not None
        assert app.device in ["cuda", "cpu"]
    
    def test_calculate_similarity_valid_inputs(self, app, sample_image):
        """Test similarity calculation with valid inputs."""
        text = "a colorful image"
        similarity, calc_time = app.calculate_similarity(sample_image, text)
        
        assert isinstance(similarity, float)
        assert -1.0 <= similarity <= 1.0
        
        assert isinstance(calc_time, float)
        assert calc_time > 0
    
    def test_calculate_similarity_empty_text(self, app, sample_image):
        """Test similarity calculation with empty text."""
        similarity, calc_time = app.calculate_similarity(sample_image, "")
        
        assert similarity == 0.0
        assert calc_time == 0.0
    
    def test_calculate_similarity_no_image(self, app):
        """Test similarity calculation with no image."""
        similarity, calc_time = app.calculate_similarity(None, "test text")
        
        assert similarity == 0.0
        assert calc_time == 0.0
    
    def test_process_similarity_valid_inputs(self, app, sample_image):
        """Test the process_similarity method with valid inputs."""
        text = "a test image"
        similarity_result, performance_info = app.process_similarity(sample_image, text)
        
        assert isinstance(similarity_result, str)
        assert isinstance(performance_info, str)
        
        assert "類似度:" in similarity_result
        assert "%" in similarity_result
        
        assert "計算時間:" in performance_info
        assert "秒" in performance_info
    
    def test_process_similarity_no_image(self, app):
        """Test process_similarity with no image."""
        similarity_result, performance_info = app.process_similarity(None, "test")
        
        assert "画像をアップロードしてください。" in similarity_result
        assert performance_info == ""
    
    def test_process_similarity_no_text(self, app, sample_image):
        """Test process_similarity with no text."""
        similarity_result, performance_info = app.process_similarity(sample_image, "")
        
        assert "テキストを入力してください。" in similarity_result
        assert performance_info == ""
    
    def test_similarity_consistency(self, app, sample_image):
        """Test that similarity calculation is consistent."""
        text = "consistent test"
        
        similarity1, _ = app.calculate_similarity(sample_image, text)
        similarity2, _ = app.calculate_similarity(sample_image, text)
        
        assert abs(similarity1 - similarity2) < 1e-6
    
    def test_performance_measurement(self, app, sample_image):
        """Test that performance measurement works correctly."""
        text = "performance test"
        similarity, calc_time = app.calculate_similarity(sample_image, text)
        
        assert calc_time < 10.0
        assert calc_time > 0.0
    
    def test_different_image_sizes(self, app):
        """Test that the app handles different image sizes."""
        sizes = [(100, 100), (300, 200), (500, 500)]
        text = "size test"
        
        for width, height in sizes:
            image_array = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
            image = Image.fromarray(image_array)
            
            similarity, calc_time = app.calculate_similarity(image, text)
            
            assert isinstance(similarity, float)
            assert -1.0 <= similarity <= 1.0
            assert calc_time > 0

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
