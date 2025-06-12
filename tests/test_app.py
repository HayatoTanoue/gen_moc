import pytest
from PIL import Image
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import app


def test_compute_similarity():
    img = Image.new('RGB', (32, 32), color='red')
    score, t = app.compute_similarity(img, 'a red square')
    assert isinstance(score, float)
    assert isinstance(t, float)
    assert t >= 0


def test_interface_exists():
    assert app.iface is not None
