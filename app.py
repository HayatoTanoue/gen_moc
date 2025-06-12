from flask import Flask, request, render_template_string
from PIL import Image
import numpy as np
from io import BytesIO

app = Flask(__name__)

HTML = """
<!doctype html>
<title>Image & Text Similarity</title>
<h1>Image & Text Similarity</h1>
<form method=post enctype=multipart/form-data>
  <label for=image>Upload image:</label><br>
  <input type=file name=image accept="image/*" required><br><br>
  <label for=text>Input text:</label><br>
  <textarea name=text rows=4 cols=50 required>{{ text or '' }}</textarea><br><br>
  <input type=submit value="Compute similarity">
</form>
{% if similarity is not none %}
  <h2>Similarity: {{ '{:.4f}'.format(similarity) }}</h2>
{% endif %}
"""

def image_histogram(fileobj):
    img = Image.open(fileobj).convert('L')
    arr = np.array(img)
    hist, _ = np.histogram(arr, bins=256, range=(0, 256), density=True)
    return hist

def text_histogram(text):
    arr = np.frombuffer(text.encode('utf-8', 'ignore'), dtype=np.uint8)
    hist, _ = np.histogram(arr, bins=256, range=(0, 256), density=True)
    return hist

def compute_similarity(img_hist, text_hist):
    return float(np.dot(img_hist, text_hist))

@app.route('/', methods=['GET', 'POST'])
def index():
    similarity = None
    text = ''
    if request.method == 'POST':
        text = request.form.get('text', '')
        img_file = request.files.get('image')
        if img_file and text:
            img_bytes = img_file.read()
            img_hist = image_histogram(BytesIO(img_bytes))
            text_hist = text_histogram(text)
            similarity = compute_similarity(img_hist, text_hist)
    return render_template_string(HTML, similarity=similarity, text=text)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
