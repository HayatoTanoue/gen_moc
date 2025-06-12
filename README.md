# gen_moc

This repository contains a minimal demo application that computes image–text similarity using OpenAI's CLIP model. A simple UI built with [Gradio](https://gradio.app/) allows you to upload an image and enter a text prompt to see how similar they are.

## Requirements

- Python 3.8+
- `torch`
- `transformers`
- `gradio`
- `Pillow`

You can install the dependencies with:

```bash
pip install torch torchvision transformers gradio pillow
```

The CPU versions of the packages are sufficient to run the demo.

## Running the app

Launch the application with:

```bash
python app.py
```

A local web server will start and display an interface where you can upload an image, enter text, and view the similarity score produced by CLIP.

