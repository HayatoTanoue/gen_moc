# gen_moc

This repository contains a minimal Flask application that demonstrates a simple approach to calculating a pseudo similarity between an uploaded image and user-provided text.

## Requirements

The application relies on system packages available on Ubuntu. Install them using `apt`:

```bash
sudo apt-get update
sudo apt-get install -y python3-flask python3-numpy python3-pil
```

## Running the app

Launch the server with the system Python:

```bash
/usr/bin/python3 app.py
```

Then open `http://localhost:5000` in your browser. Upload an image, enter some text and the page will display a numeric similarity value.
