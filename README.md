# Gen MoC

This repository contains a minimal image-text similarity application using a CLIP model with a Gradio interface.

## Usage

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
python app.py
```

## Testing

Tests use pytest:

```bash
pytest
```

If CLIP weights are not available locally, the application falls back to a simple similarity implementation. This may occur in offline environments.
