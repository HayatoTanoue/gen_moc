from setuptools import setup, find_packages

setup(
    name="gen_moc",
    version="1.0.0",
    description="MoC - Multimodal Comparison App for image-text similarity",
    author="Hayato Tanoue",
    packages=find_packages(),
    install_requires=[
        "torch",
        "torchvision", 
        "transformers",
        "gradio",
        "pillow",
        "numpy",
        "pytest",
        "pytest-cov",
        "clip-by-openai",
    ],
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "moc-app=app:main",
        ],
    },
)
