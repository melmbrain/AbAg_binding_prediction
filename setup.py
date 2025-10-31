"""
AbAg Binding Affinity Prediction
Setup script for pip installation
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
long_description = ""
if readme_file.exists():
    with open(readme_file, encoding='utf-8') as f:
        long_description = f.read()

setup(
    name="abag-affinity",
    version="1.0.0",
    author="Antibody Research Team",
    description="Production-ready antibody-antigen binding affinity prediction using deep learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/AbAg_binding_prediction",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.12.0",
        "transformers>=4.20.0",
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "tqdm>=4.62.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
        ],
    },
    include_package_data=True,
    package_data={
        "abag_affinity": ["../models/*.pth", "../models/*.json"],
    },
    entry_points={
        "console_scripts": [
            "abag-predict=abag_affinity.cli:main",
        ],
    },
    keywords="antibody antigen binding affinity deep-learning bioinformatics",
    project_urls={
        "Documentation": "https://github.com/yourusername/AbAg_binding_prediction",
        "Source": "https://github.com/yourusername/AbAg_binding_prediction",
        "Bug Reports": "https://github.com/yourusername/AbAg_binding_prediction/issues",
    },
)
