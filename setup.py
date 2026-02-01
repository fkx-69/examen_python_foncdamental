"""
Configuration setup.py to make the package installable.
This file allows installing the package with pip install.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Reads the README for the long description
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

setup(
    name="ds-toolkit-examen-project", # Unique name for TestPyPI
    version="1.0.0",
    author="Master Data Science Student",
    author_email="student@example.com",
    description="Object-Oriented Data Science Toolkit - Final Project",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/oop-data-science",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Education",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
    ],
    python_requires=">=3.8",
    install_requires=[
        "pandas>=1.3.0",
        "numpy>=1.21.0",
        "scikit-learn>=1.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "twine",
            "build",
        ],
    },
    include_package_data=True,
    keywords="data-science machine-learning oop python education",
)
