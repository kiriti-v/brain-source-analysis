#!/usr/bin/env python3
"""
Setup script for Brain Source Localization Analysis Platform
"""

from setuptools import setup, find_packages
import os

# Read the README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="brain-source-analysis",
    version="2.0.0",
    author="Research Team",
    author_email="research@institution.edu",
    description="Web-based platform for analyzing EEG/MEG source localization data with advanced statistical analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/brain-source-analysis",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Visualization",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "pre-commit>=2.15.0",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "brain-analyzer=modern_brain_explorer:main",
            "generate-brain-images=modern_brain_explorer:generate_images_cli",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.md", "*.txt", "*.yml", "*.yaml"],
    },
    project_urls={
        "Bug Reports": "https://github.com/your-org/brain-source-analysis/issues",
        "Source": "https://github.com/your-org/brain-source-analysis",
        "Documentation": "https://brain-source-analysis.readthedocs.io/",
    },
) 