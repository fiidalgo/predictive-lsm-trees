from setuptools import setup, find_packages

setup(
    name="predictive-lsm-trees",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.19.0",
        "scikit-learn>=0.24.0",
        "torch>=1.8.0",
        "matplotlib>=3.3.0",
        "pytest>=6.0.0",
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="A machine learning enhanced LSM tree implementation",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/predictive-lsm-trees",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
) 