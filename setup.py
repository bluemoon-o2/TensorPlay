from setuptools import setup, find_packages
from TensorPlay import __version__, __description__, __author__, __url__, __email__

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='TensorPlay',
    version=__version__,
    license='MIT License',
    install_requires=['numpy'],
    description=__description__,
    long_description=long_description,
    long_description_content_type="text/markdown",
    author=__author__,
    author_email=__email__,
    url=__url__,
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
    ],
    python_requires='>=3.8',
)