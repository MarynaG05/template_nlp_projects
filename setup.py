from setuptools import find_packages, setup

setup(
    name="tuwnlpie",
    version="0.0.1",
    description="Project template for Natural Language Processing and Information Extraction course, 2022WS",
    author="Maryna Gutruf, Luka Skarica, Lukas Prenner, Anton Peter",
    license="MIT",
    install_requires=[
        "nltk==3.7",
        "numpy==1.23.3",
        "torch==1.12.1",
        "pandas==1.5.0",
        "scikit-learn==1.1.2",
        "regex==2022.10.31",
        "black==22.10.0",
        "pytest==7.2.0",
    ],
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Education",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
    ],
    zip_safe=False
)
