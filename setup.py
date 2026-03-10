#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import codecs
import os
from setuptools import setup, find_packages

# 读取README文件作为长描述
here = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(here, "README.md"), encoding="utf-8") as fh:
    long_description = "\n" + fh.read()

# 版本号
VERSION = '1.0.0'
DESCRIPTION = '基于大模型和OCR的通用PDF文档智能拆分工具'
LONG_DESCRIPTION = 'PDF智能拆分工具，支持目录检测和LLM智能拆分，适用于将合并的PDF文档拆分为多个独立文档'

setup(
    name="pdf-split-tool",
    version=VERSION,
    author="PDF Split Tool Team",
    author_email="",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=long_description,
    packages=find_packages(),
    install_requires=[
        "numpy>=1.26.4",
        "Pillow>=11.3.0",
        "psutil>=5.9.8",
        "tqdm>=4.67.1",
        "pymupdf>=1.23.0",
        "paddlepaddle>=2.5.0",
        "paddleocr>=2.7.0",
        "openai>=1.0.0",
        "python-dotenv>=1.0.0",
    ],
    python_requires=">=3.8",
    entry_points={
        'console_scripts': [
            'pdf-split=pdf_split_tool.cli:main',
        ],
    },
    keywords=['python', 'pdf', 'split', 'ocr', 'llm', 'document', '拆分', 'PDF拆分'],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: End Users/Desktop",
        "Topic :: Office/Business",
        "Topic :: Utilities",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
        "License :: OSI Approved :: MIT License",
    ],
    zip_safe=False,
)


