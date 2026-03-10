"""
PDF智能拆分工具

基于大模型和OCR的通用PDF文档智能拆分工具
"""

from .splitter import LLMDocumentSplitter, DocumentSplit, TextBlock

__version__ = '1.0.0'
__all__ = ['LLMDocumentSplitter', 'DocumentSplit', 'TextBlock']


