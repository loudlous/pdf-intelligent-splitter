# PDF Intelligent Splitter

An intelligent PDF document splitting tool based on Large Language Models (LLM) and OCR, designed to split merged PDF documents into multiple independent documents.

## 🚀 Features

1. **Automatic OCR Recognition**: Supports automatic GPU/CPU switching using PaddleOCR for text recognition
2. **Intelligent Splitting Strategy**:
   - **Priority 1**: Uses PDF table of contents for precise splitting (zero cost, no model calls)
   - **Priority 2**: Uses LLM for intelligent splitting when TOC is unavailable
3. **Universal Document Support**: Works with all document types (legal documents, academic papers, general documents, etc.)
4. **Automatic File Naming**: Generates standardized filenames based on document titles
5. **Configurable Keywords**: Supplementary materials and page type keywords are configurable for easy extension

## 📋 Requirements

- Python 3.8+
- GPU support (optional, auto-detected)
- Sufficient disk space for OCR results and split PDFs

## 📦 Installation

### Method 1: Install from PyPI (Recommended)

The easiest and quickest way to install:

```bash
pip install pdf-intelligent-splitter
```

After installation, you can use the `pdf-split` command directly:

```bash
pdf-split input.pdf -o ./result --document-type legal
```

### Method 2: Install from Source

```bash
git clone https://github.com/loudlous/pdf-intelligent-splitter.git
cd pdf-intelligent-splitter
pip install -r requirements.txt
```

### Core Dependencies

**OCR Support (Required)**:
```bash
pip install paddlepaddle paddleocr
```

**PDF Processing (Required)**:
```bash
pip install pymupdf
```

**LLM API (Required)**:
```bash
pip install openai
```

**Optional Dependencies**:
```bash
pip install python-dotenv  # Environment variable support
```

## ⚙️ Configuration

### Environment Variables

Create a `.env` file in the project root (optional):

```bash
# LLM API Configuration
LLM_API_KEY=your_api_key_here
LLM_API_BASE_URL=https://api.example.com/v1

# Or use DeepSeek
DEEPSEEK_API_KEY=your_deepseek_key_here
```

### Security Notice

⚠️ **For security and open-source best practices, this project does NOT hardcode any API keys in the source code.**

You **must** configure LLM access parameters through environment variables:
- `LLM_API_KEY` or `DEEPSEEK_API_KEY`: Your API key (required)
- `LLM_API_BASE_URL`: API base URL (optional, defaults to `https://one-api.maas.com.cn/v1`)

If API keys are not set, the tool will raise an error when attempting to use LLM-based splitting.

## 🎯 Usage

### Basic Usage

**If installed from PyPI:**
```bash
pdf-split <input.pdf> -o <output_dir>
```

**If installed from source:**
```bash
python pdf-split.py <input.pdf> -o <output_dir>
```

### Full Parameters

```bash
python pdf-split.py <input.pdf> \
    -o <output_dir> \                    # Output directory
    --document-type <type> \              # Document type: general/legal/academic
    --ocr-json <path> \                  # Use existing OCR JSON file
    --use-gpu \                           # Force GPU usage (if available)
    --use-cpu \                           # Force CPU usage
    --image-scale <scale>                 # Image scale factor (default: 1.0, lower for large files)
```

### Examples

```bash
# Basic splitting (after PyPI installation)
pdf-split document.pdf -o ./result

# Specify document type
pdf-split academic_papers.pdf -o ./result --document-type academic

# Use existing OCR results (skip OCR step)
pdf-split document.pdf -o ./result --ocr-json ./ocr_result.json

# Large file optimization (reduce memory usage)
pdf-split large_document.pdf -o ./result --image-scale 0.5

# Run from source (if cloned from GitHub)
python pdf-split.py document.pdf -o ./result
```

## 📤 Output

After splitting, the output directory contains:

- `split_points.json`: Split point information (JSON format)
  - `total_pages`: Total number of pages
  - `splits`: List of split results
    - `start_page`: Start page number
    - `end_page`: End page number
    - `title`: Document title
- `*_ocr.json`: OCR recognition results (optional, for subsequent processing)
- `01_<title>.pdf`, `02_<title>.pdf`, ...: Split PDF files

## 🔧 How It Works

### 1. OCR Recognition Phase

- Automatically detects GPU availability
- Uses PaddleOCR for text recognition
- Generates simplified OCR JSON (only key information: page number, page height, text, and Y coordinates)

### 2. Splitting Strategy

**Strategy 1: Table of Contents Splitting (Priority)**
- Automatically detects TOC pages in PDF
- Extracts TOC entries and page numbers
- Splits precisely based on TOC (zero cost, no model calls)

**Strategy 2: LLM Intelligent Splitting**
- Extracts key page information (headers, first 3 lines, page type)
- Builds compact prompts and sends to LLM
- LLM analyzes document structure and returns splitting suggestions

### 3. Post-processing

- Corrects overlapping pages
- Merges supplementary materials (appendices, references, etc.)
- Validates complete page coverage
- Normalizes filenames

## 🎨 Configurable Keywords

The tool uses a configurable keyword system for easy extension:

### Supplementary Material Keywords

```python
SUPPLEMENT_KEYWORDS = {
    'appendix': ['appendix', '附录'],
    'references': ['references', 'bibliography', '参考文献'],
    'supplementary': ['supplementary', '补充材料', 'supplement']
}
```

### Page Type Keywords

```python
PAGE_TYPE_KEYWORDS = {
    'toc': ['目录', 'contents', 'table of contents', '目 录'],
    'abstract': ['abstract', '摘要'],
    'references': ['references', 'bibliography', '参考文献'],
    'title_page': ['abstract', '摘要', 'introduction', '引言', ...]
}
```

You can modify these configurations in the code to adapt to different document types.

## ⚡ Performance Optimization

### Memory Optimization

- Automatically reduces image scale for large files (`--image-scale 0.5-0.6`)
- Reduces OCR batch size for memory-constrained environments
- Timely memory release (using `gc.collect()`)

### Token Optimization

- OCR JSON only contains key information (text and Y coordinates)
- Prompts only include headers and first 3 lines of text
- Large files are automatically truncated to ensure all pages are processed

### GPU Acceleration

- Automatically detects GPU availability
- Supports PaddleOCR GPU acceleration
- Automatically falls back to CPU when GPU is unavailable

## 🐛 Troubleshooting

### Common Issues

1. **OCR Initialization Failed**
   - Check if PaddleOCR is correctly installed
   - Check GPU drivers and CUDA version
   - Try using `--use-cpu` to force CPU usage

2. **Out of Memory (Exit code 137)**
   - Reduce `--image-scale` (e.g., 0.5)
   - Use existing OCR JSON to skip OCR step
   - Process files in smaller batches

3. **Inaccurate Splitting Results**
   - Check OCR quality (view OCR JSON)
   - Try different document type parameters
   - Verify LLM API is working correctly

4. **API Call Failed**
   - Check API key and base URL configuration
   - Verify network connectivity
   - Check API service status

## 📝 Notes

1. **Splitting Principle**: The tool follows the principle of "prefer over-splitting over incorrect merging"
2. **Title Extraction**: If a clear title cannot be extracted, header information or default titles will be used
3. **Page Coverage**: The tool validates that all pages are covered without gaps or overlaps
4. **File Naming**: Special characters in filenames are replaced with underscores to ensure filesystem compatibility

## 📦 PyPI Package Information

- **PyPI Package Name**: `pdf-intelligent-splitter`
- **PyPI URL**: https://pypi.org/project/pdf-intelligent-splitter/
- **Install Command**: `pip install pdf-intelligent-splitter`
- **GitHub URL**: https://github.com/loudlous/pdf-intelligent-splitter

## 📄 License

MIT License - See [LICENSE](LICENSE) file for details

## 📚 Documentation

- [Usage Guide](docs/USAGE.md)
- [Configuration Guide](docs/CONFIG.md)
- [Architecture Overview](docs/ARCHITECTURE.md)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 Changelog

### v1.0.0
- Initial release
- Support for TOC-based and LLM-based splitting
- GPU/CPU automatic switching
- Configurable keyword system
- Token optimization and performance improvements

## 🙏 Acknowledgments

- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) for OCR capabilities
- [PyMuPDF](https://github.com/pymupdf/PyMuPDF) for PDF processing
- OpenAI-compatible API providers for LLM support

---

**Made with ❤️ for the open-source community**

