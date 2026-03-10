# 配置说明

## 环境变量配置

### 创建 .env 文件

在项目根目录创建 `.env` 文件：

```bash
# LLM API配置
LLM_API_KEY=your_api_key_here
LLM_API_BASE_URL=https://api.example.com/v1

# 或使用DeepSeek
DEEPSEEK_API_KEY=your_deepseek_key_here
```

### 环境变量说明

- `LLM_API_KEY`: LLM API密钥（优先使用）
- `LLM_API_BASE_URL`: LLM API基础URL
- `DEEPSEEK_API_KEY`: DeepSeek API密钥（备选）

### 代码中的默认配置

开源版本中**不会在代码里硬编码任何API密钥**，仅保留如下默认行为：

- 如果未设置 `LLM_API_BASE_URL`，默认使用 `https://one-api.maas.com.cn/v1`
- 如果未设置 `LLM_API_KEY` / `DEEPSEEK_API_KEY`，在需要调用大模型时会抛出错误并提示配置方式

**注意**: 始终使用环境变量管理敏感信息，不要将API密钥硬编码在代码中或提交到版本库。

## 关键词配置

### 修改补充材料关键词

在 `pdf-split.py` 中修改 `SUPPLEMENT_KEYWORDS` 类常量：

```python
SUPPLEMENT_KEYWORDS = {
    'appendix': ['appendix', '附录'],
    'references': ['references', 'bibliography', '参考文献'],
    'supplementary': ['supplementary', '补充材料', 'supplement']
}
```

### 修改页面类型关键词

在 `pdf-split.py` 中修改 `PAGE_TYPE_KEYWORDS` 类常量：

```python
PAGE_TYPE_KEYWORDS = {
    'toc': ['目录', 'contents', 'table of contents', '目 录'],
    'abstract': ['abstract', '摘要'],
    'references': ['references', 'bibliography', '参考文献'],
    'title_page': ['abstract', '摘要', 'introduction', '引言', ...]
}
```

## 性能参数配置

### 图像缩放比例

```bash
--image-scale 0.6  # 默认值，大文件可降低到0.5
```

### OCR批处理大小

在代码中修改 `batch_size` 参数（默认2，大文件建议1）。

### 内存优化建议

- 大文件（>200页）: `--image-scale 0.5`
- 中等文件（50-200页）: `--image-scale 0.6`
- 小文件（<50页）: 默认设置

## 文档类型配置

### 支持的文档类型

- `general`: 通用文档（默认）
- `legal`: 法律文书
- `academic`: 学术论文

### 添加新的文档类型

1. 在 `_build_system_prompt()` 中添加新类型的规则
2. 在 `_post_process_splits()` 中添加后处理逻辑
3. 在命令行参数中添加新选项


