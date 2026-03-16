from app.tracking.format_analyzer import analyze_markdown_format


def test_analyze_markdown_format_counts_headings_correctly() -> None:
    """
    What this test checks:
    - Heading counts by level are computed correctly.

    Why this matters:
    - Heading structure is one of the most important formatting signals
      for later chunking and hierarchy tracking.
    """
    text = """# Title

## Section A
Some text.

### Subsection A1
More text.

#### Detail
Even more text.
"""

    stats = analyze_markdown_format(text)

    assert stats["heading_count_h1"] == 1
    assert stats["heading_count_h2"] == 1
    assert stats["heading_count_h3"] == 1
    assert stats["heading_count_h4_plus"] == 1
    assert stats["heading_depth_max"] == 4


def test_analyze_markdown_format_counts_lists_images_and_tables() -> None:
    """
    What this test checks:
    - Bullet lists, numbered lists, image refs, and table stubs are detected.

    Why this matters:
    - These structures often affect chunking and retrieval behavior.
    """
    text = """# Title

- item one
- item two
1. first
2. second

![chart](chart.png)

[TABLE: Lab ranges]
| Test | Value |
|------|-------|
"""

    stats = analyze_markdown_format(text)

    assert stats["bullet_count"] == 2
    assert stats["numbered_list_count"] == 2
    assert stats["image_ref_count"] == 1
    assert stats["table_stub_count"] == 1
    assert stats["table_like_line_count"] >= 2


def test_analyze_markdown_format_detects_code_blocks() -> None:
    """
    What this test checks:
    - Fenced code blocks are counted correctly.

    Why this matters:
    - We do not want code-like regions to be mistaken for plain narrative text.
    """
    text = """# Example

```python
print("hello")
```

"""

    stats = analyze_markdown_format(text)

    assert stats["code_block_count"] == 1

def test_analyze_markdown_format_detects_broken_heading_order() -> None:
    """
    What this test checks:
    - Suspicious heading jumps such as H1 -> H3 are flagged.

    Why this matters:
    - Broken hierarchy often means the markdown needs cleanup
      before chunking becomes reliable.
    """
    text = """# Title
### Jumped Subsection

Some text here.
"""

    stats = analyze_markdown_format(text)

    assert stats["has_broken_heading_order"] is True
    assert stats["has_clear_hierarchy"] is False

def test_analyze_markdown_format_detects_large_flat_blocks() -> None:
    """
    What this test checks:
    - Large unstructured documents are flagged as flat blocks.

    Why this matters:
    - Flat documents are usually harder to chunk and retrieve from effectively.
    """
    large_paragraph = "word " * 350
    text = large_paragraph.strip()

    stats = analyze_markdown_format(text)

    assert stats["has_large_flat_blocks"] is True
    assert stats["section_count"] == 1

def test_analyze_markdown_format_detects_dense_list_regions() -> None:
    """
    What this test checks:
    - Many consecutive list items are recognized as a dense list region.

    Why this matters:
    - Dense lists may need special chunk handling later.
    """
    text = """# Checklist

- one
- two
- three
- four
- five
- six
"""

    stats = analyze_markdown_format(text)

    assert stats["has_dense_list_regions"] is True

def test_analyze_markdown_format_handles_empty_text() -> None:
    """
    What this test checks:
    - Empty markdown input is handled safely without crashing.

    Why this matters:
    - Source tracking should be robust even when files are blank or partially cleaned.
    """
    stats = analyze_markdown_format("")

    assert stats["char_count"] == 0
    assert stats["word_count"] == 0
    assert stats["line_count"] == 0
    assert stats["section_count"] == 0
    assert stats["blank_line_ratio"] == 0.0