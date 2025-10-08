"""
Local assets module to replace babeldoc.assets.assets functionality
"""
import json
from pathlib import Path
from typing import Tuple, Dict, Any


def get_font_and_metadata(font_name: str) -> Tuple[Path, Dict[str, Any]]:
    """
    Get font path and metadata for a given font name using local assets.

    Args:
        font_name: Name of the font file (e.g., "GoNotoKurrent-Regular.ttf")

    Returns:
        Tuple of (font_path, metadata_dict)
    """
    # Path to local assets
    assets_dir = Path(__file__).parent / "BabelDOC-Assets-main"
    fonts_dir = assets_dir / "fonts"
    metadata_file = assets_dir / "font_metadata.json"

    # Check if assets directory exists
    if not assets_dir.exists():
        raise FileNotFoundError(f"Assets directory not found: {assets_dir}")

    # Load font metadata
    if not metadata_file.exists():
        raise FileNotFoundError(f"Font metadata file not found: {metadata_file}")

    with open(metadata_file, 'r', encoding='utf-8') as f:
        font_metadata = json.load(f)

    # Check if font exists in metadata
    if font_name not in font_metadata:
        raise FileNotFoundError(f"Font '{font_name}' not found in metadata")

    metadata = font_metadata[font_name]

    # Check if font file exists
    font_path = fonts_dir / font_name
    if not font_path.exists():
        raise FileNotFoundError(f"Font file not found: {font_path}")

    return font_path, metadata
