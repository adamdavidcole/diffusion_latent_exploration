"""
Batch image grid visualization utilities.
"""

import logging
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from typing import List, Dict, Any, Optional, Tuple
import textwrap

from ..utils.batch_utils import extract_batch_metadata, calculate_seed_for_video


def create_batch_image_grid(
    batch_path: str,
    output_path: Optional[str] = None,
    max_width: int = 1920,
    max_height: int = 1080,
    thumbnail_size: Optional[Tuple[int, int]] = None,
    margin: int = 10,
    text_margin: int = 8,
    outer_padding: int = 20,
    header_height: int = 120,
    row_label_width: int = 200,
    col_label_height: int = 30
) -> str:
    """Create a comprehensive image grid visualization for a batch.
    
    Args:
        batch_path: Path to the batch directory
        output_path: Path for output image (if None, will auto-generate)
        max_width: Maximum width of the output image
        max_height: Maximum height of the output image
        thumbnail_size: Fixed size for thumbnails (if None, will calculate)
        margin: Margin between grid cells
        text_margin: Margin around text elements
        outer_padding: Padding around the entire content area
        header_height: Height reserved for header text
        row_label_width: Width reserved for row labels
        col_label_height: Height reserved for column labels
        
    Returns:
        Path to the generated image file
    """
    logger = logging.getLogger(__name__)
    logger.info(f"🎨 Creating batch image grid for: {batch_path}")
    
    # Extract batch metadata
    metadata = extract_batch_metadata(batch_path)
    batch_name = metadata['batch_name']
    prompt_template = metadata['prompt_template']
    prompt_groups = metadata['prompt_groups']
    prompt_metadata = metadata['prompt_metadata']
    model_metadata = metadata['model_metadata']
    video_metadata = metadata['video_metadata']
    videos_per_variation = metadata['videos_per_variation']
    
    if not prompt_groups:
        raise ValueError(f"No prompt groups found in {batch_path}")
    
    # Determine grid dimensions
    n_rows = len(prompt_groups)
    n_cols = videos_per_variation
    
    logger.info(f"📐 Grid dimensions: {n_rows} rows × {n_cols} columns")
    
    # Calculate thumbnail size if not provided
    if thumbnail_size is None:
        available_width = max_width - row_label_width - (n_cols + 1) * margin - 2 * outer_padding
        available_height = max_height - header_height - col_label_height - (n_rows + 1) * margin - 2 * outer_padding
        
        thumb_width = available_width // n_cols
        thumb_height = available_height // n_rows
        
        # Maintain aspect ratio based on video metadata
        video_width = video_metadata.get('width', 848)
        video_height = video_metadata.get('height', 480)
        aspect_ratio = video_width / video_height
        
        if thumb_width / thumb_height > aspect_ratio:
            thumb_width = int(thumb_height * aspect_ratio)
        else:
            thumb_height = int(thumb_width / aspect_ratio)
        
        thumbnail_size = (thumb_width, thumb_height)
    
    thumb_width, thumb_height = thumbnail_size
    logger.info(f"🖼️ Thumbnail size: {thumb_width} × {thumb_height}")
    
    # Calculate final image dimensions with outer padding
    grid_width = n_cols * thumb_width + (n_cols + 1) * margin
    grid_height = n_rows * thumb_height + (n_rows + 1) * margin
    
    content_width = row_label_width + grid_width
    content_height = header_height + col_label_height + grid_height
    
    final_width = content_width + 2 * outer_padding
    final_height = content_height + 2 * outer_padding
    
    logger.info(f"📏 Final image size: {final_width} × {final_height}")
    
    # Create the output image with dark background
    background_color = '#272727'
    img = Image.new('RGB', (final_width, final_height), color=background_color)
    draw = ImageDraw.Draw(img)
    
    # Try to load fonts
    try:
        title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        header_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
        label_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10)
        small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 8)
    except OSError:
        # Fallback to default font
        title_font = ImageFont.load_default()
        header_font = ImageFont.load_default()
        label_font = ImageFont.load_default()
        small_font = ImageFont.load_default()
    
    # Draw header (offset by outer padding)
    _draw_header(draw, content_width, header_height, batch_name, prompt_template, 
                model_metadata, video_metadata, title_font, header_font, text_margin, outer_padding)
    
    # Draw column labels (seeds) - offset by outer padding
    base_seed = model_metadata.get('seed', 999)
    col_start_x = outer_padding + row_label_width + margin
    col_label_y = outer_padding + header_height + text_margin
    
    for col in range(n_cols):
        video_number = col + 1
        seed = calculate_seed_for_video(base_seed, video_number)
        col_x = col_start_x + col * (thumb_width + margin)
        col_center_x = col_x + thumb_width // 2
        
        seed_text = f"Seed: {seed}"
        bbox = draw.textbbox((0, 0), seed_text, font=label_font)
        text_width = bbox[2] - bbox[0]
        
        draw.text((col_center_x - text_width // 2, col_label_y), seed_text, 
                 fill='white', font=label_font)
    
    # Draw grid with images - offset by outer padding
    grid_start_y = outer_padding + header_height + col_label_height
    
    for row, prompt_group in enumerate(prompt_groups):
        # Draw row label (prompt variation)
        row_y = grid_start_y + margin + row * (thumb_height + margin)
        
        prompt_var_text = prompt_metadata.get(prompt_group, {}).get('prompt_var_text', prompt_group)
        _draw_row_label(draw, outer_padding, row_y, row_label_width, thumb_height, 
                       prompt_var_text, label_font, text_margin)
        
        # Draw thumbnails for this row
        for col in range(n_cols):
            video_number = col + 1
            video_filename = f"video_{video_number:03d}.jpg"
            
            thumb_x = outer_padding + row_label_width + margin + col * (thumb_width + margin)
            thumb_y = row_y
            
            # Try to load the thumbnail
            video_path = Path(batch_path) / "videos" / prompt_group / video_filename
            
            if video_path.exists():
                try:
                    thumbnail = Image.open(video_path)
                    thumbnail = thumbnail.resize(thumbnail_size, Image.Resampling.LANCZOS)
                    img.paste(thumbnail, (thumb_x, thumb_y))
                except Exception as e:
                    logger.warning(f"⚠️ Error loading {video_path}: {e}")
                    _draw_placeholder(draw, thumb_x, thumb_y, thumb_width, thumb_height, 
                                    f"Error\n{prompt_group}\nvideo_{video_number:03d}", small_font)
            else:
                logger.warning(f"⚠️ Missing thumbnail: {video_path}")
                _draw_placeholder(draw, thumb_x, thumb_y, thumb_width, thumb_height, 
                                f"Missing\n{prompt_group}\nvideo_{video_number:03d}", small_font)
    
    # Save the image
    if output_path is None:
        output_path = Path(batch_path) / f"{batch_name}_grid.png"
    else:
        output_path = Path(output_path)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path, 'PNG', optimize=True)
    
    logger.info(f"✅ Batch grid saved to: {output_path}")
    return str(output_path)


def _draw_header(draw, width, height, batch_name, prompt_template, model_metadata, 
                video_metadata, title_font, header_font, text_margin, outer_padding):
    """Draw the header section with batch name, prompt template, and metadata."""
    current_y = outer_padding + text_margin
    
    # Draw batch name
    draw.text((outer_padding + text_margin, current_y), batch_name, fill='white', font=title_font)
    bbox = draw.textbbox((0, 0), batch_name, font=title_font)
    current_y += bbox[3] - bbox[1] + text_margin
    
    # Draw prompt template (wrapped if necessary)
    wrapped_template = textwrap.fill(prompt_template, width=120)
    for line in wrapped_template.split('\n'):
        draw.text((outer_padding + text_margin, current_y), line, fill='lightgray', font=header_font)
        bbox = draw.textbbox((0, 0), line, font=header_font)
        current_y += bbox[3] - bbox[1] + 2
    
    current_y += text_margin
    
    # Draw model metadata
    model_text = (f"Model: {model_metadata.get('model_id', 'Unknown')}  |  "
                 f"Steps: {model_metadata.get('steps', 'Unknown')}  |  "
                 f"CFG: {model_metadata.get('cfg_scale', 'Unknown')}  |  "
                 f"Sampler: {model_metadata.get('sampler', 'Unknown')}")
    draw.text((outer_padding + text_margin, current_y), model_text, fill='white', font=header_font)
    bbox = draw.textbbox((0, 0), model_text, font=header_font)
    current_y += bbox[3] - bbox[1] + 2
    
    # Draw video metadata
    video_text = (f"Resolution: {video_metadata.get('width', 'Unknown')} × "
                 f"{video_metadata.get('height', 'Unknown')}  |  "
                 f"Frames: {video_metadata.get('frames', 'Unknown')}")
    draw.text((outer_padding + text_margin, current_y), video_text, fill='white', font=header_font)


def _draw_row_label(draw, x, y, width, height, text, font, margin):
    """Draw a row label with wrapped text."""
    # Wrap text to fit in the available width
    wrapped_lines = []
    words = text.split()
    current_line = ""
    
    for word in words:
        test_line = current_line + (" " if current_line else "") + word
        bbox = draw.textbbox((0, 0), test_line, font=font)
        text_width = bbox[2] - bbox[0]
        
        if text_width <= width - 2 * margin:
            current_line = test_line
        else:
            if current_line:
                wrapped_lines.append(current_line)
                current_line = word
            else:
                wrapped_lines.append(word)  # Word is too long, but add it anyway
    
    if current_line:
        wrapped_lines.append(current_line)
    
    # Calculate vertical centering
    line_height = draw.textbbox((0, 0), "Ag", font=font)[3] - draw.textbbox((0, 0), "Ag", font=font)[1]
    total_text_height = len(wrapped_lines) * line_height
    start_y = y + (height - total_text_height) // 2
    
    # Draw each line
    for i, line in enumerate(wrapped_lines):
        line_y = start_y + i * line_height
        draw.text((x + margin, line_y), line, fill='white', font=font)


def _draw_placeholder(draw, x, y, width, height, text, font):
    """Draw a placeholder rectangle with text."""
    # Draw dark gray background with lighter border
    draw.rectangle([x, y, x + width, y + height], fill='#404040', outline='#606060')
    
    # Draw text in center
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    text_x = x + (width - text_width) // 2
    text_y = y + (height - text_height) // 2
    
    draw.text((text_x, text_y), text, fill='lightgray', font=font)
