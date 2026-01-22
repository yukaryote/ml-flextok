"""
Utilities for creating videos from search history.
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from typing import List, Tuple, Optional
import os


def add_text_to_image(
    image: Image.Image,
    iteration: int,
    token: int,
    score: float,
    font_size: int = 40,
    bg_color: tuple = (0, 0, 0),
    text_color: tuple = (255, 255, 255),
) -> Image.Image:
    """
    Add iteration number, token, and score text above an image.

    Args:
        image: PIL Image
        iteration: Iteration number
        token: Token value
        score: DreamSim score
        font_size: Font size for text
        bg_color: Background color for text area
        text_color: Text color

    Returns:
        New image with text header
    """
    # Create new image with extra space on top for text
    img_width, img_height = image.size
    header_height = font_size + 40  # Extra padding

    new_img = Image.new('RGB', (img_width, img_height + header_height), bg_color)

    # Paste original image below header
    new_img.paste(image, (0, header_height))

    # Draw text on header
    draw = ImageDraw.Draw(new_img)

    # Try to load a decent font, fall back to default if not available
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except:
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size)
        except:
            font = ImageFont.load_default()

    # Create text
    text = f"Iter {iteration} | Token {token} | Score {score:.4f}"

    # Get text bounding box for centering
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    # Center text horizontally
    text_x = (img_width - text_width) // 2
    text_y = (header_height - text_height) // 2

    # Draw text
    draw.text((text_x, text_y), text, fill=text_color, font=font)

    return new_img


def create_video_from_history(
    chosen_history: List[Tuple],
    output_path: str,
    fps: int = 2,
    font_size: int = 40,
    bg_color: tuple = (0, 0, 0),
    text_color: tuple = (255, 255, 255),
    add_final_pause: bool = True,
    final_pause_duration: int = 3,
) -> str:
    """
    Create a video from search history.

    Args:
        chosen_history: List of (token, image, score) tuples
        output_path: Path to save video (e.g., 'output.mp4')
        fps: Frames per second (lower = slower transitions)
        font_size: Font size for text overlay
        bg_color: Background color for text header
        text_color: Text color
        add_final_pause: Whether to hold on final frame
        final_pause_duration: Duration to hold final frame (seconds)

    Returns:
        Path to saved video
    """
    try:
        import cv2
    except ImportError:
        raise ImportError("OpenCV (cv2) is required for video creation. Install with: pip install opencv-python")

    if not chosen_history:
        raise ValueError("chosen_history is empty")

    # Process all frames
    frames = []
    for i, (token, image_pil, score) in enumerate(chosen_history):
        # Extract token value
        if isinstance(token, torch.Tensor):
            token_val = token.item()
        else:
            token_val = token

        # Add text overlay
        frame_with_text = add_text_to_image(
            image_pil,
            iteration=i + 1,
            token=token_val,
            score=score,
            font_size=font_size,
            bg_color=bg_color,
            text_color=text_color,
        )

        # Convert PIL to numpy (RGB -> BGR for OpenCV)
        frame_np = np.array(frame_with_text)
        frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
        frames.append(frame_bgr)

    # Add final pause frames
    if add_final_pause and frames:
        final_frame = frames[-1]
        num_pause_frames = int(fps * final_pause_duration)
        for _ in range(num_pause_frames):
            frames.append(final_frame.copy())

    # Get frame dimensions
    height, width = frames[0].shape[:2]

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Write frames
    for frame in frames:
        out.write(frame)

    # Release video writer
    out.release()

    print(f"Video saved to: {output_path}")
    print(f"Total frames: {len(frames)}")
    print(f"Duration: {len(frames) / fps:.1f} seconds")

    return output_path


def create_comparison_video(
    histories: List[Tuple[str, List[Tuple]]],
    output_path: str,
    fps: int = 2,
    font_size: int = 30,
) -> str:
    """
    Create a side-by-side comparison video of multiple search histories.

    Args:
        histories: List of (name, chosen_history) tuples
        output_path: Path to save video
        fps: Frames per second
        font_size: Font size for text

    Returns:
        Path to saved video
    """
    try:
        import cv2
    except ImportError:
        raise ImportError("OpenCV (cv2) is required. Install with: pip install opencv-python")

    if not histories:
        raise ValueError("histories is empty")

    # Find max length
    max_len = max(len(history) for _, history in histories)

    # Process frames for each history
    all_frames_list = []
    for name, chosen_history in histories:
        frames = []
        for i in range(max_len):
            if i < len(chosen_history):
                token, image_pil, score = chosen_history[i]
                token_val = token.item() if isinstance(token, torch.Tensor) else token

                frame_with_text = add_text_to_image(
                    image_pil,
                    iteration=i + 1,
                    token=token_val,
                    score=score,
                    font_size=font_size,
                )
            else:
                # Use last frame if this history is shorter
                frame_with_text = frames[-1] if frames else Image.new('RGB', (256, 256))

            frames.append(np.array(frame_with_text))
        all_frames_list.append(frames)

    # Concatenate frames horizontally
    combined_frames = []
    for i in range(max_len):
        row_frames = [frames[i] for frames in all_frames_list]
        combined = np.concatenate(row_frames, axis=1)
        combined_bgr = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)
        combined_frames.append(combined_bgr)

    # Write video
    height, width = combined_frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame in combined_frames:
        out.write(frame)

    out.release()

    print(f"Comparison video saved to: {output_path}")
    return output_path


def create_gif_from_history(
    chosen_history: List[Tuple],
    output_path: str,
    duration: int = 500,  # milliseconds per frame
    font_size: int = 40,
    loop: int = 0,  # 0 = infinite loop
) -> str:
    """
    Create an animated GIF from search history.

    Args:
        chosen_history: List of (token, image, score) tuples
        output_path: Path to save GIF (e.g., 'output.gif')
        duration: Duration per frame in milliseconds
        font_size: Font size for text overlay
        loop: Number of loops (0 = infinite)

    Returns:
        Path to saved GIF
    """
    if not chosen_history:
        raise ValueError("chosen_history is empty")

    # Process all frames
    frames = []
    for i, (token, image_pil, score) in enumerate(chosen_history):
        token_val = token.item() if isinstance(token, torch.Tensor) else token

        frame_with_text = add_text_to_image(
            image_pil,
            iteration=i + 1,
            token=token_val,
            score=score,
            font_size=font_size,
        )
        frames.append(frame_with_text)

    # Save as GIF
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=loop,
    )

    print(f"GIF saved to: {output_path}")
    print(f"Total frames: {len(frames)}")
    print(f"Duration: {len(frames) * duration / 1000:.1f} seconds")

    return output_path


def save_history_grid(
    chosen_history: List[Tuple],
    output_path: str,
    cols: int = 8,
    font_size: int = 20,
) -> str:
    """
    Save search history as a grid of images.

    Args:
        chosen_history: List of (token, image, score) tuples
        output_path: Path to save image
        cols: Number of columns
        font_size: Font size for text

    Returns:
        Path to saved image
    """
    if not chosen_history:
        raise ValueError("chosen_history is empty")

    # Process all frames
    frames = []
    for i, (token, image_pil, score) in enumerate(chosen_history):
        token_val = token.item() if isinstance(token, torch.Tensor) else token

        frame_with_text = add_text_to_image(
            image_pil,
            iteration=i + 1,
            token=token_val,
            score=score,
            font_size=font_size,
        )
        frames.append(frame_with_text)

    # Calculate grid dimensions
    n_frames = len(frames)
    rows = (n_frames + cols - 1) // cols

    # Get frame size
    frame_width, frame_height = frames[0].size

    # Create grid image
    grid_width = cols * frame_width
    grid_height = rows * frame_height
    grid_img = Image.new('RGB', (grid_width, grid_height), (0, 0, 0))

    # Paste frames into grid
    for i, frame in enumerate(frames):
        row = i // cols
        col = i % cols
        x = col * frame_width
        y = row * frame_height
        grid_img.paste(frame, (x, y))

    # Save grid
    grid_img.save(output_path)

    print(f"Grid saved to: {output_path}")
    print(f"Grid size: {cols}x{rows} ({n_frames} frames)")

    return output_path
