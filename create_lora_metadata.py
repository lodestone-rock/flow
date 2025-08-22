import os
import json
import argparse
from PIL import Image
from tqdm import tqdm

def get_image_dimensions(image_path):
    """Gets the width and height of an image."""
    try:
        with Image.open(image_path) as img:
            return img.width, img.height
    except Exception as e:
        print(f"Warning: Could not open or read dimensions for {image_path}: {e}")
        return None, None

def create_metadata_jsonl(image_folder, output_file, caption_extension=".txt", 
                          default_caption="a photo", recursive=False, 
                          use_filename_as_caption_fallback=False,
                          caption_prefix="", caption_suffix="",
                          is_tag_based_default=True, loss_weight=1.0):
    """
    Generates a JSONL metadata file from a folder of images.

    Args:
        image_folder (str): Path to the folder containing images.
        output_file (str): Path to save the output JSONL file.
        caption_extension (str): Extension for caption files (e.g., ".txt").
        default_caption (str): Caption to use if no caption file is found.
        recursive (bool): Whether to search for images in subdirectories.
        use_filename_as_caption_fallback (bool): If true and no caption file, use filename (without ext) as caption.
        caption_prefix (str): Prefix to add to all captions/tags.
        caption_suffix (str): Suffix to add to all captions/tags.
        is_tag_based_default (bool): Default value for is_tag_based if using default_caption or filename.
                                     If a .txt file is found, it's assumed to be tags (is_tag_based=True).
        loss_weight (float): Default loss_weight for all entries.
    """
    image_extensions = ('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.gif')
    image_paths = []

    print(f"Scanning for images in: {image_folder}")
    if recursive:
        for root, _, files in os.walk(image_folder):
            for file in files:
                if file.lower().endswith(image_extensions):
                    image_paths.append(os.path.join(root, file))
    else:
        for file in os.listdir(image_folder):
            if file.lower().endswith(image_extensions):
                image_paths.append(os.path.join(image_folder, file))
    
    if not image_paths:
        print("No images found. Exiting.")
        return

    print(f"Found {len(image_paths)} images. Processing...")

    with open(output_file, 'w', encoding='utf-8') as f_out:
        for image_path in tqdm(image_paths, desc="Processing images"):
            width, height = get_image_dimensions(image_path)
            if width is None or height is None:
                continue

            # Filename should be relative to the image_folder
            # This is important for the dataloader in train_chroma_lora.py
            relative_image_path = os.path.relpath(image_path, image_folder)
            # Ensure consistent path separators (Unix-style)
            relative_image_path = relative_image_path.replace(os.sep, '/')


            caption_or_tags = None
            is_tag_based = is_tag_based_default # Default assumption

            # Try to find a companion caption/tag file
            base_filename, _ = os.path.splitext(image_path)
            caption_file_path = base_filename + caption_extension
            
            if os.path.exists(caption_file_path):
                try:
                    with open(caption_file_path, 'r', encoding='utf-8') as cf:
                        caption_or_tags = cf.read().strip()
                    is_tag_based = True # Assume .txt files contain tags
                except Exception as e:
                    print(f"Warning: Could not read caption file {caption_file_path}: {e}")
            
            if not caption_or_tags: # If no caption file or it was empty/unreadable
                if use_filename_as_caption_fallback:
                    filename_no_ext, _ = os.path.splitext(os.path.basename(image_path))
                    caption_or_tags = filename_no_ext.replace('_', ' ').replace('-', ' ') # Basic cleanup
                else:
                    caption_or_tags = default_caption
            
            # Add prefix and suffix
            final_caption = caption_or_tags
            if caption_prefix:
                final_caption = caption_prefix.strip() + (" " if not caption_prefix.endswith(" ") and final_caption else "") + final_caption
            if caption_suffix:
                final_caption = final_caption + (" " if not caption_suffix.startswith(" ") and final_caption else "") + caption_suffix.strip()
            
            # Ensure final_caption is not empty
            if not final_caption:
                print(f"Warning: Empty caption for {image_path}, using default: '{default_caption}'")
                final_caption = default_caption


            metadata_entry = {
                "filename": relative_image_path,
                "caption_or_tags": final_caption,
                "width": width,
                "height": height,
                "is_tag_based": is_tag_based,
                "is_url_based": False,
                "loss_weight": loss_weight
            }
            
            f_out.write(json.dumps(metadata_entry) + '\n')

    print(f"Successfully created metadata file: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create JSONL metadata from an image folder for Chroma LoRA training.")
    parser.add_argument("image_folder", type=str, help="Path to the folder containing images.")
    parser.add_argument("output_file", type=str, help="Path to save the output JSONL file.")
    parser.add_argument("--caption_extension", type=str, default=".txt", help="Extension for caption files (e.g., '.txt').")
    parser.add_argument("--default_caption", type=str, default="a photo", help="Caption to use if no caption file is found.")
    parser.add_argument("--recursive", action="store_true", help="Search for images in subdirectories.")
    parser.add_argument("--use_filename_as_caption_fallback", action="store_true", 
                        help="If true and no caption file is found, use the image filename (without extension, spaces for underscores/hyphens) as caption.")
    parser.add_argument("--caption_prefix", type=str, default="", help="Prefix to add to all captions/tags (e.g., 'my_lora_style, ').")
    parser.add_argument("--caption_suffix", type=str, default="", help="Suffix to add to all captions/tags.")
    parser.add_argument("--is_tag_based_default", type=lambda x: (str(x).lower() == 'true'), default=True,
                        help="Default for 'is_tag_based' if using default_caption or filename as caption. If a caption file is found, it's assumed to be tags (True). (True/False)")
    parser.add_argument("--loss_weight", type=float, default=1.0, help="Default loss_weight for all entries.")

    args = parser.parse_args()

    create_metadata_jsonl(
        args.image_folder,
        args.output_file,
        args.caption_extension,
        args.default_caption,
        args.recursive,
        args.use_filename_as_caption_fallback,
        args.caption_prefix,
        args.caption_suffix,
        args.is_tag_based_default,
        args.loss_weight
    )

    print("\n--- Example usage for your training_config_chroma_lora.json ---")
    print("In the 'dataloader' section, make sure 'jsonl_metadata_path' points to your generated file,")
    print(f"and 'image_folder_path' points to the absolute path of '{os.path.abspath(args.image_folder)}'.")
    print("------------------------------------------------------------------")