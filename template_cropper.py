#!/usr/bin/env python3
"""
Template Cropper and Normalizer

Processes raw template images to create distinctive templates:
1. Gentle cropping to remove border areas
2. Resize to uniform 64x64 pixels
3. Enhance contrast for better matching

This addresses the issue where templates match too broadly by focusing
on the distinctive center portions of each piece.
"""

import cv2 as cv
import numpy as np
from pathlib import Path
import shutil

def crop_template_center(image, crop_percentage=0.15):
    """
    Gently crop the template to focus on the center distinctive area.
    
    Args:
        image: Input image array
        crop_percentage: Percentage to crop from each edge (0.15 = 15% from each side)
    
    Returns:
        Cropped image focusing on center distinctive area
    """
    h, w = image.shape[:2]
    
    # Calculate crop amounts
    crop_h = int(h * crop_percentage)
    crop_w = int(w * crop_percentage)
    
    # Ensure we don't crop too much
    crop_h = min(crop_h, h // 4)  # Max 25% crop
    crop_w = min(crop_w, w // 4)  # Max 25% crop
    
    # Crop from all sides
    cropped = image[crop_h:h-crop_h, crop_w:w-crop_w]
    
    return cropped

def enhance_contrast(image, alpha=1.2, beta=10):
    """
    Enhance image contrast to make distinctive features more prominent.
    
    Args:
        image: Input image
        alpha: Contrast multiplier (1.0 = no change, >1.0 = more contrast)
        beta: Brightness offset
    
    Returns:
        Contrast-enhanced image
    """
    enhanced = cv.convertScaleAbs(image, alpha=alpha, beta=beta)
    return enhanced

def process_templates():
    """
    Process all templates with cropping and normalization.
    """
    input_dir = Path("data/templates/royal-match/inputs/pieces")
    output_dir = Path("data/templates/royal-match/extracted/pieces")
    
    print("🔧 Processing Templates with Gentle Cropping")
    print("=" * 50)
    
    if not input_dir.exists():
        print(f"❌ Input directory not found: {input_dir}")
        return
    
    # Clear output directory
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all image files
    image_files = []
    for ext in ['*.png', '*.jpg', '*.jpeg']:
        image_files.extend(input_dir.glob(ext))
    
    if not image_files:
        print(f"❌ No image files found in {input_dir}")
        return
    
    print(f"📁 Found {len(image_files)} template images")
    
    # Process each template
    for i, image_file in enumerate(sorted(image_files), 1):
        print(f"\n--- Processing {image_file.name} ---")
        
        # Load image
        img = cv.imread(str(image_file), cv.IMREAD_COLOR)
        if img is None:
            print(f"❌ Could not load {image_file.name}")
            continue
        
        original_shape = img.shape
        print(f"📐 Original size: {original_shape[1]}x{original_shape[0]}")
        
        # Step 1: Gentle cropping to remove borders
        print("✂️  Applying gentle crop (15% border removal)...")
        cropped = crop_template_center(img, crop_percentage=0.15)
        crop_shape = cropped.shape
        print(f"📐 After crop: {crop_shape[1]}x{crop_shape[0]}")
        
        # Step 2: Resize to standard 64x64
        print("📏 Resizing to 64x64...")
        resized = cv.resize(cropped, (64, 64), interpolation=cv.INTER_CUBIC)
        
        # Step 3: Enhance contrast for better matching
        print("🎨 Enhancing contrast...")
        enhanced = enhance_contrast(resized, alpha=1.1, beta=5)
        
        # Step 4: Save processed template
        output_path = output_dir / f"{i}.png"
        success = cv.imwrite(str(output_path), enhanced)
        
        if success:
            print(f"✅ Saved: {output_path}")
        else:
            print(f"❌ Failed to save: {output_path}")
    
    print(f"\n🎉 Processing complete! {len(image_files)} templates processed.")
    print(f"📁 Output directory: {output_dir}")

def test_processed_templates():
    """
    Quick test to verify the processed templates look reasonable.
    """
    print("\n🔍 Testing Processed Templates")
    print("=" * 30)
    
    templates_dir = Path("data/templates/royal-match/extracted/pieces")
    template_files = list(templates_dir.glob("*.png"))
    
    if not template_files:
        print("❌ No processed templates found")
        return
    
    for template_file in sorted(template_files):
        img = cv.imread(str(template_file))
        if img is not None:
            # Calculate some basic stats
            mean_color = np.mean(img, axis=(0,1))
            std_color = np.std(img, axis=(0,1))
            print(f"{template_file.name}: Mean BGR={mean_color.astype(int)}, Std={std_color.astype(int)}")
        else:
            print(f"❌ Could not load {template_file.name}")

def visualize_cropping_effect():
    """
    Create a comparison showing before/after cropping effect.
    """
    input_dir = Path("data/templates/royal-match/inputs/pieces")
    image_files = list(input_dir.glob("*.png"))
    
    if not image_files:
        print("❌ No raw templates found for visualization")
        return
    
    print(f"\n👀 Cropping Effect Visualization")
    print("=" * 35)
    
    # Process first template as example
    first_template = image_files[0]
    img = cv.imread(str(first_template))
    
    if img is None:
        print("❌ Could not load template for visualization")
        return
    
    # Show original vs cropped
    cropped = crop_template_center(img, crop_percentage=0.15)
    
    print(f"Original: {img.shape[1]}x{img.shape[0]}")
    print(f"Cropped:  {cropped.shape[1]}x{cropped.shape[0]}")
    
    # Save comparison
    comparison_path = Path("template_cropping_comparison.png")
    
    # Resize both to same height for side-by-side comparison
    height = 200
    orig_resized = cv.resize(img, (int(img.shape[1] * height / img.shape[0]), height))
    crop_resized = cv.resize(cropped, (int(cropped.shape[1] * height / cropped.shape[0]), height))
    
    # Create side-by-side comparison
    comparison = np.hstack([orig_resized, np.ones((height, 10, 3), dtype=np.uint8) * 255, crop_resized])
    cv.imwrite(str(comparison_path), comparison)
    
    print(f"💾 Comparison saved: {comparison_path}")

if __name__ == "__main__":
    # Run the complete processing pipeline
    process_templates()
    test_processed_templates()
    visualize_cropping_effect()
    
    print("\n🚀 Next Steps:")
    print("1. Test the new cropped templates with the bot")
    print("2. Check if match counts are more reasonable (10-50 per template)")
    print("3. Adjust crop percentage if needed (currently 15%)") 