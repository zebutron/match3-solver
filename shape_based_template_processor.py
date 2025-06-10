#!/usr/bin/env python3
"""
Shape-Based Template Processor

Processes templates to focus on shape matching rather than color:
1. NO cropping - preserve distinctive edges and shapes
2. Convert to high-contrast grayscale
3. Enhance edge definition for robust shape matching
4. Normalize to uniform size

This approach is more robust against color variations and obstacles.
"""

import cv2 as cv
import numpy as np
from pathlib import Path
import shutil

def enhance_edges_grayscale(image, contrast_alpha=2.0, brightness_beta=0):
    """
    Convert to high-contrast grayscale and enhance edges for shape matching.
    
    Args:
        image: Input BGR image
        contrast_alpha: Contrast multiplier (higher = more contrast)
        brightness_beta: Brightness offset
    
    Returns:
        High-contrast grayscale image optimized for shape matching
    """
    # Convert to grayscale
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    
    # Apply contrast enhancement
    enhanced = cv.convertScaleAbs(gray, alpha=contrast_alpha, beta=brightness_beta)
    
    # Optional: Apply slight Gaussian blur to reduce noise while preserving edges
    # denoised = cv.GaussianBlur(enhanced, (3, 3), 0)
    
    return enhanced

def create_edge_template(image):
    """
    Create a template optimized for edge/shape detection.
    
    Args:
        image: Input grayscale image
    
    Returns:
        Edge-enhanced template
    """
    # Apply adaptive histogram equalization for better contrast
    clahe = cv.createCLAHE(clipLimit=3.0, tileGridSize=(4,4))
    equalized = clahe.apply(image)
    
    # Optional: Add subtle edge detection to emphasize shapes
    # This can help with shape matching
    edges = cv.Canny(equalized, 50, 150)
    
    # Combine original with edge information (subtle)
    # 90% original + 10% edges for shape emphasis
    combined = cv.addWeighted(equalized, 0.9, edges, 0.1, 0)
    
    return combined

def process_shape_templates():
    """
    Process templates for shape-based matching.
    """
    input_dir = Path("data/templates/royal-match/inputs/pieces")
    output_dir = Path("data/templates/royal-match/extracted/pieces")
    
    print("🔶 Processing Templates for Shape-Based Matching")
    print("=" * 55)
    
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
        
        # Step 1: Convert to high-contrast grayscale (NO CROPPING!)
        print("🔳 Converting to high-contrast grayscale...")
        gray_enhanced = enhance_edges_grayscale(img, contrast_alpha=2.0, brightness_beta=0)
        
        # Step 2: Resize to standard 64x64 (preserving aspect ratio info)
        print("📏 Resizing to 64x64...")
        resized = cv.resize(gray_enhanced, (64, 64), interpolation=cv.INTER_CUBIC)
        
        # Step 3: Create edge-enhanced template for shape matching
        print("🔍 Enhancing edges for shape detection...")
        shape_template = create_edge_template(resized)
        
        # Step 4: Save processed template (grayscale)
        output_path = output_dir / f"{i}.png"
        success = cv.imwrite(str(output_path), shape_template)
        
        if success:
            print(f"✅ Saved: {output_path}")
            
            # Show some stats
            mean_val = np.mean(shape_template)
            std_val = np.std(shape_template)
            print(f"   Template stats: Mean={mean_val:.1f}, Std={std_val:.1f}")
        else:
            print(f"❌ Failed to save: {output_path}")
    
    print(f"\n🎉 Processing complete! {len(image_files)} shape-based templates created.")
    print(f"📁 Output directory: {output_dir}")
    print("\n✨ Templates now optimized for:")
    print("   • Shape-based matching (edges preserved)")
    print("   • Grayscale (color-independent)")
    print("   • High contrast (robust detection)")

def test_shape_templates_preview():
    """
    Quick preview of the processed shape templates.
    """
    print("\n🔍 Shape Template Preview")
    print("=" * 30)
    
    templates_dir = Path("data/templates/royal-match/extracted/pieces")
    template_files = list(templates_dir.glob("*.png"))
    
    if not template_files:
        print("❌ No processed templates found")
        return
    
    for template_file in sorted(template_files):
        img = cv.imread(str(template_file), cv.IMREAD_GRAYSCALE)
        if img is not None:
            mean_val = np.mean(img)
            std_val = np.std(img)
            min_val = np.min(img)
            max_val = np.max(img)
            print(f"{template_file.name}: Mean={mean_val:.1f}, Std={std_val:.1f}, Range=[{min_val}-{max_val}]")
        else:
            print(f"❌ Could not load {template_file.name}")

def create_comparison_visualization():
    """
    Create before/after comparison showing color vs shape processing.
    """
    input_dir = Path("data/templates/royal-match/inputs/pieces")
    image_files = list(input_dir.glob("*.png"))
    
    if not image_files:
        print("❌ No input templates found for visualization")
        return
    
    print(f"\n👀 Creating Color vs Shape Comparison")
    print("=" * 40)
    
    # Process first template as example
    first_template = image_files[0]
    img = cv.imread(str(first_template))
    
    if img is None:
        print("❌ Could not load template for visualization")
        return
    
    # Original color
    orig_resized = cv.resize(img, (64, 64))
    
    # Shape-processed version
    gray_enhanced = enhance_edges_grayscale(img, contrast_alpha=2.0)
    gray_resized = cv.resize(gray_enhanced, (64, 64))
    shape_template = create_edge_template(gray_resized)
    
    # Convert grayscale back to 3-channel for side-by-side comparison
    shape_bgr = cv.cvtColor(shape_template, cv.COLOR_GRAY2BGR)
    
    # Create side-by-side comparison
    comparison = np.hstack([orig_resized, np.ones((64, 10, 3), dtype=np.uint8) * 255, shape_bgr])
    
    # Scale up for better visibility
    comparison_large = cv.resize(comparison, (400, 150), interpolation=cv.INTER_NEAREST)
    
    comparison_path = Path("color_vs_shape_comparison.png")
    cv.imwrite(str(comparison_path), comparison_large)
    
    print(f"💾 Comparison saved: {comparison_path}")
    print("   Left: Original color | Right: Shape-optimized grayscale")

if __name__ == "__main__":
    # Run the shape-based processing pipeline
    process_shape_templates()
    test_shape_templates_preview()
    create_comparison_visualization()
    
    print("\n🚀 Next Steps:")
    print("1. Test shape-based templates with grayscale board matching")
    print("2. Templates should now be robust against color variations")
    print("3. Shape matching should work despite obstacles on board") 