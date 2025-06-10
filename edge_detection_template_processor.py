#!/usr/bin/env python3
"""
Edge Detection Template Processor

Uses edge detection to focus on sharp color transitions between pieces and board.
This approach:
1. Finds boundaries between piece colors and board background
2. Preserves distinctive shape information regardless of color
3. Works with any color scheme or obstacles
4. Focuses on the most visually distinctive information

Match-3 games are designed with high contrast for visual clarity - we leverage this.
"""

import cv2 as cv
import numpy as np
from pathlib import Path
import shutil

def create_edge_template(image, blur_kernel=3, canny_low=50, canny_high=150, dilate_iterations=1):
    """
    Create an edge-based template that focuses on shape boundaries.
    
    Args:
        image: Input BGR image
        blur_kernel: Gaussian blur kernel size (odd number)
        canny_low: Lower threshold for Canny edge detection
        canny_high: Upper threshold for Canny edge detection
        dilate_iterations: Number of dilation iterations to thicken edges
    
    Returns:
        Edge-enhanced template emphasizing shape boundaries
    """
    # Step 1: Apply slight blur to reduce noise
    blurred = cv.GaussianBlur(image, (blur_kernel, blur_kernel), 0)
    
    # Step 2: Convert to grayscale for edge detection
    gray = cv.cvtColor(blurred, cv.COLOR_BGR2GRAY)
    
    # Step 3: Apply Canny edge detection to find sharp transitions
    edges = cv.Canny(gray, canny_low, canny_high)
    
    # Step 4: Dilate edges slightly to make them more prominent
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    edges_dilated = cv.dilate(edges, kernel, iterations=dilate_iterations)
    
    # Step 5: Invert so edges are white on black background
    edges_inverted = cv.bitwise_not(edges_dilated)
    
    return edges_inverted

def create_multi_scale_edge_template(image):
    """
    Create template with multiple edge detection scales for robustness.
    
    Args:
        image: Input BGR image
    
    Returns:
        Combined edge template with multiple scales
    """
    # Fine edges (detailed boundaries)
    fine_edges = create_edge_template(image, blur_kernel=3, canny_low=50, canny_high=150, dilate_iterations=1)
    
    # Coarse edges (major boundaries)
    coarse_edges = create_edge_template(image, blur_kernel=5, canny_low=30, canny_high=100, dilate_iterations=2)
    
    # Combine both scales (fine details + major boundaries)
    combined = cv.bitwise_and(fine_edges, coarse_edges)
    
    return combined

def process_edge_templates():
    """
    Process templates using edge detection for shape-based matching.
    """
    input_dir = Path("data/templates/royal-match/inputs/pieces")
    output_dir = Path("data/templates/royal-match/extracted/pieces")
    
    print("🔍 Processing Templates with Edge Detection")
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
        
        # Step 1: Create edge-based template (NO CROPPING!)
        print("🔍 Detecting edges and boundaries...")
        edge_template = create_multi_scale_edge_template(img)
        
        # Step 2: Resize to standard 64x64
        print("📏 Resizing to 64x64...")
        resized = cv.resize(edge_template, (64, 64), interpolation=cv.INTER_CUBIC)
        
        # Step 3: Save processed template
        output_path = output_dir / f"{i}.png"
        success = cv.imwrite(str(output_path), resized)
        
        if success:
            print(f"✅ Saved: {output_path}")
            
            # Show edge detection stats
            edge_pixels = np.sum(resized == 255)  # White pixels (edges)
            total_pixels = resized.shape[0] * resized.shape[1]
            edge_percentage = (edge_pixels / total_pixels) * 100
            
            print(f"   Edge density: {edge_percentage:.1f}% ({edge_pixels}/{total_pixels} pixels)")
        else:
            print(f"❌ Failed to save: {output_path}")
    
    print(f"\n🎉 Processing complete! {len(image_files)} edge-based templates created.")
    print(f"📁 Output directory: {output_dir}")
    print("\n✨ Templates now optimized for:")
    print("   • Edge-based shape matching")
    print("   • Boundary detection (piece vs background)")
    print("   • Color-independent matching")
    print("   • High contrast shape information")

def test_edge_templates_preview():
    """
    Preview the edge detection results.
    """
    print("\n🔍 Edge Template Preview")
    print("=" * 30)
    
    templates_dir = Path("data/templates/royal-match/extracted/pieces")
    template_files = list(templates_dir.glob("*.png"))
    
    if not template_files:
        print("❌ No processed templates found")
        return
    
    for template_file in sorted(template_files):
        img = cv.imread(str(template_file), cv.IMREAD_GRAYSCALE)
        if img is not None:
            edge_pixels = np.sum(img == 255)  # White pixels (edges)
            total_pixels = img.shape[0] * img.shape[1]
            edge_percentage = (edge_pixels / total_pixels) * 100
            
            print(f"{template_file.name}: {edge_percentage:.1f}% edges ({edge_pixels}/{total_pixels} pixels)")
        else:
            print(f"❌ Could not load {template_file.name}")

def create_edge_comparison_visualization():
    """
    Create comparison showing original vs edge-detected templates.
    """
    input_dir = Path("data/templates/royal-match/inputs/pieces")
    image_files = list(input_dir.glob("*.png"))
    
    if not image_files:
        print("❌ No input templates found for visualization")
        return
    
    print(f"\n👀 Creating Original vs Edge Detection Comparison")
    print("=" * 50)
    
    # Process all templates for comparison
    comparisons = []
    
    for i, image_file in enumerate(image_files[:4]):  # Limit to 4 for visualization
        img = cv.imread(str(image_file))
        if img is None:
            continue
        
        # Original resized
        orig_resized = cv.resize(img, (64, 64))
        
        # Edge-processed version
        edge_template = create_multi_scale_edge_template(img)
        edge_resized = cv.resize(edge_template, (64, 64))
        
        # Convert edge template to 3-channel for comparison
        edge_bgr = cv.cvtColor(edge_resized, cv.COLOR_GRAY2BGR)
        
        # Side-by-side comparison for this template
        comparison = np.hstack([orig_resized, np.ones((64, 10, 3), dtype=np.uint8) * 255, edge_bgr])
        comparisons.append(comparison)
    
    if comparisons:
        # Stack all comparisons vertically
        full_comparison = np.vstack(comparisons)
        
        # Scale up for better visibility
        comparison_large = cv.resize(full_comparison, (400, len(comparisons) * 80), interpolation=cv.INTER_NEAREST)
        
        comparison_path = Path("original_vs_edge_comparison.png")
        cv.imwrite(str(comparison_path), comparison_large)
        
        print(f"💾 Comparison saved: {comparison_path}")
        print("   Left: Original color | Right: Edge-detected")

if __name__ == "__main__":
    # Run the edge-based processing pipeline
    process_edge_templates()
    test_edge_templates_preview()
    create_edge_comparison_visualization()
    
    print("\n🚀 Next Steps:")
    print("1. Test edge-based templates with edge-processed board")
    print("2. Should work excellent with any color scheme")
    print("3. Robust against obstacles and color variations")
    print("4. Focuses on the most distinctive visual information") 