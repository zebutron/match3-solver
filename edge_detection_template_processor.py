#!/usr/bin/env python3
"""
Unified Edge Detection Template Processor

Uses color-contrast-enhanced edge detection to create precise templates with color-based naming.
This approach:
1. Detects primary color in input images for meaningful naming (R, B, G, O, etc.)
2. Enhances color saturation and contrast (leveraging match-3 game design)
3. Uses color-based edge detection instead of grayscale conversion
4. Creates clean, precise boundaries for robust template matching
5. Preserves color mapping for better debugging and validation
"""

import cv2 as cv
import numpy as np
from pathlib import Path
import shutil
from collections import Counter

def detect_primary_color(image):
    """
    Detect the primary color in an image and return the corresponding letter.
    
    Args:
        image: Input BGR image
    
    Returns:
        Color letter (R, B, G, Y, P, O, C, etc.) based on dominant color
    """
    # Convert BGR to HSV for better color analysis
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    
    # Get non-black pixels (ignore background)
    mask = cv.inRange(hsv, (0, 30, 30), (180, 255, 255))
    hsv_masked = hsv[mask > 0]
    
    if len(hsv_masked) == 0:
        return "X"  # Unknown/no color detected
    
    # Get the median hue value
    median_hue = np.median(hsv_masked[:, 0])
    median_sat = np.median(hsv_masked[:, 1])
    median_val = np.median(hsv_masked[:, 2])
    
    print(f"   Color analysis: H={median_hue:.1f}, S={median_sat:.1f}, V={median_val:.1f}")
    
    # Color ranges in HSV (Hue is 0-179 in OpenCV)
    if median_sat < 50:  # Low saturation = grayscale/white
        return "W"  # White/Gray
    
    if median_hue <= 10 or median_hue >= 160:
        return "R"  # Red
    elif 11 <= median_hue <= 25:
        return "O"  # Orange
    elif 26 <= median_hue <= 35:
        return "Y"  # Yellow
    elif 36 <= median_hue <= 85:
        return "G"  # Green
    elif 86 <= median_hue <= 95:
        return "C"  # Cyan
    elif 96 <= median_hue <= 125:
        return "B"  # Blue
    elif 126 <= median_hue <= 159:
        return "P"  # Purple/Magenta
    else:
        return "X"  # Unknown

def enhance_color_contrast(image):
    """
    Enhance hue differences while flattening brightness variations.
    This maximizes color-to-color boundaries that match-3 games are designed around.
    """
    # Convert BGR to HSV for better color manipulation
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    
    # Maximize saturation (make colors as vivid as possible)
    hsv[:, :, 1] = cv.multiply(hsv[:, :, 1], 1.5)  # Increase saturation by 50%
    hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)   # Ensure values stay in range
    
    # Flatten brightness variations - normalize value to reduce light/dark contrast
    # This makes all colors roughly the same brightness so we focus on hue differences
    hsv[:, :, 2] = cv.multiply(hsv[:, :, 2], 0.8)  # Reduce brightness variations
    hsv[:, :, 2] = np.clip(hsv[:, :, 2], 80, 200)  # Constrain to middle brightness range
    
    # Convert back to BGR
    enhanced = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
    
    return enhanced

def detect_hue_based_edges(image, blur_kernel=5, threshold1=80, threshold2=160):
    """
    Detect edges based purely on hue differences rather than brightness differences.
    This focuses on color-to-color transitions that define piece boundaries.
    """
    # First enhance to maximize hue differences and flatten brightness
    enhanced = enhance_color_contrast(image)
    
    # Apply more blur to focus on major boundaries only
    blurred = cv.GaussianBlur(enhanced, (blur_kernel, blur_kernel), 0)
    
    # Convert to HSV to work directly with hue
    hsv = cv.cvtColor(blurred, cv.COLOR_BGR2HSV)
    hue_channel = hsv[:, :, 0]
    
    # Apply Canny edge detection directly to the hue channel with higher thresholds
    # This detects only strong transitions between different colors
    hue_edges = cv.Canny(hue_channel, threshold1, threshold2, apertureSize=5, L2gradient=True)
    
    # Much more conservative RGB edge detection
    b, g, r = cv.split(blurred)
    edges_b = cv.Canny(b, threshold1 + 20, threshold2 + 40, apertureSize=5, L2gradient=True)
    edges_g = cv.Canny(g, threshold1 + 20, threshold2 + 40, apertureSize=5, L2gradient=True)
    edges_r = cv.Canny(r, threshold1 + 20, threshold2 + 40, apertureSize=5, L2gradient=True)
    
    # Combine RGB edges conservatively
    rgb_edges = cv.bitwise_or(edges_b, cv.bitwise_or(edges_g, edges_r))
    
    # Strong emphasis on hue edges, minimal RGB contribution
    final_edges = cv.addWeighted(hue_edges, 0.8, rgb_edges, 0.2, 0)
    
    # Clean up to binary with higher threshold
    _, final_binary = cv.threshold(final_edges, 180, 255, cv.THRESH_BINARY)
    
    return final_binary

def create_edge_template(image, blur_kernel=3, canny_low=50, canny_high=150, dilate_iterations=1):
    """
    Create an edge-based template using hue-focused edge detection.
    """
    # Use hue-based edge detection instead of brightness-based
    edges = detect_hue_based_edges(image, blur_kernel, canny_low, canny_high)
    
    # Clean up edges with morphological operations
    if dilate_iterations > 0:
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2, 2))  # Smaller kernel for thinner edges
        edges_processed = cv.dilate(edges, kernel, iterations=dilate_iterations)
    else:
        edges_processed = edges
    
    # Invert so edges are white on black background
    edges_inverted = cv.bitwise_not(edges_processed)
    
    return edges_inverted

def create_multi_scale_edge_template(image):
    """
    Create template with very selective hue-based detection for clean boundaries.
    """
    # Ultra-selective hue edges (only the strongest boundaries)
    primary_edges = create_edge_template(image, blur_kernel=7, canny_low=100, canny_high=200, dilate_iterations=0)
    
    # Secondary selective edges (medium strength boundaries)
    secondary_edges = create_edge_template(image, blur_kernel=5, canny_low=80, canny_high=160, dilate_iterations=0)
    
    # Combine with primary emphasis - only use secondary where primary agrees
    combined = cv.bitwise_and(primary_edges, secondary_edges)
    
    # Aggressive cleanup to remove isolated pixels and noise
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    cleaned = cv.morphologyEx(combined, cv.MORPH_OPEN, kernel)
    
    # Additional erosion to thin edges
    kernel_erode = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2, 2))
    thinned = cv.erode(cleaned, kernel_erode, iterations=1)
    
    return thinned

def process_unified_edge_templates():
    """
    Process templates using color detection for naming and color-contrast-enhanced edge detection.
    """
    input_dir = Path("data/templates/royal-match/inputs/pieces")
    output_dir = Path("data/templates/royal-match/extracted/pieces")
    debug_dir = Path("debug/template_processing")
    
    print("🎨 Processing Templates with Color Detection + Color-Enhanced Edge Detection")
    print("=" * 70)
    
    if not input_dir.exists():
        print(f"❌ Input directory not found: {input_dir}")
        return
    
    # Clear output directory
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create debug directory
    debug_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all image files
    image_files = []
    for ext in ['*.png', '*.jpg', '*.jpeg']:
        image_files.extend(input_dir.glob(ext))
    
    if not image_files:
        print(f"❌ No image files found in {input_dir}")
        return
    
    print(f"📁 Found {len(image_files)} template images")
    print(f"🔍 Debug outputs will be saved to: {debug_dir}")
    
    # Track color mappings for debugging
    color_mappings = {}
    
    # Process each template
    for image_file in sorted(image_files):
        print(f"\n--- Processing {image_file.name} ---")
        
        # Load image
        img = cv.imread(str(image_file), cv.IMREAD_COLOR)
        if img is None:
            print(f"❌ Could not load {image_file.name}")
            continue
        
        original_shape = img.shape
        print(f"📐 Original size: {original_shape[1]}x{original_shape[0]}")
        
        # Step 0: Save original for comparison
        original_debug_path = debug_dir / f"{image_file.stem}_0_original.png"
        cv.imwrite(str(original_debug_path), img)
        print(f"💾 Original saved: {original_debug_path}")
        
        # Step 1: Detect primary color for naming
        print("🎨 Detecting primary color...")
        color_letter = detect_primary_color(img)
        print(f"✅ Primary color detected: {color_letter}")
        
        # Step 1.5: Apply and save color contrast enhancement
        print("🌈 Applying color contrast enhancement...")
        enhanced_img = enhance_color_contrast(img)
        enhanced_debug_path = debug_dir / f"{image_file.stem}_1_enhanced.png"
        cv.imwrite(str(enhanced_debug_path), enhanced_img)
        print(f"💾 Color-enhanced image saved: {enhanced_debug_path}")
        
        # Step 2: Create color-contrast-enhanced edge template
        print("🔍 Creating color-based edge template...")
        edge_template = create_multi_scale_edge_template(img)
        
        # Step 2.5: Save edge template before processing
        edge_debug_path = debug_dir / f"{image_file.stem}_2_edges.png"
        cv.imwrite(str(edge_debug_path), edge_template)
        print(f"💾 Edge template saved: {edge_debug_path}")
        
        # Step 3: Find content boundaries and crop whitespace
        print("✂️  Cropping whitespace and centering content...")
        
        # Find content boundaries (non-black pixels in edge template)
        content_pixels = np.where(edge_template == 255)  # White pixels (edges)
        
        if len(content_pixels[0]) > 0:
            # Get bounding box of actual content
            min_y, max_y = np.min(content_pixels[0]), np.max(content_pixels[0])
            min_x, max_x = np.min(content_pixels[1]), np.max(content_pixels[1])
            
            # Add padding around content
            padding = 8
            min_y = max(0, min_y - padding)
            max_y = min(edge_template.shape[0], max_y + padding)
            min_x = max(0, min_x - padding)
            max_x = min(edge_template.shape[1], max_x + padding)
            
            # Crop to content with padding
            cropped = edge_template[min_y:max_y, min_x:max_x]
            print(f"   Content bounds: ({min_x},{min_y}) to ({max_x},{max_y})")
            print(f"   Cropped size: {max_x-min_x}x{max_y-min_y}")
        else:
            # No content found, use original
            cropped = edge_template
            print(f"   No content detected, using original")
        
        # Step 4: Create uniform template size while preserving aspect ratio
        print("📐 Creating uniform template size...")
        
        # Target uniform size
        target_size = 64
        
        # Get cropped dimensions and calculate aspect ratio
        crop_height, crop_width = cropped.shape[:2]
        aspect_ratio = crop_width / crop_height
        
        # Calculate dimensions that preserve aspect ratio within target_size
        if crop_width > crop_height:
            # Width is larger
            new_width = target_size
            new_height = int(target_size / aspect_ratio)
        else:
            # Height is larger or equal
            new_height = target_size
            new_width = int(target_size * aspect_ratio)
        
        # Resize while preserving aspect ratio
        resized = cv.resize(cropped, (new_width, new_height), interpolation=cv.INTER_LANCZOS4)
        
        # Step 5: Center on uniform background
        print("🎯 Centering content on uniform background...")
        
        # Create uniform square background
        uniform_template = np.zeros((target_size, target_size), dtype=np.uint8)
        
        # Calculate centering position
        start_y = (target_size - new_height) // 2
        start_x = (target_size - new_width) // 2
        
        # Place resized content centered on uniform background
        uniform_template[start_y:start_y+new_height, start_x:start_x+new_width] = resized
        
        print(f"   Resized: {new_width}x{new_height} (aspect ratio: {aspect_ratio:.3f})")
        print(f"   Centered on: {target_size}x{target_size} at ({start_x},{start_y})")
        
        # Step 6: Final binary cleanup
        _, final_template = cv.threshold(uniform_template, 127, 255, cv.THRESH_BINARY)
        
        # Step 6.5: Save final template
        final_debug_path = debug_dir / f"{image_file.stem}_3_final.png"
        cv.imwrite(str(final_debug_path), final_template)
        print(f"💾 Final template saved: {final_debug_path}")
        
        # Step 7: Save with color-based name
        output_path = output_dir / f"{color_letter}.png"
        
        # Handle duplicate colors by adding numbers
        counter = 1
        while output_path.exists():
            output_path = output_dir / f"{color_letter}{counter}.png"
            counter += 1
        
        success = cv.imwrite(str(output_path), final_template)
        
        if success:
            print(f"✅ Saved: {output_path}")
            color_mappings[image_file.name] = output_path.name
            
            # Show edge detection stats
            edge_pixels = np.sum(final_template == 255)  # White pixels (edges)
            total_pixels = final_template.shape[0] * final_template.shape[1]
            edge_percentage = (edge_pixels / total_pixels) * 100
            
            print(f"   Edge density: {edge_percentage:.1f}% ({edge_pixels}/{total_pixels} pixels)")
            print(f"   Template size: {target_size}x{target_size} (uniform, content centered)")
            
            # Quality metrics
            if edge_percentage < 15:
                print(f"   Quality: ✨ Precise and elegant")
            elif edge_percentage < 30:
                print(f"   Quality: ✅ Good balance")
            elif edge_percentage < 50:
                print(f"   Quality: ⚠️  Moderate density")
            else:
                print(f"   Quality: ❌ Too dense")
        else:
            print(f"❌ Failed to save: {output_path}")
    
    print(f"\n🎉 Processing complete! {len(image_files)} color-named edge templates created.")
    print(f"📁 Output directory: {output_dir}")
    print(f"🔍 Debug files saved to: {debug_dir}")
    
    print(f"\n🗺️  Color Mapping:")
    for original, processed in color_mappings.items():
        print(f"   {original} → {processed}")
    
    print("\n✨ Templates now optimized for:")
    print("   • Color-based naming for easy identification")
    print("   • Color-contrast-enhanced edge detection")
    print("   • Leverages match-3 game color design principles")
    print("   • Robust template matching")
    print("   • Visual debugging and validation")
    
    print(f"\n🔍 Check debug directory for step-by-step processing:")
    print(f"   • *_0_original.png - Original input")
    print(f"   • *_1_enhanced.png - Color contrast enhanced")
    print(f"   • *_2_edges.png - Edge detection result")
    print(f"   • *_3_final.png - Final 64x64 template")

def create_color_legend():
    """
    Create a visual legend showing color mappings.
    """
    print("\n🎨 Color Legend:")
    print("=" * 20)
    print("R = Red")
    print("O = Orange") 
    print("Y = Yellow")
    print("G = Green")
    print("C = Cyan")
    print("B = Blue")
    print("P = Purple/Magenta")
    print("W = White/Gray")
    print("X = Unknown/Other")

def test_unified_templates_preview():
    """
    Preview the unified color-named edge detection results.
    """
    print("\n🔍 Unified Template Preview")
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
            
            # Quality assessment
            if edge_percentage < 15:
                quality = "✨ Precise"
            elif edge_percentage < 30:
                quality = "✅ Good"
            elif edge_percentage < 50:
                quality = "⚠️  Moderate"
            else:
                quality = "❌ Dense"
            
            color_name = template_file.stem
            print(f"{color_name}: {edge_percentage:.1f}% edges ({edge_pixels}/{total_pixels} pixels) - {quality}")
        else:
            print(f"❌ Could not load {template_file.name}")

def create_unified_comparison_visualization():
    """
    Create comparison showing original vs color-contrast-enhanced edge templates.
    """
    input_dir = Path("data/templates/royal-match/inputs/pieces")
    image_files = list(input_dir.glob("*.png"))
    
    if not image_files:
        print("❌ No input templates found for visualization")
        return
    
    print(f"\n👀 Creating Original vs Color-Enhanced Edge Detection Comparison")
    print("=" * 65)
    
    # Process all templates for comparison
    comparisons = []
    
    for i, image_file in enumerate(image_files[:4]):  # Limit to 4 for visualization
        img = cv.imread(str(image_file))
        if img is None:
            continue
        
        # Original resized
        orig_resized = cv.resize(img, (64, 64))
        
        # Color-enhanced edge-processed version
        edge_template = create_multi_scale_edge_template(img)
        edge_resized = cv.resize(edge_template, (64, 64), interpolation=cv.INTER_LANCZOS4)
        
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
        
        comparison_path = Path("unified_edge_comparison.png")
        cv.imwrite(str(comparison_path), comparison_large)
        
        print(f"💾 Comparison saved: {comparison_path}")
        print("   Left: Original color | Right: Color-enhanced edge-detected")

if __name__ == "__main__":
    # Run the unified processing pipeline
    create_color_legend()
    process_unified_edge_templates()
    test_unified_templates_preview()
    create_unified_comparison_visualization()
    
    print("\n🚀 Next Steps:")
    print("1. Bot script uses same color-contrast-enhanced edge detection")
    print("2. Templates are color-named (R, B, G, O) for easy debugging")
    print("3. Leverages match-3 game color design principles")
    print("4. Should significantly improve matching accuracy") 