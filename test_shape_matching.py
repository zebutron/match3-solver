#!/usr/bin/env python3
"""
Test Shape-Based Template Matching

Test the new shape-based grayscale templates against a grayscale version
of the live board to see if shape matching is more robust than color matching.
"""

import cv2 as cv
import numpy as np
import pyautogui
from pathlib import Path

def enhance_board_grayscale(image, contrast_alpha=2.0, brightness_beta=0):
    """
    Apply the same grayscale enhancement to the board as used for templates.
    """
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    enhanced = cv.convertScaleAbs(gray, alpha=contrast_alpha, beta=brightness_beta)
    return enhanced

def test_shape_based_matching():
    print("🔶 Testing Shape-Based Template Matching")
    print("=" * 50)
    
    # Load saved coordinates
    coords_file = Path("data/templates/royal-match/coordinates.txt")
    with open(coords_file, 'r') as f:
        coords = f.read().strip().split(',')
        x1, y1, x2, y2 = int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])
    
    # Capture live board
    width = x2 - x1
    height = y2 - y1
    screenshot = pyautogui.screenshot(region=(x1, y1, width, height))
    live_board_color = cv.cvtColor(np.array(screenshot), cv.COLOR_RGB2BGR)
    
    # Convert board to same grayscale format as templates
    live_board_gray = enhance_board_grayscale(live_board_color, contrast_alpha=2.0)
    
    print(f"📸 Captured live board: {live_board_color.shape}")
    print(f"🔳 Converted to grayscale: {live_board_gray.shape}")
    
    # Load shape-based templates
    templates_dir = Path("data/templates/royal-match/extracted/pieces")
    template_files = sorted(list(templates_dir.glob("*.png")))
    
    print(f"📋 Testing {len(template_files)} shape-based templates\n")
    
    # Test different thresholds for each template
    threshold_range = np.arange(0.1, 0.9, 0.05)
    optimal_thresholds = {}
    
    for template_file in template_files:
        print(f"--- Template {template_file.name} ---")
        
        # Load template (already grayscale)
        tmpl = cv.imread(str(template_file), cv.IMREAD_GRAYSCALE)
        if tmpl is None:
            continue
        
        # Scale template (0.5 scale as used by bot)
        scale = 0.5
        new_width = int(tmpl.shape[1] * scale)
        new_height = int(tmpl.shape[0] * scale)
        tmpl_scaled = cv.resize(tmpl, (new_width, new_height), interpolation=cv.INTER_CUBIC)
        
        print(f"Template size: {tmpl_scaled.shape}")
        
        # Test template matching (grayscale vs grayscale)
        res = cv.matchTemplate(live_board_gray, tmpl_scaled, cv.TM_CCOEFF_NORMED)
        max_confidence = np.max(res)
        
        print(f"Max confidence: {max_confidence:.3f}")
        
        # Find optimal threshold
        best_threshold = None
        best_count = 0
        
        print("Threshold tests:")
        for threshold in threshold_range:
            matches = len(np.where(res >= threshold)[0])
            print(f"  @ {threshold:.2f}: {matches:3d} matches")
            
            # Target 5-40 matches (reasonable for individual piece types)
            if 5 <= matches <= 40:
                if best_threshold is None or abs(matches - 20) < abs(best_count - 20):
                    best_threshold = threshold
                    best_count = matches
        
        if best_threshold is not None:
            optimal_thresholds[template_file.name] = best_threshold
            print(f"✅ Optimal threshold: {best_threshold:.2f} ({best_count} matches)")
        else:
            # Find the threshold that gives closest to 20 matches
            match_counts = []
            for threshold in threshold_range:
                matches = len(np.where(res >= threshold)[0])
                match_counts.append((threshold, matches))
            
            # Find closest to 20
            best_match = min(match_counts, key=lambda x: abs(x[1] - 20))
            optimal_thresholds[template_file.name] = best_match[0]
            print(f"🔧 Fallback threshold: {best_match[0]:.2f} ({best_match[1]} matches)")
        
        print()
    
    # Summary
    print("📊 SHAPE-BASED MATCHING RESULTS:")
    print("=" * 40)
    total_matches = 0
    for template_name, threshold in optimal_thresholds.items():
        # Re-test with optimal threshold
        template_file = templates_dir / template_name
        tmpl = cv.imread(str(template_file), cv.IMREAD_GRAYSCALE)
        tmpl_scaled = cv.resize(tmpl, (int(tmpl.shape[1] * 0.5), int(tmpl.shape[0] * 0.5)), interpolation=cv.INTER_CUBIC)
        res = cv.matchTemplate(live_board_gray, tmpl_scaled, cv.TM_CCOEFF_NORMED)
        matches = len(np.where(res >= threshold)[0])
        total_matches += matches
        print(f"{template_name}: {threshold:.3f} → {matches:2d} matches")
    
    print(f"\nTotal detected pieces: {total_matches}")
    print(f"Expected range: 40-80 pieces for 8x8 board")
    
    if 40 <= total_matches <= 80:
        print("✅ EXCELLENT! Shape matching works perfectly")
    elif total_matches < 40:
        print("⚠️  Low match count - may need lower thresholds")
    else:
        print("⚠️  High match count - may need higher thresholds")
    
    # Save the grayscale board for inspection
    board_save_path = Path("grayscale_board_capture.png")
    cv.imwrite(str(board_save_path), live_board_gray)
    print(f"\n💾 Grayscale board saved: {board_save_path}")
    
    return optimal_thresholds

def compare_color_vs_shape_matching():
    """
    Direct comparison between color-based and shape-based matching.
    """
    print("\n🔄 Comparing Color vs Shape Matching")
    print("=" * 45)
    
    # Load saved coordinates
    coords_file = Path("data/templates/royal-match/coordinates.txt")
    with open(coords_file, 'r') as f:
        coords = f.read().strip().split(',')
        x1, y1, x2, y2 = int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])
    
    # Capture live board
    width = x2 - x1
    height = y2 - y1
    screenshot = pyautogui.screenshot(region=(x1, y1, width, height))
    live_board_color = cv.cvtColor(np.array(screenshot), cv.COLOR_RGB2BGR)
    live_board_gray = enhance_board_grayscale(live_board_color, contrast_alpha=2.0)
    
    # Load templates
    templates_dir = Path("data/templates/royal-match/extracted/pieces")
    template_files = sorted(list(templates_dir.glob("*.png")))
    
    print("Template | Color Max | Shape Max | Difference")
    print("-" * 45)
    
    for template_file in template_files:
        # Load template and create both versions
        tmpl_gray = cv.imread(str(template_file), cv.IMREAD_GRAYSCALE)
        tmpl_color = cv.cvtColor(tmpl_gray, cv.COLOR_GRAY2BGR)  # Convert back to color
        
        # Scale both
        scale = 0.5
        new_width = int(tmpl_gray.shape[1] * scale)
        new_height = int(tmpl_gray.shape[0] * scale)
        
        tmpl_gray_scaled = cv.resize(tmpl_gray, (new_width, new_height), interpolation=cv.INTER_CUBIC)
        tmpl_color_scaled = cv.resize(tmpl_color, (new_width, new_height), interpolation=cv.INTER_CUBIC)
        
        # Test both matching approaches
        res_color = cv.matchTemplate(live_board_color, tmpl_color_scaled, cv.TM_CCOEFF_NORMED)
        res_shape = cv.matchTemplate(live_board_gray, tmpl_gray_scaled, cv.TM_CCOEFF_NORMED)
        
        color_max = np.max(res_color)
        shape_max = np.max(res_shape)
        difference = shape_max - color_max
        
        print(f"{template_file.name:8} | {color_max:8.3f} | {shape_max:8.3f} | {difference:+8.3f}")
    
    print("\nIf shape matching consistently scores higher, it's working better!")

if __name__ == "__main__":
    optimal_thresholds = test_shape_based_matching()
    compare_color_vs_shape_matching()
    
    print(f"\n🚀 Shape-based thresholds ready for bot:")
    for template_name, threshold in optimal_thresholds.items():
        print(f"  {template_name}: {threshold:.3f}")
    
    print("\n✨ Key advantages of shape-based matching:")
    print("   • Robust against color variations")
    print("   • Works despite obstacles on board")
    print("   • Focus on piece shape, not color")
    print("   • More distinctive edge information") 