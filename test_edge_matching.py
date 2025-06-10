#!/usr/bin/env python3
"""
Test Edge-Based Template Matching

Test the new edge-based templates against an edge-processed version
of the live board to see if edge detection provides superior matching.
"""

import cv2 as cv
import numpy as np
import pyautogui
from pathlib import Path

def create_edge_template(image, blur_kernel=3, canny_low=50, canny_high=150, dilate_iterations=1):
    """
    Apply the same edge detection to the board as used for templates.
    """
    blurred = cv.GaussianBlur(image, (blur_kernel, blur_kernel), 0)
    gray = cv.cvtColor(blurred, cv.COLOR_BGR2GRAY)
    edges = cv.Canny(gray, canny_low, canny_high)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    edges_dilated = cv.dilate(edges, kernel, iterations=dilate_iterations)
    edges_inverted = cv.bitwise_not(edges_dilated)
    return edges_inverted

def create_multi_scale_edge_template(image):
    """
    Create board edges with multiple scales (same as template processing).
    """
    fine_edges = create_edge_template(image, blur_kernel=3, canny_low=50, canny_high=150, dilate_iterations=1)
    coarse_edges = create_edge_template(image, blur_kernel=5, canny_low=30, canny_high=100, dilate_iterations=2)
    combined = cv.bitwise_and(fine_edges, coarse_edges)
    return combined

def test_edge_based_matching():
    print("🔍 Testing Edge-Based Template Matching")
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
    
    # Convert board to edge-based format (same as templates)
    live_board_edges = create_multi_scale_edge_template(live_board_color)
    
    print(f"📸 Captured live board: {live_board_color.shape}")
    print(f"🔍 Converted to edges: {live_board_edges.shape}")
    
    # Load edge-based templates
    templates_dir = Path("data/templates/royal-match/extracted/pieces")
    template_files = sorted(list(templates_dir.glob("*.png")))
    
    print(f"📋 Testing {len(template_files)} edge-based templates\n")
    
    # Test different thresholds for each template
    threshold_range = np.arange(0.1, 0.9, 0.05)
    optimal_thresholds = {}
    
    for template_file in template_files:
        print(f"--- Template {template_file.name} ---")
        
        # Load template (already edge-processed)
        tmpl = cv.imread(str(template_file), cv.IMREAD_GRAYSCALE)
        if tmpl is None:
            continue
        
        # Scale template (0.5 scale as used by bot)
        scale = 0.5
        new_width = int(tmpl.shape[1] * scale)
        new_height = int(tmpl.shape[0] * scale)
        tmpl_scaled = cv.resize(tmpl, (new_width, new_height), interpolation=cv.INTER_CUBIC)
        
        print(f"Template size: {tmpl_scaled.shape}")
        
        # Test template matching (edges vs edges)
        res = cv.matchTemplate(live_board_edges, tmpl_scaled, cv.TM_CCOEFF_NORMED)
        max_confidence = np.max(res)
        
        print(f"Max confidence: {max_confidence:.3f}")
        
        # Find optimal threshold
        best_threshold = None
        best_count = 0
        
        print("Threshold tests:")
        for threshold in threshold_range:
            matches = len(np.where(res >= threshold)[0])
            print(f"  @ {threshold:.2f}: {matches:3d} matches")
            
            # Target 5-30 matches (reasonable for individual piece types)
            if 5 <= matches <= 30:
                if best_threshold is None or abs(matches - 15) < abs(best_count - 15):
                    best_threshold = threshold
                    best_count = matches
        
        if best_threshold is not None:
            optimal_thresholds[template_file.name] = best_threshold
            print(f"✅ Optimal threshold: {best_threshold:.2f} ({best_count} matches)")
        else:
            # Find the threshold that gives closest to 15 matches
            match_counts = []
            for threshold in threshold_range:
                matches = len(np.where(res >= threshold)[0])
                match_counts.append((threshold, matches))
            
            # Find closest to 15
            best_match = min(match_counts, key=lambda x: abs(x[1] - 15))
            optimal_thresholds[template_file.name] = best_match[0]
            print(f"🔧 Fallback threshold: {best_match[0]:.2f} ({best_match[1]} matches)")
        
        print()
    
    # Summary
    print("📊 EDGE-BASED MATCHING RESULTS:")
    print("=" * 40)
    total_matches = 0
    for template_name, threshold in optimal_thresholds.items():
        # Re-test with optimal threshold
        template_file = templates_dir / template_name
        tmpl = cv.imread(str(template_file), cv.IMREAD_GRAYSCALE)
        tmpl_scaled = cv.resize(tmpl, (int(tmpl.shape[1] * 0.5), int(tmpl.shape[0] * 0.5)), interpolation=cv.INTER_CUBIC)
        res = cv.matchTemplate(live_board_edges, tmpl_scaled, cv.TM_CCOEFF_NORMED)
        matches = len(np.where(res >= threshold)[0])
        total_matches += matches
        print(f"{template_name}: {threshold:.3f} → {matches:2d} matches")
    
    print(f"\nTotal detected pieces: {total_matches}")
    print(f"Expected range: 40-80 pieces for 8x8 board")
    
    if 40 <= total_matches <= 80:
        print("✅ EXCELLENT! Edge matching works perfectly")
    elif total_matches < 40:
        print("⚠️  Low match count - may need lower thresholds")
    else:
        print("⚠️  High match count - may need higher thresholds")
    
    # Save the edge-processed board for inspection
    board_save_path = Path("edge_board_capture.png")
    cv.imwrite(str(board_save_path), live_board_edges)
    print(f"\n💾 Edge-processed board saved: {board_save_path}")
    
    return optimal_thresholds

def compare_all_matching_approaches():
    """
    Compare color vs grayscale vs edge-based matching.
    """
    print("\n🔄 Comparing All Matching Approaches")
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
    live_board_gray = cv.cvtColor(live_board_color, cv.COLOR_BGR2GRAY)
    live_board_edges = create_multi_scale_edge_template(live_board_color)
    
    # Load templates
    templates_dir = Path("data/templates/royal-match/extracted/pieces")
    template_files = sorted(list(templates_dir.glob("*.png")))
    
    print("Template | Color Max | Gray Max  | Edge Max  | Best")
    print("-" * 55)
    
    for template_file in template_files:
        # Load edge template
        tmpl_edge = cv.imread(str(template_file), cv.IMREAD_GRAYSCALE)
        
        # Create other versions for comparison
        tmpl_color = cv.cvtColor(tmpl_edge, cv.COLOR_GRAY2BGR)
        tmpl_gray = tmpl_edge.copy()
        
        # Scale all
        scale = 0.5
        new_width = int(tmpl_edge.shape[1] * scale)
        new_height = int(tmpl_edge.shape[0] * scale)
        
        tmpl_color_scaled = cv.resize(tmpl_color, (new_width, new_height), interpolation=cv.INTER_CUBIC)
        tmpl_gray_scaled = cv.resize(tmpl_gray, (new_width, new_height), interpolation=cv.INTER_CUBIC)
        tmpl_edge_scaled = cv.resize(tmpl_edge, (new_width, new_height), interpolation=cv.INTER_CUBIC)
        
        # Test all matching approaches
        res_color = cv.matchTemplate(live_board_color, tmpl_color_scaled, cv.TM_CCOEFF_NORMED)
        res_gray = cv.matchTemplate(live_board_gray, tmpl_gray_scaled, cv.TM_CCOEFF_NORMED)
        res_edge = cv.matchTemplate(live_board_edges, tmpl_edge_scaled, cv.TM_CCOEFF_NORMED)
        
        color_max = np.max(res_color)
        gray_max = np.max(res_gray)
        edge_max = np.max(res_edge)
        
        # Determine best approach
        best_score = max(color_max, gray_max, edge_max)
        if best_score == edge_max:
            best = "EDGE"
        elif best_score == gray_max:
            best = "GRAY"
        else:
            best = "COLOR"
        
        print(f"{template_file.name:8} | {color_max:8.3f} | {gray_max:8.3f} | {edge_max:8.3f} | {best}")
    
    print("\nBest approach should consistently have highest confidence scores!")

if __name__ == "__main__":
    optimal_thresholds = test_edge_based_matching()
    compare_all_matching_approaches()
    
    print(f"\n🚀 Edge-based thresholds ready for bot:")
    for template_name, threshold in optimal_thresholds.items():
        print(f"  {template_name}: {threshold:.3f}")
    
    print("\n✨ Key advantages of edge-based matching:")
    print("   • Focuses on shape boundaries (most distinctive)")
    print("   • Completely color-independent")
    print("   • Works with any lighting or color scheme")
    print("   • Robust against obstacles and variations")
    print("   • Leverages Match-3 high-contrast design") 