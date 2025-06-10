#!/usr/bin/env python3
"""
Test Cropped Templates with Threshold Optimization

Find optimal thresholds for the newly cropped templates to achieve
realistic match counts (10-50 matches per template).
"""

import cv2 as cv
import numpy as np
import pyautogui
from pathlib import Path

def find_optimal_thresholds_for_cropped():
    print("🎯 Finding Optimal Thresholds for Cropped Templates")
    print("=" * 55)
    
    # Load saved coordinates
    coords_file = Path("data/templates/royal-match/coordinates.txt")
    with open(coords_file, 'r') as f:
        coords = f.read().strip().split(',')
        x1, y1, x2, y2 = int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])
    
    # Capture live board
    width = x2 - x1
    height = y2 - y1
    screenshot = pyautogui.screenshot(region=(x1, y1, width, height))
    live_board = cv.cvtColor(np.array(screenshot), cv.COLOR_RGB2BGR)
    print(f"📸 Captured live board: {live_board.shape}")
    
    # Load templates
    templates_dir = Path("data/templates/royal-match/extracted/pieces")
    template_files = sorted(list(templates_dir.glob("*.png")))
    
    print(f"📋 Testing {len(template_files)} cropped templates\n")
    
    # Test different thresholds for each template
    threshold_range = np.arange(0.1, 0.8, 0.05)  # 0.1 to 0.8 in steps of 0.05
    optimal_thresholds = {}
    
    for template_file in template_files:
        print(f"--- Template {template_file.name} ---")
        
        # Load template
        tmpl = cv.imread(str(template_file), cv.IMREAD_COLOR)
        if tmpl is None:
            continue
        
        # Scale template (0.5 scale as used by bot)
        scale = 0.5
        new_width = int(tmpl.shape[1] * scale)
        new_height = int(tmpl.shape[0] * scale)
        tmpl_scaled = cv.resize(tmpl, (new_width, new_height), interpolation=cv.INTER_CUBIC)
        
        # Test template matching
        res = cv.matchTemplate(live_board, tmpl_scaled, cv.TM_CCOEFF_NORMED)
        max_confidence = np.max(res)
        
        print(f"Max confidence: {max_confidence:.3f}")
        
        # Find optimal threshold
        best_threshold = None
        best_count = 0
        
        print("Threshold tests:")
        for threshold in threshold_range:
            matches = len(np.where(res >= threshold)[0])
            print(f"  @ {threshold:.2f}: {matches:3d} matches")
            
            # Target 10-50 matches
            if 10 <= matches <= 50:
                if best_threshold is None or abs(matches - 30) < abs(best_count - 30):
                    best_threshold = threshold
                    best_count = matches
        
        if best_threshold is not None:
            optimal_thresholds[template_file.name] = best_threshold
            print(f"✅ Optimal threshold: {best_threshold:.2f} ({best_count} matches)")
        else:
            # Find the threshold that gives closest to 30 matches
            match_counts = []
            for threshold in threshold_range:
                matches = len(np.where(res >= threshold)[0])
                match_counts.append((threshold, matches))
            
            # Find closest to 30
            best_match = min(match_counts, key=lambda x: abs(x[1] - 30))
            optimal_thresholds[template_file.name] = best_match[0]
            print(f"🔧 Fallback threshold: {best_match[0]:.2f} ({best_match[1]} matches)")
        
        print()
    
    # Summary
    print("📊 OPTIMAL THRESHOLDS SUMMARY:")
    print("=" * 35)
    total_matches = 0
    for template_name, threshold in optimal_thresholds.items():
        # Re-test with optimal threshold
        template_file = templates_dir / template_name
        tmpl = cv.imread(str(template_file), cv.IMREAD_COLOR)
        tmpl_scaled = cv.resize(tmpl, (int(tmpl.shape[1] * 0.5), int(tmpl.shape[0] * 0.5)), interpolation=cv.INTER_CUBIC)
        res = cv.matchTemplate(live_board, tmpl_scaled, cv.TM_CCOEFF_NORMED)
        matches = len(np.where(res >= threshold)[0])
        total_matches += matches
        print(f"{template_name}: {threshold:.3f} → {matches:2d} matches")
    
    print(f"\nTotal detected pieces: {total_matches}")
    print(f"Expected range: 40-80 pieces for 8x8 board")
    
    if 40 <= total_matches <= 80:
        print("✅ EXCELLENT! Match count is in realistic range")
    elif total_matches < 40:
        print("⚠️  Low match count - may need lower thresholds")
    else:
        print("⚠️  High match count - may need higher thresholds")
    
    return optimal_thresholds

if __name__ == "__main__":
    optimal_thresholds = find_optimal_thresholds_for_cropped()
    
    print(f"\n🚀 Ready to update bot with these thresholds:")
    for template_name, threshold in optimal_thresholds.items():
        print(f"  {template_name}: {threshold:.3f}") 