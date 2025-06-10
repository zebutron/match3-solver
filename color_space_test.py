#!/usr/bin/env python3
"""
Color Space Mismatch Test

Test if low template matching scores are caused by color space differences
between saved PNG templates and live iPhone mirroring feed.
"""

import cv2 as cv
import numpy as np
import pyautogui
from pathlib import Path

def test_color_vs_grayscale():
    print("🎨 Testing Color Space Mismatch Hypothesis")
    print("=" * 50)
    
    # Load saved coordinates
    coords_file = Path("data/templates/royal-match/coordinates.txt")
    if not coords_file.exists():
        print("❌ No saved coordinates found")
        return
        
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
    template_files = list(templates_dir.glob("*.png"))
    
    if not template_files:
        print("❌ No templates found")
        return
    
    print(f"📋 Testing {len(template_files)} templates\n")
    
    for template_file in sorted(template_files):
        print(f"--- Template {template_file.name} ---")
        
        # Load template
        tmpl = cv.imread(str(template_file), cv.IMREAD_UNCHANGED)
        if tmpl is None:
            print("❌ Could not load template")
            continue
            
        # Handle alpha channel if present
        if tmpl.shape[2] == 4:  # Has alpha
            tmpl_bgr = tmpl[:,:,:3]
        else:
            tmpl_bgr = tmpl
            
        print(f"Template shape: {tmpl_bgr.shape}")
        
        # Scale template to match what bot uses (0.5 scale)
        scale = 0.5
        new_width = int(tmpl_bgr.shape[1] * scale)
        new_height = int(tmpl_bgr.shape[0] * scale)
        tmpl_scaled = cv.resize(tmpl_bgr, (new_width, new_height), interpolation=cv.INTER_CUBIC)
        
        # Test 1: Raw BGR matching
        try:
            res_raw = cv.matchTemplate(live_board, tmpl_scaled, cv.TM_CCOEFF_NORMED)
            raw_max = np.max(res_raw) if res_raw.size > 0 else 0
        except Exception as e:
            print(f"❌ Raw BGR matching failed: {e}")
            raw_max = 0
        
        # Test 2: Grayscale-only matching
        try:
            live_gray = cv.cvtColor(live_board, cv.COLOR_BGR2GRAY)
            tmpl_gray = cv.cvtColor(tmpl_scaled, cv.COLOR_BGR2GRAY)
            res_gray = cv.matchTemplate(live_gray, tmpl_gray, cv.TM_CCOEFF_NORMED)
            gray_max = np.max(res_gray) if res_gray.size > 0 else 0
        except Exception as e:
            print(f"❌ Grayscale matching failed: {e}")
            gray_max = 0
        
        # Results
        difference = gray_max - raw_max
        print(f"Raw BGR:    {raw_max:.3f}")
        print(f"Grayscale:  {gray_max:.3f}")
        print(f"Difference: {difference:.3f} ({'SIGNIFICANT' if difference >= 0.20 else 'minor'})")
        
        # Test 3: Count matches at different thresholds
        print("Match counts at different thresholds:")
        for threshold in [0.5, 0.6, 0.7, 0.8]:
            raw_matches = len(np.where(res_raw >= threshold)[0]) if res_raw.size > 0 else 0
            gray_matches = len(np.where(res_gray >= threshold)[0]) if res_gray.size > 0 else 0
            print(f"  @ {threshold:.1f}: Raw={raw_matches}, Gray={gray_matches}")
        
        print()
    
    # Overall conclusion
    print("🔍 DIAGNOSIS:")
    print("If grayscale consistently scores ≥0.20 higher than raw BGR:")
    print("  → COLOR SPACE MISMATCH confirmed")
    print("  → Solution: Switch bot to grayscale matching")
    print("If scores are similar:")
    print("  → Color space is not the issue")
    print("  → Problem likely in template quality or scaling")

if __name__ == "__main__":
    test_color_vs_grayscale() 