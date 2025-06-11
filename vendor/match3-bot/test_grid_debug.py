#!/usr/bin/env python3
"""
Quick test script to check anchor-based grid detection and generate debug images
"""

import cv2 as cv
import numpy as np
import pyautogui
from pathlib import Path

# Import functions from the main bot
import sys
sys.path.append('.')
from Python_match_3_bot_test import (
    load_templates, calibrate_grid_from_pieces, validate_anchor_based_grid_detection,
    create_multi_scale_edge_template, find_optimal_template_scale, find_optimal_thresholds
)

def load_saved_coordinates():
    """Load previously saved coordinates."""
    coords_file = Path("../../data/templates/royal-match/coordinates.txt")
    
    if not coords_file.exists():
        print("❌ No saved coordinates found")
        return None
        
    try:
        with open(coords_file, 'r') as f:
            coords = f.read().strip().split(',')
            x1, y1, x2, y2 = int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])
            print(f"📂 Loaded saved coordinates: ({x1}, {y1}) to ({x2}, {y2})")
            return x1, y1, x2, y2
    except (ValueError, FileNotFoundError) as e:
        print(f"⚠️  Could not load saved coordinates: {e}")
        return None

def main():
    print("🔍 Testing Anchor-Based Grid Detection")
    print("=" * 50)
    
    # Load templates
    available_templates = load_templates()
    if not available_templates:
        print("❌ Template loading failed!")
        return
    
    # Load saved coordinates
    coords = load_saved_coordinates()
    if coords is None:
        print("❌ No coordinates available for testing")
        return
        
    gridX1, gridY1, gridX2, gridY2 = coords
    
    # Take screenshot
    print("📸 Taking screenshot...")
    width = gridX2 - gridX1
    height = gridY2 - gridY1
    final = gridX1, gridY1, width, height
    im = pyautogui.screenshot(region=(final))
    board_image = cv.cvtColor(np.array(im), cv.COLOR_RGB2BGR)
    
    print(f"Board dimensions: {board_image.shape[1]}x{board_image.shape[0]}")
    
    # Test anchor-based grid detection
    print("\n🎯 Testing Anchor-Based Grid Detection...")
    result = calibrate_grid_from_pieces(board_image, available_templates)
    
    if result[0] is not None:
        cell_size, grid_origin_x, grid_origin_y, detected_cols, detected_rows, anchor_pieces = result
        
        print(f"✅ ANCHOR-BASED DETECTION SUCCESS!")
        print(f"   Detected grid: {detected_cols}x{detected_rows} = {detected_cols*detected_rows} cells")
        print(f"   Cell size: {cell_size}x{cell_size}")
        print(f"   Grid origin: ({grid_origin_x}, {grid_origin_y})")
        print(f"   Found {len(anchor_pieces)} anchor pieces")
        
        # Generate debug image
        validate_anchor_based_grid_detection(
            board_image, cell_size, grid_origin_x, grid_origin_y, anchor_pieces
        )
        print("📸 Debug image saved as: ../../debug_anchor_based_grid.png")
        
    else:
        print("❌ ANCHOR-BASED DETECTION FAILED")
        print("   Falling back to manual grid overlay...")
        
        # Create fallback debug image showing 8x10 grid
        debug_image = board_image.copy()
        board_height, board_width = board_image.shape[:2]
        
        # Fallback grid calculation
        cols, rows = 8, 10
        cellW = board_width // cols
        cellH = board_height // rows
        cell_size = min(cellW, cellH)
        
        print(f"   Fallback grid: {cols}x{rows}, cell size: {cell_size}x{cell_size}")
        
        # Draw fallback grid
        for i in range(cols + 1):
            x = i * cell_size
            if x < board_width:
                cv.line(debug_image, (x, 0), (x, board_height), (0, 0, 255), 2)  # Red lines
        
        for j in range(rows + 1):
            y = j * cell_size
            if y < board_height:
                cv.line(debug_image, (0, y), (board_width, y), (0, 0, 255), 2)  # Red lines
        
        # Draw cell centers
        for i in range(cols):
            for j in range(rows):
                center_x = i * cell_size + cell_size // 2
                center_y = j * cell_size + cell_size // 2
                if center_x < board_width and center_y < board_height:
                    cv.circle(debug_image, (center_x, center_y), 3, (0, 0, 255), -1)  # Red dots
                    cv.putText(debug_image, f"{i},{j}", (center_x-10, center_y-10), cv.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
        cv.imwrite("../../debug_fallback_grid.png", debug_image)
        print("📸 Fallback debug image saved as: ../../debug_fallback_grid.png")
    
    print("\n✅ Grid detection test complete!")

if __name__ == "__main__":
    main() 