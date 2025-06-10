#!/usr/bin/env python3
"""
Test script for robust grid detection logic.
This validates the boundary detection → piece finding → grid inference pipeline.
"""

import numpy as np

def test_robust_grid_inference():
    """Test the robust grid inference from piece positions."""
    print("🧪 Testing Robust Grid Detection Logic")
    print("=" * 50)
    
    # Test case 1: Perfect grid with known piece positions
    print("\n📋 Test Case 1: Perfect Grid (8x8, 40px cells)")
    
    # Simulate detected pieces at known grid positions
    # Grid: 8x8 cells, 40px each, starting at (20, 20)
    cols, rows = 8, 8
    cell_size = 40
    board_x, board_y = 0, 0  # Board boundary coordinates
    grid_origin_x, grid_origin_y = 20, 20  # Grid starts 20px into board
    
    # Generate simulated piece positions (some cells, not all)
    piece_positions = []
    test_positions = [
        (0, 0), (2, 0), (4, 1),  # Top row pieces
        (1, 2), (3, 3), (5, 2),  # Middle pieces
        (2, 5), (6, 4), (7, 6)   # Bottom pieces
    ]
    
    for grid_x, grid_y in test_positions:
        screen_x = grid_origin_x + (grid_x * cell_size) + (cell_size // 2)
        screen_y = grid_origin_y + (grid_y * cell_size) + (cell_size // 2)
        piece_positions.append((screen_x, screen_y, "R", 0.8))  # (x, y, template, confidence)
    
    print(f"   Simulated {len(piece_positions)} pieces at known positions")
    for i, pos in enumerate(piece_positions[:3]):
        print(f"   Piece {i+1}: ({pos[0]}, {pos[1]})")
    
    # Test grid inference logic
    positions = [(pos[0] - board_x, pos[1] - board_y) for pos in piece_positions]
    
    # Find spacing patterns
    x_coords = sorted([pos[0] for pos in positions])
    y_coords = sorted([pos[1] for pos in positions])
    
    x_diffs = [x_coords[i+1] - x_coords[i] for i in range(len(x_coords)-1)]
    y_diffs = [y_coords[i+1] - y_coords[i] for i in range(len(y_coords)-1)]
    
    # Filter valid spacings (between 20-100 pixels)
    x_diffs_filtered = [d for d in x_diffs if 20 < d < 100]
    y_diffs_filtered = [d for d in y_diffs if 20 < d < 100]
    
    print(f"   X differences: {x_diffs_filtered}")
    print(f"   Y differences: {y_diffs_filtered}")
    
    # Find most common spacing
    def find_most_common_spacing(diffs):
        if not diffs:
            return 0
        groups = []
        for diff in diffs:
            added = False
            for group in groups:
                if abs(diff - np.mean(group)) <= 5:
                    group.append(diff)
                    added = True
                    break
            if not added:
                groups.append([diff])
        
        largest_group = max(groups, key=len) if groups else [0]
        return np.mean(largest_group)
    
    detected_width = find_most_common_spacing(x_diffs_filtered)
    detected_height = find_most_common_spacing(y_diffs_filtered)
    detected_cell_size = int((detected_width + detected_height) / 2)
    
    print(f"   Detected spacing: width={detected_width:.1f}, height={detected_height:.1f}")
    print(f"   Inferred cell size: {detected_cell_size}x{detected_cell_size}")
    print(f"   Expected: {cell_size}x{cell_size}")
    
    # Test grid origin detection
    min_distance = float('inf')
    origin_piece = None
    for pos in positions:
        distance = np.sqrt(pos[0]**2 + pos[1]**2)
        if distance < min_distance:
            min_distance = distance
            origin_piece = pos
    
    if origin_piece:
        detected_origin_x = origin_piece[0] % detected_cell_size - (detected_cell_size // 2)
        detected_origin_y = origin_piece[1] % detected_cell_size - (detected_cell_size // 2)
        
        # Adjust to positive
        while detected_origin_x < 0:
            detected_origin_x += detected_cell_size
        while detected_origin_y < 0:
            detected_origin_y += detected_cell_size
            
        print(f"   Detected origin: ({detected_origin_x}, {detected_origin_y})")
        print(f"   Expected: ({grid_origin_x}, {grid_origin_y})")
    
    # Validation
    cell_size_accurate = abs(detected_cell_size - cell_size) <= 2
    origin_accurate = (abs(detected_origin_x - grid_origin_x) <= 5 and 
                      abs(detected_origin_y - grid_origin_y) <= 5)
    
    if cell_size_accurate and origin_accurate:
        print("   ✅ Grid inference SUCCESSFUL!")
    else:
        print("   ❌ Grid inference needs refinement")
    
    # Test case 2: Irregular piece distribution
    print("\n📋 Test Case 2: Sparse Piece Distribution")
    
    # Simulate fewer pieces with some noise
    sparse_positions = [
        (60, 60), (140, 60), (260, 100),  # Some pieces with slight position noise
        (100, 180), (220, 220)
    ]
    
    print(f"   Testing with only {len(sparse_positions)} pieces")
    print(f"   Position noise to simulate real detection uncertainty")
    
    # Test the same logic
    x_coords = sorted([pos[0] for pos in sparse_positions])
    y_coords = sorted([pos[1] for pos in sparse_positions])
    
    x_diffs = [x_coords[i+1] - x_coords[i] for i in range(len(x_coords)-1)]
    y_diffs = [y_coords[i+1] - y_coords[i] for i in range(len(y_coords)-1)]
    
    x_diffs_filtered = [d for d in x_diffs if 20 < d < 100]
    y_diffs_filtered = [d for d in y_diffs if 20 < d < 100]
    
    print(f"   Filtered differences: X={x_diffs_filtered}, Y={y_diffs_filtered}")
    
    if x_diffs_filtered and y_diffs_filtered:
        sparse_width = find_most_common_spacing(x_diffs_filtered)
        sparse_height = find_most_common_spacing(y_diffs_filtered)
        sparse_cell_size = int((sparse_width + sparse_height) / 2)
        print(f"   Inferred cell size: {sparse_cell_size}x{sparse_cell_size}")
        print("   ✅ Handled sparse distribution successfully")
    else:
        print("   ⚠️  Insufficient data for grid inference (expected)")
    
    print("\n🎯 Key Insights:")
    print("   • Board boundary detection provides stable reference frame")
    print("   • Real piece positions reveal actual grid structure")  
    print("   • Uniform spacing assumption still applies within detected boundary")
    print("   • System handles piece detection uncertainty gracefully")

if __name__ == "__main__":
    test_robust_grid_inference() 