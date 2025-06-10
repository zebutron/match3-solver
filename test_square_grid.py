#!/usr/bin/env python3
"""
Test script for square grid coordinate mapping logic.
This validates that our grid detection produces sensible coordinates.
"""

def test_square_grid_coordinates():
    """Test the square grid coordinate calculation logic."""
    print("🧪 Testing Square Grid Coordinate Mapping")
    print("=" * 50)
    
    # Test parameters
    cols = 8
    rows = 8
    
    # Test case 1: Perfect square board
    print("\n📋 Test Case 1: Perfect Square Board (640x640)")
    board_width, board_height = 640, 640
    
    # Calculate square cell size
    initial_cell_w = board_width // cols  # 80
    initial_cell_h = board_height // rows  # 80
    cell_size = min(initial_cell_w, initial_cell_h)  # 80
    
    # Calculate offsets (should be 0 for perfect square)
    required_width = cell_size * cols  # 640
    required_height = cell_size * rows  # 640  
    offset_x = (board_width - required_width) // 2  # 0
    offset_y = (board_height - required_height) // 2  # 0
    
    print(f"   Initial estimates: cellW={initial_cell_w}, cellH={initial_cell_h}")
    print(f"   Square cell size: {cell_size}x{cell_size}")
    print(f"   Grid offsets: ({offset_x}, {offset_y})")
    
    # Calculate cell centers
    cell_centers_x = []
    cell_centers_y = []
    for i in range(cols):
        center_x = offset_x + (cell_size * i) + (cell_size // 2)
        cell_centers_x.append(center_x)
    for j in range(rows):
        center_y = offset_y + (cell_size * j) + (cell_size // 2)
        cell_centers_y.append(center_y)
    
    print(f"   Cell centers X: {cell_centers_x}")
    print(f"   Cell centers Y: {cell_centers_y}")
    print(f"   ✅ Expected: centers at 40, 120, 200, 280, 360, 440, 520, 600")
    
    # Test case 2: Rectangular board (common issue)
    print("\n📋 Test Case 2: Rectangular Board (800x600)")
    board_width, board_height = 800, 600
    
    initial_cell_w = board_width // cols  # 100
    initial_cell_h = board_height // rows  # 75
    cell_size = min(initial_cell_w, initial_cell_h)  # 75 (use smaller to fit)
    
    required_width = cell_size * cols  # 600
    required_height = cell_size * rows  # 600
    offset_x = (board_width - required_width) // 2  # 100
    offset_y = (board_height - required_height) // 2  # 0
    
    print(f"   Initial estimates: cellW={initial_cell_w}, cellH={initial_cell_h}")
    print(f"   Square cell size: {cell_size}x{cell_size} (enforced square)")
    print(f"   Grid offsets: ({offset_x}, {offset_y}) (centers grid in capture area)")
    
    # This should produce a centered square grid within the rectangular capture
    cell_centers_x = []
    cell_centers_y = []
    for i in range(cols):
        center_x = offset_x + (cell_size * i) + (cell_size // 2)
        cell_centers_x.append(center_x)
    for j in range(rows):
        center_y = offset_y + (cell_size * j) + (cell_size // 2)
        cell_centers_y.append(center_y)
    
    print(f"   Cell centers X: {cell_centers_x[:3]}...{cell_centers_x[-3:]}")
    print(f"   Cell centers Y: {cell_centers_y[:3]}...{cell_centers_y[-3:]}")
    
    # Validate grid fits within board
    max_x = max(cell_centers_x) + (cell_size // 2)
    max_y = max(cell_centers_y) + (cell_size // 2)
    print(f"   Grid bounds: (0,0) to ({max_x},{max_y}) within ({board_width},{board_height})")
    
    if max_x <= board_width and max_y <= board_height:
        print("   ✅ Grid fits within board boundaries")
    else:
        print("   ❌ Grid exceeds board boundaries!")
    
    print("\n🎯 Key Insights:")
    print("   • Square cells enforce match-3 game constraints")
    print("   • Offsets center the grid within captured area")
    print("   • Smaller dimension determines cell size to ensure fit")
    print("   • This should eliminate row/column misalignment issues")

if __name__ == "__main__":
    test_square_grid_coordinates() 