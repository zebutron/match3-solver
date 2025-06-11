#!/usr/bin/env python3
"""
Enhanced Match3-Bot for Royal Match (QoL Improved) - EDGE DETECTION VERSION

Edge detection based template matching for robust piece detection:
- Uses edge detection to focus on piece shapes rather than colors
- Robust against obstacles and color variations
- Preserves dynamic threshold optimization
- Interactive coordinate setup and all existing UX improvements
- Now uses COLOR-NAMED templates (R, B, G, O) for better debugging

Usage:
    python3 Python_match_3_bot_test.py
    
Controls:
- Press Ctrl+C in terminal to stop
- Move mouse to top-left corner for emergency stop
"""

import cv2 as cv
import numpy as np
#from matplotlib import pyplot as plt
import pyautogui
import time #sleep()
import sys
from pathlib import Path
cols = 8   # Default, will be updated by grid detection
rows = 10  # Default, will be updated by grid detection
MyFilename = "./screenshots/dupa solo.png"

# Updated color-based preferences - using template names instead of numbers
BlockIDsPreference = ["R", "G", "B", "O"]  # Red, Green, Blue, Orange

debug = False

# Safety settings
pyautogui.FAILSAFE = True  # Move to corner to stop
pyautogui.PAUSE = 0.1

# Board state tracking for stuck move detection
previous_board_state = None
stuck_move_count = 0
MAX_STUCK_ATTEMPTS = 5
failed_moves_history = []  # Track moves that have failed recently
last_failed_move = None  # Track last failed move for distance-based selection

# Template scaling
optimal_template_scale = 1.0

# Edge detection globals
edge_processed_board = None

# Grid calibration globals
true_cell_size = 0
grid_offset_x = 0
grid_offset_y = 0
systematic_debug_created = False  # Track if we've created the systematic debug image

# Board state array - will be reinitialized based on detected grid dimensions
mainArray = []
for i in range(rows):
    row = []
    for j in range(cols):
        row.append("")
    mainArray.append(row)

def enhance_color_contrast(image):
    """
    Enhance hue differences while flattening brightness variations.
    This maximizes color-to-color boundaries that match-3 games are designed around.
    Same function as template processor.
    """
    # Handle both color and grayscale images
    if len(image.shape) == 2:
        # Image is grayscale, convert to color first
        image = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
    elif len(image.shape) == 3 and image.shape[2] == 1:
        # Single channel but 3D array
        image = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
    
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
    Same function as template processor.
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

def create_edge_template(image, blur_kernel=5, canny_low=50, canny_high=150, dilate_iterations=1):
    """
    Create an edge-based template using hue-focused edge detection.
    Same function as template processor.
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
    Same function as template processor.
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

def process_board_for_color_edge_matching(board_image):
    """
    Convert the live board to hue-focused edge format matching the templates.
    """
    global edge_processed_board
    print("🎨 Processing board with hue-focused edge detection...")
    edge_processed_board = create_multi_scale_edge_template(board_image)
    
    # Save debug image to see the hue-focused edge detection results
    debug_path = "../../hue_edge_board_capture.png"
    cv.imwrite(debug_path, edge_processed_board)
    print(f"📸 Debug: Hue-focused edge-processed board saved to {debug_path}")
    
    return edge_processed_board

def get_board_state_hash():
    """Create a hashable representation of the current board state."""
    return tuple(tuple(row) for row in mainArray)

def board_states_equal(state1, state2):
    """Compare two board states for equality."""
    if state1 is None or state2 is None:
        return False
    return state1 == state2

def calculate_move_distance(move1, move2):
    """Calculate Manhattan distance between two moves."""
    if not move1 or not move2:
        return float('inf')
    x1, y1 = move1[0], move1[1]
    x2, y2 = move2[0], move2[1]
    return abs(x1 - x2) + abs(y1 - y2)

def handle_stuck_move(attempted_move):
    """Handle situation where board state didn't change after a move."""
    global stuck_move_count, LegalMoves, failed_moves_history, last_failed_move
    
    stuck_move_count += 1
    print(f"⚠️  Board state unchanged after move #{stuck_move_count}")
    print(f"   Attempted move: {attempted_move}")
    
    # Add the failed move to history to avoid retrying it
    if attempted_move:
        move_signature = (attempted_move[0], attempted_move[1], attempted_move[2], attempted_move[3])  # x, y, direction, size
        # Check for duplicates before adding
        if move_signature not in failed_moves_history:
            failed_moves_history.append(move_signature)
            print(f"   🚫 Added move to failed history: {move_signature}")
            
            # Track the last failed move for distance-based selection
            last_failed_move = attempted_move
            print(f"   📍 Last failed move set to: {attempted_move[0:2]} for distance-based selection")
        else:
            print(f"   ⚠️  Move already in failed history: {move_signature}")
        
        print(f"   📝 Failed moves history size: {len(failed_moves_history)}")
        
        # Keep only recent failures (last 20 moves) to prevent history from growing too large
        if len(failed_moves_history) > 20:
            failed_moves_history.pop(0)
            print(f"   🧹 Cleaned old failed moves, history size: {len(failed_moves_history)}")
    
    if stuck_move_count >= MAX_STUCK_ATTEMPTS:
        print(f"❌ Stuck after {MAX_STUCK_ATTEMPTS} failed moves, clearing move list and failed history")
        LegalMoves.clear()
        failed_moves_history.clear()  # Also clear failed history when completely stuck
        stuck_move_count = 0
        last_failed_move = None
        return True  # Force skip this cycle
    
    # Remove the failed move from current legal moves list
    if attempted_move in LegalMoves:
        LegalMoves.remove(attempted_move)
        print(f"   ✂️  Removed failed move from current list, {len(LegalMoves)} moves remaining")
    
    return False  # Continue trying other moves

def printMainArray():
    print("📋 Current board:")
    for i in range(rows):
        row_str = "   "
        for j in range(cols):
            if mainArray[i][j] == "":
                row_str += ". "
            else:
                row_str += f"{mainArray[i][j]} "
        print(row_str)

def getGridLocation(game_name="royal-match"):
    """Interactive coordinate setup with multiple options."""
    print("🎯 Setting up game board coordinates")
    print("=" * 50)
    print(f"📱 Make sure {game_name.upper()} is visible and positioned on your screen")
    print()
    print("Choose coordinate input method:")
    print("  1. Click on corners (easiest)")
    print("  2. Enter coordinates manually")
    print("  3. Use saved coordinates (if available)")
    print()
    
    choice = input("Choose method (1-3): ").strip()
    
    if choice == "1":
        return _setup_by_clicking(game_name)
    elif choice == "2":
        return _setup_manually(game_name)
    elif choice == "3":
        coords = _load_saved_coordinates(game_name)
        if coords:
            return coords
        else:
            print("No saved coordinates found, using manual entry...")
            return _setup_manually(game_name)
    else:
        print("Invalid choice, using click method...")
        return _setup_by_clicking(game_name)

def _setup_by_clicking(game_name="royal-match"):
    """Setup by clicking on screen corners."""
    print("\n🖱️  Click Setup")
    print("Instructions:")
    print("1. Position your mouse over the TOP-LEFT corner of the game board")
    print("2. Press ENTER")
    print("3. Position your mouse over the BOTTOM-RIGHT corner")
    print("4. Press ENTER")
    print()
    print("⚠️  Important: Point to the actual game tiles, not the board border!")
    print("    The bot needs to know where the playable area is.")
    print()
    
    input("Position mouse over TOP-LEFT corner of game board, then press ENTER...")
    pos1 = pyautogui.position()
    gridX1, gridY1 = pos1.x, pos1.y
    print(f"✅ Top-left: ({gridX1}, {gridY1})")
    
    input("Position mouse over BOTTOM-RIGHT corner of game board, then press ENTER...")
    pos2 = pyautogui.position()
    gridX2, gridY2 = pos2.x, pos2.y
    print(f"✅ Bottom-right: ({gridX2}, {gridY2})")
    
    # Validate and save
    if _validate_coordinates(gridX1, gridY1, gridX2, gridY2):
        _save_coordinates(gridX1, gridY1, gridX2, gridY2, game_name)
        return gridX1, gridY1, gridX2, gridY2
    else:
        print("❌ Invalid coordinates, please try again")
        return _setup_manually(game_name)

def _setup_manually(game_name="royal-match"):
    """Manual coordinate entry."""
    print("\n⌨️  Manual Coordinate Entry")
    print("Tip: Use screenshot tools to find exact coordinates")
    print()
    
    try:
        gridX1 = int(input("X coordinate of top-left corner: "))
        gridY1 = int(input("Y coordinate of top-left corner: "))
        gridX2 = int(input("X coordinate of bottom-right corner: "))
        gridY2 = int(input("Y coordinate of bottom-right corner: "))
        
        if _validate_coordinates(gridX1, gridY1, gridX2, gridY2):
            _save_coordinates(gridX1, gridY1, gridX2, gridY2, game_name)
            return gridX1, gridY1, gridX2, gridY2
        else:
            print("❌ Invalid coordinates")
            return None, None, None, None
    except ValueError:
        print("❌ Please enter valid numbers")
        return None, None, None, None

def _validate_coordinates(x1, y1, x2, y2):
    """Validate that coordinates make sense."""
    width = x2 - x1
    height = y2 - y1
    
    if width < 200 or height < 200:
        print(f"❌ Board too small: {width}x{height} (minimum 200x200)")
        return False
        
    if width > 2000 or height > 2000:
        print(f"❌ Board too large: {width}x{height} (maximum 2000x2000)")
        return False
        
    print(f"✅ Board size: {width}x{height} pixels")
    return True

def _save_coordinates(x1, y1, x2, y2, game_name="royal-match"):
    """Save coordinates for future use."""
    coords_file = Path(f"../../data/templates/{game_name}/coordinates.txt")
    coords_file.parent.mkdir(parents=True, exist_ok=True)
    with open(coords_file, 'w') as f:
        f.write(f"{x1},{y1},{x2},{y2}")
    print(f"💾 Coordinates saved to {coords_file}")

def _load_saved_coordinates(game_name="royal-match"):
    """Load previously saved coordinates."""
    coords_file = Path(f"../../data/templates/{game_name}/coordinates.txt")
    try:
        with open(coords_file, 'r') as f:
            coords = f.read().strip().split(',')
            x1, y1, x2, y2 = map(int, coords)
            print(f"📂 Loaded saved coordinates: ({x1}, {y1}) to ({x2}, {y2})")
            return x1, y1, x2, y2
    except FileNotFoundError:
        return None
    except Exception as e:
        print(f"⚠️  Could not load saved coordinates: {e}")
        return None

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def find_optimal_template_scale(template_path, grid_image):
    """Find the optimal scale for templates by testing different sizes."""
    template = cv.imread(template_path, cv.IMREAD_GRAYSCALE)
    if template is None:
        return 1.0, 0.0
    
    # Process grid image with same edge detection as templates
    edge_processed_grid = create_multi_scale_edge_template(grid_image)
    
    # Calculate expected template size based on grid
    grid_height, grid_width = grid_image.shape[:2]
    cellW = grid_width // cols
    cellH = grid_height // rows
    
    print(f"   Scale detection: cell size {cellW}x{cellH}, template size {template.shape[1]}x{template.shape[0]}")
    
    # Target scale should make templates roughly match cell size
    # Templates are 64x64, so target scale should be cellsize/64
    target_scale_w = cellW / 64.0
    target_scale_h = cellH / 64.0
    target_scale = (target_scale_w + target_scale_h) / 2.0
    
    print(f"   Target scale estimate: {target_scale:.3f} (based on cell/template ratio)")
    
    # Test scales around the target scale
    scale_min = max(0.3, target_scale * 0.6)  # Don't go too small
    scale_max = min(2.0, target_scale * 1.4)   # Don't go too large
    scale_step = 0.05
    
    print(f"   Testing scales: {scale_min:.2f} to {scale_max:.2f}")
    
    # Test different scales
    best_scale = target_scale
    best_confidence = 0.0
    
    for scale in np.arange(scale_min, scale_max, scale_step):
        # Resize template
        new_width = int(template.shape[1] * scale)
        new_height = int(template.shape[0] * scale)
        
        if new_width < 8 or new_height < 8 or new_width > 200 or new_height > 200:
            continue
            
        scaled_template = cv.resize(template, (new_width, new_height), interpolation=cv.INTER_CUBIC)
        
        # Test template matching against edge-processed grid
        res = cv.matchTemplate(edge_processed_grid, scaled_template, cv.TM_CCOEFF_NORMED)
        max_confidence = np.max(res) if res.size > 0 else 0
        
        if max_confidence > best_confidence:
            best_confidence = max_confidence
            best_scale = scale
            
        # Debug some scale tests
        if abs(scale - target_scale) < 0.1:  # Show results near target
            print(f"   Scale {scale:.2f}: confidence {max_confidence:.3f}, template size {new_width}x{new_height}")
    
    final_width = int(template.shape[1] * best_scale)
    final_height = int(template.shape[0] * best_scale)
    print(f"   Best scale: {best_scale:.3f} → template size {final_width}x{final_height}")
    
    return best_scale, best_confidence

def find_optimal_thresholds(available_templates, grid_image, target_pieces_min=40, target_pieces_max=60):
    """
    Find optimal thresholds for each template to achieve target piece count using PRECISE EDGE DETECTION.
    """
    print("🎯 Finding optimal thresholds for precise edge-based templates...")
    
    # Convert board to precise edge format
    edge_board = create_multi_precision_edge_template(grid_image)
    
    # Calculate cell dimensions
    global dim
    dim = grid_image.shape
    cellW = dim[1]//cols  # Fixed: Use WIDTH for cell width
    cellH = dim[0]//rows  # Fixed: Use HEIGHT for cell height
    
    template_thresholds = {}
    threshold_test_range = np.arange(0.10, 0.60, 0.025)  # Lowered range for hue-focused matching
    
    for template_name, template_path in available_templates.items():
        print(f"\n--- Optimizing Template {template_name} ---")
        
        # Load template (should already be hue-focused edge-processed)
        template = cv.imread(str(template_path), cv.IMREAD_GRAYSCALE)
        if template is None:
            print(f"⚠️  Could not load template: {template_path}")
            continue
        
        # Scale template using optimal scale
        if optimal_template_scale > 0:
            new_width = int(template.shape[1] * optimal_template_scale)
            new_height = int(template.shape[0] * optimal_template_scale)
            template = cv.resize(template, (new_width, new_height), interpolation=cv.INTER_LANCZOS4)
        
        # Template matching against hue-focused edge-processed board
        res = cv.matchTemplate(edge_processed_board, template, cv.TM_CCOEFF_NORMED)
        
        best_threshold = 0.25  # Lower default for hue-focused edges
        best_count = 0
        
        print("Threshold optimization:")
        for threshold in threshold_test_range:
            matches = len(np.where(res >= threshold)[0])
            print(f"  @ {threshold:.3f}: {matches:3d} matches")
            
            # Target range for individual templates (8-80 matches per template) - adjusted for hue-focused
            if 8 <= matches <= 80:
                if best_threshold == 0.25 or abs(matches - 40) < abs(best_count - 40):
                    best_threshold = threshold
                    best_count = matches
        
        # FIXED: Enforce minimum threshold for over-matching templates
        if best_threshold == 0.25:  # No good threshold found
            # Check if template is over-matching at low thresholds
            low_threshold_matches = len(np.where(res >= 0.20)[0])
            if low_threshold_matches > 200:  # Over-matching template (like P)
                print(f"   ⚠️  Template {template_name} over-matches ({low_threshold_matches} at 0.20), using higher minimum")
                best_threshold = 0.35  # Use higher threshold for over-matching templates
            else:
                best_threshold = 0.20  # Lower than old system for normal templates
        
        template_thresholds[template_name] = best_threshold
        print(f"✅ Template {template_name}: optimal threshold = {best_threshold:.3f} ({best_count} matches)")
    
    # Test total piece count with optimized thresholds
    total_pieces = 0
    for template_name, threshold in template_thresholds.items():
        template_path = available_templates[template_name]
        template = cv.imread(str(template_path), cv.IMREAD_GRAYSCALE)
        if template is not None:
            if optimal_template_scale > 0:
                new_width = int(template.shape[1] * optimal_template_scale)
                new_height = int(template.shape[0] * optimal_template_scale)
                template = cv.resize(template, (new_width, new_height), interpolation=cv.INTER_LANCZOS4)
            res = cv.matchTemplate(edge_processed_board, template, cv.TM_CCOEFF_NORMED)
            matches = len(np.where(res >= threshold)[0])
            total_pieces += matches
    
    print(f"\n📊 PRECISE EDGE-BASED THRESHOLD OPTIMIZATION RESULTS:")
    print(f"Total pieces detected: {total_pieces}")
    print(f"Target range: {target_pieces_min}-{target_pieces_max}")
    
    if target_pieces_min <= total_pieces <= target_pieces_max:
        print("✅ EXCELLENT! Precise thresholds are perfectly optimized")
    elif total_pieces < target_pieces_min:
        print("⚠️  Low piece count - may need lower thresholds")
    else:
        print("⚠️  High piece count - may need higher thresholds")
    
    return template_thresholds

def calculate_anchor_based_grid_coordinates(grid_origin_x, grid_origin_y, cell_size):
    """
    Calculate grid coordinates using the anchor-based detection results.
    """
    global cols, rows  # Use the globally detected dimensions
    
    cell_centers_x = []
    cell_centers_y = []
    
    for i in range(cols):
        center_x = grid_origin_x + (cell_size * i) + (cell_size // 2)
        cell_centers_x.append(center_x)
    
    for j in range(rows):
        center_y = grid_origin_y + (cell_size * j) + (cell_size // 2)
        cell_centers_y.append(center_y)
    
    return cell_centers_x, cell_centers_y

def find_all_occurences_into_mainArray(filename,template_name,color=(0,0,255),custom_threshold=None):
    """
    Find template matches using EDGE DETECTION and ANCHOR-BASED GRID mapping.
    Now uses template_name (color letter) and enforces anchor-based grid constraints.
    """
    #assume template is smaller than single "block"/rectangle with desired object
    #modifies given image
    FullGridImage = FullGridImageOriginal #given this image  | full grid

    global dim
    dim = FullGridImage.shape
    
    # Use edge-processed board instead of grayscale
    global edge_processed_board
    if edge_processed_board is None:
        edge_processed_board = create_multi_scale_edge_template(FullGridImage)
    
    # NEW: Use ANCHOR-BASED grid coordinates when available, fallback to square grid calculation
    global true_cell_size, grid_offset_x, grid_offset_y
    
    if true_cell_size > 0:
        # Use anchor-based grid coordinates derived from high-confidence pieces
        tempArrayW, tempArrayH = calculate_anchor_based_grid_coordinates(grid_offset_x, grid_offset_y, true_cell_size)
        grid_status = "ANCHOR-BASED"
        cell_info = f"cell_size: {true_cell_size} (from high-confidence piece analysis)"
    else:
        print("⚠️  Grid not calibrated - using fallback square calculation")
        # Fallback to square grid calculation
        cellW = dim[1]//cols
        cellH = dim[0]//rows
        cell_size = min(cellW, cellH)  # Force square cells
        
        tempArrayW = []
        tempArrayH = []
        for i in range(cols):
            tempArrayW.append( (cell_size * i) + (cell_size//2) )
        for j in range(rows):
             tempArrayH.append( (cell_size * j) + (cell_size//2) )
        
        grid_status = "FALLBACK-SQUARE"
        cell_info = f"cell_size: {cell_size} (fallback calculation)"

    # Load template (should be edge-processed from our template processor)
    template = cv.imread(filename, cv.IMREAD_GRAYSCALE)  # Load as grayscale since templates are edge-processed
    if template is None:
        print(f"⚠️  Could not load template: {filename}")
        return
    
    # Use the globally determined optimal scale for this puzzle
    global optimal_template_scale
    if optimal_template_scale > 0:
        new_width = int(template.shape[1] * optimal_template_scale)
        new_height = int(template.shape[0] * optimal_template_scale)
        template = cv.resize(template, (new_width, new_height), interpolation=cv.INTER_CUBIC)
    
    w, h = template.shape[::-1]
    
    # Edge vs Edge matching instead of grayscale vs grayscale
    res = cv.matchTemplate(edge_processed_board, template, cv.TM_CCOEFF_NORMED)
    
    # Use custom threshold if provided, otherwise use lower default for hue-focused edge matching
    threshold = custom_threshold if custom_threshold is not None else 0.25  # Lowered from 0.5 to catch more matches
    loc = np.where( res >= threshold)
    
    # Count matches and get confidence scores
    match_count = len(loc[0])
    max_confidence = np.max(res) if res.size > 0 else 0
    template_size = f"{template.shape[1]}x{template.shape[0]}"
    
    # Show grid calibration status in output
    print(f"   Template {template_name}: {match_count} matches (max confidence: {max_confidence:.3f}, threshold: {threshold:.3f}) [size: {template_size}] {grid_status}")
    
    if match_count > 0:
        print(f"   🔍 ANCHOR-BASED grid mapping debug for {template_name}:")
        print(f"      Cell centers W: {tempArrayW[:3]}...{tempArrayW[-3:]} ({len(tempArrayW)} total)")
        print(f"      Cell centers H: {tempArrayH[:3]}...{tempArrayH[-3:]} ({len(tempArrayH)} total)")
        print(f"      {cell_info}, Grid offsets: ({grid_offset_x}, {grid_offset_y})")
    
    match_debug_count = 0
    successful_mappings = 0
    
    for pt in zip(*loc[::-1]):
        if(debug):
            cv.rectangle(FullGridImage, pt, (pt[0] + w, pt[1] + h), color, 2)
            cv.rectangle(FullGridImage, ((pt[0] + w//2)-1, (pt[1] + h//2)-1), ((pt[0] + w//2)+1, (pt[1] + h//2)+1), color, 2)
        
        #pt is top left point eg.(106, 5)
        match_center_x = pt[0] + w//2
        match_center_y = pt[1] + h//2
        nearestW = find_nearest(tempArrayW, match_center_x)
        nearestH = find_nearest(tempArrayH, match_center_y)
        
        # Debug first few matches with distance info
        if match_debug_count < 3 and match_count <= 50:  # Only debug if reasonable match count
            expected_x = tempArrayW[nearestW]
            expected_y = tempArrayH[nearestH]
            distance_x = abs(match_center_x - expected_x)
            distance_y = abs(match_center_y - expected_y)
            print(f"      Match {match_debug_count+1}: center({match_center_x},{match_center_y}) → grid({nearestW},{nearestH}) [distance: ({distance_x},{distance_y})]")
            match_debug_count += 1
        
        if(debug):
            print(match_center_x, match_center_y, "    |   ",nearestW,nearestH,"   ",template_name)
        
        # Only assign if cell is empty (avoid overlaps)
        if mainArray[nearestH][nearestW] == "":
            mainArray[nearestH][nearestW] = template_name  # Store color letter instead of number
            successful_mappings += 1
    
    if match_count > 0:
        print(f"      ✅ Successfully mapped {successful_mappings}/{match_count} matches to grid cells")
        
    if(debug):
        cv.imwrite('./screenshots/ress.png',FullGridImage)

def cell_exists_and_in_bounds(x, y):
    """Check if a cell position exists on the board and is within bounds."""
    global cell_exists_grid
    if 0 <= y < len(cell_exists_grid) and 0 <= x < len(cell_exists_grid[0]):
        return cell_exists_grid[y][x]
    return False

def check5moves(x,y):
    myBlock = mainArray[y][x]
    
    # Skip empty cells or non-existent cells
    if myBlock == "" or not cell_exists_and_in_bounds(x, y):
        return
    
    #check 1 
    # --X-- \/
    # OO-OO
    if(y+1<=rows-1 and x-2>=0 and x+2<=cols-1):
        if(cell_exists_and_in_bounds(x-2, y+1) and cell_exists_and_in_bounds(x-1, y+1) and 
           cell_exists_and_in_bounds(x+1, y+1) and cell_exists_and_in_bounds(x+2, y+1) and
           mainArray[y+1][x-2] == mainArray[y+1][x-1] == mainArray[y+1][x+1] == mainArray[y+1][x+2] == myBlock):
            print("legal 5 move down")
            LegalMoves.append((x,y,"down",5,myBlock))
    
    #check 2 
    # OO-OO /\
    # --X--
    if(y-1>=0 and x-2>=0 and x+2<=cols-1):
        if(cell_exists_and_in_bounds(x-2, y-1) and cell_exists_and_in_bounds(x-1, y-1) and 
           cell_exists_and_in_bounds(x+1, y-1) and cell_exists_and_in_bounds(x+2, y-1) and
           mainArray[y-1][x-2] == mainArray[y-1][x-1] == mainArray[y-1][x+1] == mainArray[y-1][x+2] == myBlock):
            print("legal 5 move up")
            LegalMoves.append((x,y,"up",5,myBlock))
    
    #check 3 
    # O---- <
    # O----
    # -X---
    # O----
    # O----
    if(x-1>=0 and y-2>=0 and y+2<=rows-1):
        if(cell_exists_and_in_bounds(x-1, y-1) and cell_exists_and_in_bounds(x-1, y-2) and 
           cell_exists_and_in_bounds(x-1, y+1) and cell_exists_and_in_bounds(x-1, y+2) and
           mainArray[y-1][x-1] == mainArray[y-2][x-1] == mainArray[y+1][x-1] == mainArray[y+2][x-1] == myBlock):
            print("legal 5 move left")
            LegalMoves.append((x,y,"left",5,myBlock))

    
    #check 4 
    # -O-- >
    # -O--
    # X---
    # -O--
    # -O--
    if(x+1<=cols-1 and y-2>=0 and y+2<=rows-1):
        if(cell_exists_and_in_bounds(x+1, y-1) and cell_exists_and_in_bounds(x+1, y-2) and 
           cell_exists_and_in_bounds(x+1, y+1) and cell_exists_and_in_bounds(x+1, y+2) and
           mainArray[y-1][x+1] == mainArray[y-2][x+1] == mainArray[y+1][x+1] == mainArray[y+2][x+1] == myBlock):
            print("legal 5 move right")
            LegalMoves.append((x,y,"right",5,myBlock))

def check4moves(x,y):
    myBlock = mainArray[y][x]
    
    # Skip empty cells or non-existent cells
    if myBlock == "" or not cell_exists_and_in_bounds(x, y):
        return
    
    #check 1 
    # --X- \/
    # OO-O
    if(y+1<=rows-1 and x-2>=0 and x+1<=cols-1):
        if(cell_exists_and_in_bounds(x-2, y+1) and cell_exists_and_in_bounds(x-1, y+1) and 
           cell_exists_and_in_bounds(x+1, y+1) and
           mainArray[y+1][x-2] == mainArray[y+1][x-1] == mainArray[y+1][x+1] == myBlock):
            print("legal 4 move down (left)")
            LegalMoves.append((x,y,"down",4,myBlock))
    #check 2 
    # -X-- \/
    # O-OO
    if(y+1<=rows-1 and x-1>=0 and x+2<=cols-1):
        if(cell_exists_and_in_bounds(x-1, y+1) and cell_exists_and_in_bounds(x+1, y+1) and 
           cell_exists_and_in_bounds(x+2, y+1) and
           mainArray[y+1][x-1] == mainArray[y+1][x+1] == mainArray[y+1][x+2] == myBlock):
            print("legal 4 move down (right)")
            LegalMoves.append((x,y,"down",4,myBlock))
    
    #check 3
    # OO-O /\
    # --X-
    if(y-1>=0 and x-2>=0 and x+1<=cols-1):
        if(cell_exists_and_in_bounds(x-2, y-1) and cell_exists_and_in_bounds(x-1, y-1) and 
           cell_exists_and_in_bounds(x+1, y-1) and
           mainArray[y-1][x-2] == mainArray[y-1][x-1] == mainArray[y-1][x+1] == myBlock):
            print("legal 4 move up (left)")
            LegalMoves.append((x,y,"up",4,myBlock))
    #check 4
    # O-OO /\
    # -X--
    if(y-1>=0 and x-1>=0 and x+2<=cols-1):
        if(cell_exists_and_in_bounds(x-1, y-1) and cell_exists_and_in_bounds(x+1, y-1) and 
           cell_exists_and_in_bounds(x+2, y-1) and
           mainArray[y-1][x-1] == mainArray[y-1][x+1] == mainArray[y-1][x+2] == myBlock):
            print("legal 4 move up (right)")
            LegalMoves.append((x,y,"up",4,myBlock))
    
    #check 5 
    # O---- <
    # -X---
    # O----
    # O----
    if(x-1>=0 and y-1>=0 and y+2<=rows-1):
        if(cell_exists_and_in_bounds(x-1, y-1) and cell_exists_and_in_bounds(x-1, y+1) and 
           cell_exists_and_in_bounds(x-1, y+2) and
           mainArray[y-1][x-1] == mainArray[y+1][x-1] == mainArray[y+2][x-1]== myBlock):
            print("legal 4 move left (down)")
            LegalMoves.append((x,y,"left",4,myBlock))
    #check 6 
    # O---- <
    # O----
    # -X---
    # O----
    if(x-1>=0 and y-2>=0 and y+1<=rows-1):
        if(cell_exists_and_in_bounds(x-1, y-1) and cell_exists_and_in_bounds(x-1, y-2) and 
           cell_exists_and_in_bounds(x-1, y+1) and
           mainArray[y-1][x-1] == mainArray[y-2][x-1] == mainArray[y+1][x-1]== myBlock):
            print("legal 4 move left (up)")
            LegalMoves.append((x,y,"left",4,myBlock))

    #check 7 
    # -O-- >
    # -O--
    # X---
    # -O--
    if(x+1<=cols-1 and y-2>=0 and y+1<=rows-1):
        if(cell_exists_and_in_bounds(x+1, y-1) and cell_exists_and_in_bounds(x+1, y-2) and 
           cell_exists_and_in_bounds(x+1, y+1) and
           mainArray[y-1][x+1] == mainArray[y-2][x+1] == mainArray[y+1][x+1]== myBlock):
            print("legal 4 move right (up)")
            LegalMoves.append((x,y,"right",4,myBlock))
    #check 8 
    # -O-- >
    # X---
    # -O--
    # -O--
    if(x+1<=cols-1 and y-1>=0 and y+2<=rows-1):
        if(cell_exists_and_in_bounds(x+1, y-1) and cell_exists_and_in_bounds(x+1, y+2) and 
           cell_exists_and_in_bounds(x+1, y+1) and
           mainArray[y-1][x+1] == mainArray[y+2][x+1] == mainArray[y+1][x+1]== myBlock):
            print("legal 4 move right (down)")
            LegalMoves.append((x,y,"right",4,myBlock))

def check3moves(x,y):
    myBlock = mainArray[y][x]
    
    # Skip empty cells or non-existent cells
    if myBlock == "" or not cell_exists_and_in_bounds(x, y):
        return
    
    #check 1
    # O-O /\
    # -X-
    if(y-1>=0 and x-1>=0 and x+1<=cols-1):
        if(cell_exists_and_in_bounds(x-1, y-1) and cell_exists_and_in_bounds(x+1, y-1) and
           mainArray[y-1][x-1] == mainArray[y-1][x+1] == myBlock):
            print("legal 3 move up")
            LegalMoves.append((x,y,"up",3,myBlock))

    #check 2
    # -X- \/
    # O-O
    if(y+1<=rows-1 and x-1>=0 and x+1<=cols-1):
        if(cell_exists_and_in_bounds(x-1, y+1) and cell_exists_and_in_bounds(x+1, y+1) and
           mainArray[y+1][x-1] == mainArray[y+1][x+1] == myBlock):
            print("legal 3 move down")
            LegalMoves.append((x,y,"down",3,myBlock))
            
    #check 3
    # O- <
    # -X
    # O-
    if(y+1<=rows-1 and y-1>=0 and x-1>=0):
        if(cell_exists_and_in_bounds(x-1, y+1) and cell_exists_and_in_bounds(x-1, y-1) and
           mainArray[y+1][x-1] == mainArray[y-1][x-1] == myBlock):
            print("legal 3 move left")
            LegalMoves.append((x,y,"left",3,myBlock))
    #check 4
    # -O >
    # X-
    # -O
    if(y+1<=rows-1 and y-1>=0 and x+1<cols-1):
        if(cell_exists_and_in_bounds(x+1, y+1) and cell_exists_and_in_bounds(x+1, y-1) and
           mainArray[y+1][x+1] == mainArray[y-1][x+1] == myBlock):
            print("legal 3 move right")
            LegalMoves.append((x,y,"right",3,myBlock))
    
    #check 5
    # O /\
    # O
    # -
    # X
    if(y-3>=0):
        if(mainArray[y-1][x] == mainArray[y-2][x] == myBlock):
            print("legal 3 move up (double)")
            LegalMoves.append((x,y,"up",3,myBlock))

    #check 6
    # X \/
    # -
    # O
    # O
    if(y+3<=rows-1):
        if(mainArray[y+2][x] == mainArray[y+3][x] == myBlock):
            print("legal 3 move down (double)")
            LegalMoves.append((x,y,"down",3,myBlock))

    #check 7
    # X-OO >
    if(x+3<=cols-1):
        if(mainArray[y][x+2] == mainArray[y][x+3] == myBlock):
            print("legal 3 move right (double)")
            LegalMoves.append((x,y,"right",3,myBlock))
    #check 8
    # OO-X <
    if(x-3>=0):
        if(mainArray[y][x-2] == mainArray[y][x-3] == myBlock):
            print("legal 3 move left (double)")
            LegalMoves.append((x,y,"left",3,myBlock))

    #check 9
    # X-- \/
    # -OO 
    if(y+1<=rows-1 and x+2<=cols-1):
        if(mainArray[y+1][x+1] == mainArray[y+1][x+2] == myBlock):
            print("legal 3 move down (double)(right)")
            LegalMoves.append((x,y,"down",3,myBlock))
    #check 10
    # --X \/
    # OO- 
    if(y+1<=rows-1 and x-2>=0):
        if(mainArray[y+1][x-1] == mainArray[y+1][x-2] == myBlock):
            print("legal 3 move down (double)(left)")
            LegalMoves.append((x,y,"down",3,myBlock))

    #check 11
    # OO- /\
    # --X 
    if(y-1>=0 and x-2>=0):
        if(mainArray[y-1][x-1] == mainArray[y-1][x-2] == myBlock):
            print("legal 3 move up (double)(left)")
            LegalMoves.append((x,y,"up",3,myBlock))
            
    #check 12
    # -OO /\
    # X-- 
    if(y-1>=0 and x+2<=cols-1):
        if(mainArray[y-1][x+1] == mainArray[y-1][x+2] == myBlock):
            print("legal 3 move up (double)(right)")
            LegalMoves.append((x,y,"up",3,myBlock))

    #check 13
    # X- >
    # -O 
    # -O
    if(y+2<=rows-1 and x+1<=cols-1):
        if(mainArray[y+1][x+1] == mainArray[y+2][x+1] == myBlock):
            print("legal 3 move right (double)(down)")
            LegalMoves.append((x,y,"right",3,myBlock))

    #check 14
    # -O >
    # -O 
    # X-
    if(y-2>=0 and x+1<=cols-1):
        if(mainArray[y-2][x+1] == mainArray[y-1][x+1] == myBlock):
            print("legal 3 move right (double)(up)")
            LegalMoves.append((x,y,"right",3,myBlock))

    #check 15
    # -X <
    # O- 
    # O-
    if(y+2<=rows-1 and x-1>=0):
        if(mainArray[y+2][x-1] == mainArray[y+1][x-1] == myBlock):
            print("legal 3 move left (double)(down)")
            LegalMoves.append((x,y,"left",3,myBlock))

    #check 16
    # O- <
    # O- 
    # -X
    if(y-2>=0 and x-1>=0):
        if(mainArray[y-2][x-1] == mainArray[y-1][x-1] == myBlock):
            print("legal 3 move left (double)(up)")
            LegalMoves.append((x,y,"left",3,myBlock))

def find_powerups_on_board(available_powerups, optimal_thresholds):
    """
    Find powerups on the board using the same edge detection techniques as pieces.
    Returns list of (x, y, powerup_name, confidence) tuples.
    """
    if not available_powerups:
        return []
    
    powerup_positions = []
    
    # Use same edge detection as pieces
    global edge_processed_board
    if edge_processed_board is None:
        return []
    
    # Debug colors for powerups (different from pieces)
    powerup_colors = [
        (255,0,255),    # Magenta
        (0,255,255),    # Cyan  
        (255,255,0),    # Yellow
        (255,128,0),    # Orange
        (128,255,128),  # Light Green
    ]
    
    print("⚡ Scanning for POWERUPS on board...")
    
    for i, (powerup_name, powerup_path) in enumerate(available_powerups.items()):
        color = powerup_colors[i % len(powerup_colors)]
        powerup_threshold = optimal_thresholds.get(powerup_name, 0.95)  # Extremely high threshold for powerups to eliminate false positives
        
        # Load powerup template
        template = cv.imread(str(powerup_path), cv.IMREAD_GRAYSCALE)
        if template is None:
            continue
        
        # Use same scaling as pieces
        global optimal_template_scale
        if optimal_template_scale > 0:
            new_width = int(template.shape[1] * optimal_template_scale)
            new_height = int(template.shape[0] * optimal_template_scale)
            template = cv.resize(template, (new_width, new_height), interpolation=cv.INTER_CUBIC)
        
        w, h = template.shape[::-1]
        
        # Template matching against edge-processed board (same as pieces)
        res = cv.matchTemplate(edge_processed_board, template, cv.TM_CCOEFF_NORMED)
        
        # Find matches above threshold
        loc = np.where(res >= powerup_threshold)
        match_count = len(loc[0])
        max_confidence = np.max(res) if res.size > 0 else 0
        
        print(f"   ⚡ Powerup {powerup_name}: {match_count} matches (max confidence: {max_confidence:.3f}, threshold: {powerup_threshold:.3f})")
        
        # Use same grid mapping as pieces
        global true_cell_size, grid_offset_x, grid_offset_y, cell_exists_grid
        
        if true_cell_size > 0:
            tempArrayW, tempArrayH = calculate_anchor_based_grid_coordinates(grid_offset_x, grid_offset_y, true_cell_size)
        else:
            # Fallback grid calculation
            global dim
            cellW = dim[1]//cols
            cellH = dim[0]//rows
            cell_size = min(cellW, cellH)
            
            tempArrayW = []
            tempArrayH = []
            for i in range(cols):
                tempArrayW.append( (cell_size * i) + (cell_size//2) )
            for j in range(rows):
                 tempArrayH.append( (cell_size * j) + (cell_size//2) )
        
        # Process matches
        for pt in zip(*loc[::-1]):
            match_center_x = pt[0] + w//2
            match_center_y = pt[1] + h//2
            nearestW = find_nearest(tempArrayW, match_center_x)
            nearestH = find_nearest(tempArrayH, match_center_y)
            
            # Only add if cell exists and confidence is good
            if (nearestH < len(cell_exists_grid) and nearestW < len(cell_exists_grid[0]) and
                cell_exists_grid[nearestH][nearestW]):
                confidence = res[pt[1], pt[0]]
                powerup_positions.append((nearestW, nearestH, powerup_name, confidence))
    
    if powerup_positions:
        # Sort by priority (highest number first) then by confidence
        powerup_positions.sort(key=lambda x: (int(x[2]) if x[2].isdigit() else 0, x[3]), reverse=True)
        print(f"   ✅ Found {len(powerup_positions)} powerups on board (sorted by priority)")
        for x, y, name, conf in powerup_positions[:5]:  # Show top 5
            priority = int(name) if name.isdigit() else 0
            print(f"      ⚡ Powerup {name} at ({x},{y}) - Priority {priority}, Confidence {conf:.3f}")
    
    return powerup_positions

def create_powerup_moves(powerup_positions):
    """
    Create swipe moves for detected powerups.
    Powerups can be swiped in any of the 4 directions.
    """
    global LegalMoves
    powerup_moves = []
    
    if not powerup_positions:
        return
    
    print("⚡ Creating POWERUP MOVES...")
    
    for x, y, powerup_name, confidence in powerup_positions:
        # Check all 4 directions for valid swipes
        directions = ['up', 'down', 'left', 'right']
        
        for direction in directions:
            # Check if target cell exists and is in bounds
            if direction == 'up' and y > 0 and cell_exists_and_in_bounds(x, y-1):
                move = (x, y, direction, 1, f"POWERUP_{powerup_name}")  # Size 1 for powerup activation
                powerup_moves.append(move)
                print(f"   ⚡ Powerup move: {powerup_name} at ({x},{y}) → {direction}")
            elif direction == 'down' and y < rows-1 and cell_exists_and_in_bounds(x, y+1):
                move = (x, y, direction, 1, f"POWERUP_{powerup_name}")
                powerup_moves.append(move)
                print(f"   ⚡ Powerup move: {powerup_name} at ({x},{y}) → {direction}")
            elif direction == 'left' and x > 0 and cell_exists_and_in_bounds(x-1, y):
                move = (x, y, direction, 1, f"POWERUP_{powerup_name}")
                powerup_moves.append(move)
                print(f"   ⚡ Powerup move: {powerup_name} at ({x},{y}) → {direction}")
            elif direction == 'right' and x < cols-1 and cell_exists_and_in_bounds(x+1, y):
                move = (x, y, direction, 1, f"POWERUP_{powerup_name}")
                powerup_moves.append(move)
                print(f"   ⚡ Powerup move: {powerup_name} at ({x},{y}) → {direction}")
    
    if powerup_moves:
        print(f"   ✅ Created {len(powerup_moves)} powerup moves")
        # Add powerup moves to the beginning of LegalMoves (highest priority)
        LegalMoves = powerup_moves + LegalMoves
    
    return powerup_moves

def searchMoves():
    """Search for all possible moves and print progress."""
    move_count = 0
    for y in range(rows):
        for x in range(cols):
            if debug:
                print(f"Checking position ({x},{y})")
            before_count = len(LegalMoves)
            check5moves(x,y)
            check4moves(x,y)
            check3moves(x,y)
            after_count = len(LegalMoves)
            move_count += (after_count - before_count)
    
    print(f"🎯 Found {len(LegalMoves)} possible moves")

def gridScreenshot(filename,gridX1,gridY1,gridX2,gridY2):
    """Take screenshot of game board area."""
    width = gridX2 - gridX1
    height = gridY2 - gridY1

    final = gridX1,gridY1,width,height
    im = pyautogui.screenshot(region=(final))
    return im

def chooseBestMove(order):
    """Choose best move with POWERUP PRIORITY, then piece preferences with randomness and failed move avoidance."""
    import random
    
    if not LegalMoves:
        return None
    
    # Filter out recently failed moves
    global failed_moves_history, last_failed_move, stuck_move_count
    available_moves = []
    filtered_moves = []
    
    print(f"   📝 Checking {len(LegalMoves)} moves against {len(failed_moves_history)} failed moves")
    
    for move in LegalMoves:
        move_signature = (move[0], move[1], move[2], move[3])  # x, y, direction, size
        if move_signature not in failed_moves_history:
            available_moves.append(move)
        else:
            filtered_moves.append(move_signature)
    
    if filtered_moves:
        print(f"   🚫 Filtered out {len(filtered_moves)} previously failed moves")
    
    # If all moves have failed recently, clear the history and try again
    if not available_moves:
        print("   🔄 All moves were recently failed, clearing failed move history...")
        failed_moves_history.clear()
        available_moves = LegalMoves.copy()
    
    print(f"   🎯 Move selection: {len(available_moves)}/{len(LegalMoves)} moves available (excluding recent failures)")
    
    # PRIORITY 0: POWERUP MOVES (HIGHEST PRIORITY)
    powerup_moves = []
    piece_moves = []
    
    for move in available_moves:
        piece_type = move[4]
        if piece_type.startswith("POWERUP_"):
            powerup_moves.append(move)
        else:
            piece_moves.append(move)
    
    # If powerups are found, prioritize them by number (highest first) BUT apply geographic diversity
    # However, if we've had too many consecutive powerup failures, occasionally try piece moves instead
    consecutive_powerup_failures = stuck_move_count if last_failed_move and len(last_failed_move) > 4 and 'POWERUP' in str(last_failed_move[4]) else 0
    
    if powerup_moves and (consecutive_powerup_failures < 2 or len(piece_moves) == 0):  # Use powerups unless we've had too many failures and have piece alternatives
        print(f"   ⚡ POWERUP PRIORITY: Found {len(powerup_moves)} powerup moves!")
        print(f"   📊 Decision: powerup failures={consecutive_powerup_failures}/2, piece alternatives={len(piece_moves)}")
        
        # Sort powerups by priority (extract number from POWERUP_X and sort descending)
        def get_powerup_priority(move):
            powerup_name = move[4].replace("POWERUP_", "")
            return int(powerup_name) if powerup_name.isdigit() else 0
        
        powerup_moves.sort(key=get_powerup_priority, reverse=True)
        
        # Apply distance-based selection for powerups too!
        if last_failed_move and len(powerup_moves) > 3:
            print(f"   🎯 Filtering powerups by distance from last failed move at ({last_failed_move[0]},{last_failed_move[1]})")
            moves_with_distance = [(move, calculate_move_distance(move, last_failed_move)) for move in powerup_moves]
            # Sort by distance descending (farthest first), then by priority
            moves_with_distance.sort(key=lambda x: (-x[1], -get_powerup_priority(x[0])))
            
            # Only keep powerups that are at least 2 cells away from the failed move
            distant_powerups = [move for move, dist in moves_with_distance if dist >= 2.0]
            if distant_powerups:
                print(f"   🎯 Kept {len(distant_powerups)}/{len(powerup_moves)} powerups distant from failed move")
                powerup_moves = distant_powerups
                powerup_moves.sort(key=get_powerup_priority, reverse=True)  # Re-sort by priority
            else:
                print(f"   ⚠️  No distant powerups found, keeping closest ones")
                powerup_moves = [move for move, dist in moves_with_distance[:len(powerup_moves)//2]]
        
        # Also group powerups by location to ensure geographic diversity
        location_groups = {}
        for move in powerup_moves:
            location = (move[0], move[1])
            if location not in location_groups:
                location_groups[location] = []
            location_groups[location].append(move)
        
        # If we have multiple powerups at the same location, only take the first few from each location
        diverse_powerups = []
        for location, moves in location_groups.items():
            diverse_powerups.extend(moves[:2])  # Max 2 moves per location
        
        if len(diverse_powerups) < len(powerup_moves):
            print(f"   🗺️  Applied geographic diversity: {len(powerup_moves)} → {len(diverse_powerups)} powerup moves")
            powerup_moves = diverse_powerups
            powerup_moves.sort(key=get_powerup_priority, reverse=True)
        
        # Show powerup priorities
        for move in powerup_moves[:3]:  # Show top 3
            powerup_name = move[4].replace("POWERUP_", "")
            priority = get_powerup_priority(move)
            print(f"      ⚡ Powerup {powerup_name} at ({move[0]},{move[1]}) → {move[2]} (Priority: {priority})")
        
        # Add some randomness even to powerup selection
        import random
        if len(powerup_moves) > 1 and random.random() < 0.3:  # 30% chance to pick non-first powerup
            top_powerups = powerup_moves[:min(3, len(powerup_moves))]
            best_powerup = random.choice(top_powerups)
            print(f"   ⚡ SELECTED POWERUP (random): {best_powerup[4].replace('POWERUP_', '')} at ({best_powerup[0]},{best_powerup[1]}) → {best_powerup[2]}")
        else:
            best_powerup = powerup_moves[0]
            print(f"   ⚡ SELECTED POWERUP (priority): {best_powerup[4].replace('POWERUP_', '')} at ({best_powerup[0]},{best_powerup[1]}) → {best_powerup[2]}")
        
        return best_powerup
    
    # No powerups found, or too many powerup failures - proceed with normal piece logic
    if powerup_moves and consecutive_powerup_failures >= 2:
        print(f"   🚫 TOO MANY POWERUP FAILURES ({consecutive_powerup_failures}), switching to piece moves")
    else:
        print(f"   🎮 No powerups found, using normal piece selection...")
    
    # Group piece moves by priority
    priority_1_moves = []  # 5-matches with preferred pieces
    priority_2_moves = []  # 4-matches with preferred pieces
    priority_3_moves = []  # 3-matches with preferred pieces
    other_moves = []       # All other moves
    
    for move in piece_moves:
        match_size = move[3]
        piece_color = move[4]
        
        if match_size == 5 and piece_color in order:
            priority_1_moves.append(move)
        elif match_size == 4 and piece_color in order:
            priority_2_moves.append(move)
        elif match_size == 3 and piece_color in order:
            priority_3_moves.append(move)
        else:
            other_moves.append(move)
    
    # Choose from highest priority group with some randomness and distance-based selection after failures
    def choose_with_randomness(moves_list, description):
        if not moves_list:
            return None
        
        if len(moves_list) == 1:
            print(f"   ✨ Selected {description}: {moves_list[0]}")
            return moves_list[0]
        
        # If we have a recent failed move, prefer moves that are far away from it
        if last_failed_move and len(moves_list) > 3:
            moves_with_distance = [(move, calculate_move_distance(move, last_failed_move)) for move in moves_list]
            # Sort by distance descending (farthest first), then by original order
            moves_with_distance.sort(key=lambda x: (-x[1], moves_list.index(x[0])))
            distant_moves = [move for move, dist in moves_with_distance[:len(moves_list)//2]]
            if distant_moves:
                print(f"   🎯 Prioritizing moves distant from last failed at ({last_failed_move[0]},{last_failed_move[1]})")
                moves_list = distant_moves
        
        # Add randomness: 70% chance to pick best, 30% chance to pick random from top 3
        if random.random() < 0.7:
            # Pick the first (best) move
            chosen = moves_list[0]
            print(f"   ✨ Selected {description} (best): {chosen}")
        else:
            # Pick randomly from top 3 moves
            top_moves = moves_list[:min(3, len(moves_list))]
            chosen = random.choice(top_moves)
            print(f"   ✨ Selected {description} (random from top {len(top_moves)}): {chosen}")
        
        return chosen
    
    # Try each priority level
    result = choose_with_randomness(priority_1_moves, "5-match with preferred piece")
    if result: return result
    
    result = choose_with_randomness(priority_2_moves, "4-match with preferred piece")
    if result: return result
    
    result = choose_with_randomness(priority_3_moves, "3-match with preferred piece")
    if result: return result
    
    result = choose_with_randomness(other_moves, "other available move")
    if result: return result
    
    # Fallback: if somehow nothing was selected
    if available_moves:
        fallback = random.choice(available_moves)
        print(f"   ⚠️  Fallback random selection: {fallback}")
        return fallback
    
    return None

def makeMove(move,x1,y1):
    """Execute move on screen using ANCHOR-BASED GRID coordinates."""
    if move is None:
        return False
        
    x, y, direction, match_size, piece_color = move  # Changed from piece_type to piece_color
    
    # NEW: Use ANCHOR-BASED grid coordinates for move execution
    global true_cell_size, grid_offset_x, grid_offset_y
    
    if true_cell_size > 0:
        # Use anchor-based grid coordinates derived from high-confidence pieces
        centerPointW, centerPointH = calculate_anchor_based_grid_coordinates(grid_offset_x, grid_offset_y, true_cell_size)
        grid_status = "ANCHOR-BASED"
        cell_info = f"cell_size: {true_cell_size} (from high-confidence pieces)"
    else:
        print("⚠️  Grid not calibrated - using fallback square calculation for moves")
        # Fallback to square grid calculation
        cellW = dim[1]//cols
        cellH = dim[0]//rows
        cell_size = min(cellW, cellH)  # Force square cells
        
        centerPointW = []
        centerPointH = []
        for i in range(cols):
            centerPointW.append( (cell_size * i) + (cell_size//2) )
        for j in range(rows):
             centerPointH.append( (cell_size * j) + (cell_size//2) )
        
        grid_status = "FALLBACK-SQUARE"
        cell_info = f"cell_size: {cell_size} (fallback)"

    print(f"🎯 ANCHOR-BASED Move: {piece_color}-piece {direction} (size {match_size}) from grid({x},{y})")
    
    # FIXED: Calculate screen coordinates using calibrated grid + screenshot offset
    # The grid coordinates are relative to the board image, so we need to add the screenshot offset
    start_x = centerPointW[x] + x1
    start_y = centerPointH[y] + y1
    
    # Boundary check for grid coordinates
    if x < 0 or x >= len(centerPointW) or y < 0 or y >= len(centerPointH):
        print(f"❌ Invalid grid position: ({x},{y}) - out of bounds")
        return False
    
    # Move to starting position
    pyautogui.moveTo(start_x, start_y)
    
    # Calculate destination and drag using ANCHOR-BASED GRID constraints
    if direction == 'down':
        if y+1 < rows:  # Boundary check
            dest_x, dest_y = centerPointW[x] + x1, centerPointH[y+1] + y1
        else:
            print(f"❌ Invalid move: down from row {y} (max: {rows-1})")
            return False
    elif direction == 'up':
        if y-1 >= 0:  # Boundary check
            dest_x, dest_y = centerPointW[x] + x1, centerPointH[y-1] + y1
        else:
            print(f"❌ Invalid move: up from row {y}")
            return False
    elif direction == 'left':
        if x-1 >= 0:  # Boundary check
            dest_x, dest_y = centerPointW[x-1] + x1, centerPointH[y] + y1
        else:
            print(f"❌ Invalid move: left from col {x}")
            return False
    elif direction == 'right':
        if x+1 < cols:  # Boundary check
            dest_x, dest_y = centerPointW[x+1] + x1, centerPointH[y] + y1
        else:
            print(f"❌ Invalid move: right from col {x} (max: {cols-1})")
            return False
    else:
        print(f"❌ Unknown direction: {direction}")
        return False
    
    # Additional safety check: ensure coordinates are within reasonable screen bounds
    screen_width, screen_height = pyautogui.size()
    if not (0 <= start_x <= screen_width and 0 <= start_y <= screen_height):
        print(f"❌ Start coordinates out of screen bounds: ({start_x},{start_y})")
        return False
    if not (0 <= dest_x <= screen_width and 0 <= dest_y <= screen_height):
        print(f"❌ Destination coordinates out of screen bounds: ({dest_x},{dest_y})")
        return False
    
    print(f"   Screen coordinates ({grid_status}): ({start_x},{start_y}) → ({dest_x},{dest_y})")
    print(f"   Grid info: {cell_info}, offsets: ({grid_offset_x}, {grid_offset_y})")
    
    pyautogui.dragTo(dest_x, dest_y, 0.3, button='left')
    return True

def get_target_game():
    """
    Ask user for target game name and validate that templates exist.
    """
    print("🎮 Match-3 Bot - Game Selection")
    print("=" * 40)
    
    # Show available games
    templates_base_dir = Path("../../data/templates")
    if templates_base_dir.exists():
        available_games = [d.name for d in templates_base_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
        if available_games:
            print("📁 Available games:")
            for game in sorted(available_games):
                # Check if this game has processed templates
                templates_dir = templates_base_dir / game / "extracted" / "pieces"
                template_count = len(list(templates_dir.glob("*.png"))) if templates_dir.exists() else 0
                status = f"({template_count} templates)" if template_count > 0 else "(no templates)"
                print(f"   • {game} {status}")
            print()
    
    while True:
        game_name = input("Enter target game name: ").strip()
        
        if not game_name:
            print("❌ Please enter a game name")
            continue
            
        game_dir = templates_base_dir / game_name
        templates_dir = game_dir / "extracted" / "pieces"
        
        if not game_dir.exists():
            print(f"❌ Game directory not found: {game_dir}")
            print("   Available games:", ", ".join(available_games) if available_games else "None")
            continue
        
        if not templates_dir.exists():
            print(f"❌ Templates directory not found: {templates_dir}")
            print(f"   Please run template processor for '{game_name}' first")
            continue
            
        # Check if there are any template files
        template_files = list(templates_dir.glob("*.png"))
        if not template_files:
            print(f"❌ No template files found in: {templates_dir}")
            print(f"   Please run template processor for '{game_name}' first")
            continue
            
        print(f"✅ Selected game: '{game_name}' ({len(template_files)} templates available)")
        return game_name

def load_templates(game_name):
    """Load and verify color-named edge-based template files, return dictionary of available templates."""
    templates_dir = Path(f"../../data/templates/{game_name}/extracted/pieces")
    
    print(f"📋 Loading COLOR-NAMED EDGE-BASED templates from {templates_dir}")
    
    if not templates_dir.exists():
        print(f"❌ Templates directory not found: {templates_dir}")
        print(f"   Please run: python3 edge_detection_template_processor.py")
        return {}
        
    template_files = list(templates_dir.glob("*.png"))
    if not template_files:
        print(f"❌ No template files found in {templates_dir}")
        return {}
    
    # Create dictionary mapping template_name -> path (e.g., "R" -> "R.png")
    templates = {}
    for template_file in sorted(template_files):
        # Extract template name from filename (e.g., "R.png" -> "R")
        template_name = template_file.stem
        templates[template_name] = template_file
    
    if not templates:
        print(f"❌ No valid template files found")
        return {}
        
    print(f"✅ Found {len(templates)} color-named edge-based piece templates:")
    for template_name, template_path in sorted(templates.items()):
        print(f"   🎨 {template_path.name} (Color: {template_name})")
        
    return templates

def load_powerup_templates(game_name):
    """Load powerup templates with filename preservation."""
    powerups_dir = Path(f"../../data/templates/{game_name}/extracted/powerups")
    
    print(f"⚡ Loading POWERUP templates from {powerups_dir}")
    
    if not powerups_dir.exists():
        print(f"ℹ️  Powerups directory not found: {powerups_dir}")
        print(f"   No powerups available for {game_name}")
        return {}
        
    powerup_files = list(powerups_dir.glob("*.png"))
    if not powerup_files:
        print(f"ℹ️  No powerup files found in {powerups_dir}")
        return {}
    
    # Create dictionary mapping powerup_name -> path (e.g., "3" -> "3.png")
    powerups = {}
    for powerup_file in sorted(powerup_files):
        # Extract powerup name from filename (e.g., "3.png" -> "3")
        powerup_name = powerup_file.stem
        powerups[powerup_name] = powerup_file
    
    if not powerups:
        print(f"❌ No valid powerup files found")
        return {}
        
    # Sort by priority (highest number = highest priority)
    sorted_powerups = sorted(powerups.items(), key=lambda x: int(x[0]) if x[0].isdigit() else 0, reverse=True)
    
    print(f"✅ Found {len(powerups)} powerup templates (priority order):")
    for powerup_name, powerup_path in sorted_powerups:
        priority = "HIGH" if powerup_name.isdigit() and int(powerup_name) >= 3 else "MEDIUM" if powerup_name.isdigit() and int(powerup_name) == 2 else "LOW"
        print(f"   ⚡ {powerup_path.name} (Priority: {priority})")
        
    return powerups

def create_precise_edge_template(image, blur_kernel=5, canny_low=80, canny_high=160, use_skeleton=True):
    """
    Create a precise edge-based template with thin, clean boundaries.
    Same function as used in precise template processor.
    """
    # Step 1: Apply more blur to reduce noise and focus on major edges
    blurred = cv.GaussianBlur(image, (blur_kernel, blur_kernel), 0)
    
    # Step 2: Convert to grayscale for edge detection
    gray = cv.cvtColor(blurred, cv.COLOR_BGR2GRAY)
    
    # Step 3: Apply histogram equalization for better contrast
    equalized = cv.equalizeHist(gray)
    
    # Step 4: Apply Canny edge detection with more selective thresholds
    edges = cv.Canny(equalized, canny_low, canny_high, apertureSize=3, L2gradient=True)
    
    # Step 5: Aggressive cleanup - remove small components and noise
    # Remove very small connected components
    num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(edges, connectivity=8)
    
    # Filter out small components (less than 10 pixels)
    min_component_size = 10
    edges_filtered = np.zeros_like(edges)
    
    for i in range(1, num_labels):  # Skip background (label 0)
        if stats[i, cv.CC_STAT_AREA] >= min_component_size:
            edges_filtered[labels == i] = 255
    
    # Step 6: Additional morphological cleanup
    kernel_clean = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2, 2))
    edges_cleaned = cv.morphologyEx(edges_filtered, cv.MORPH_OPEN, kernel_clean)
    
    # Step 7: Optionally apply skeletonization for thinner edges
    if use_skeleton:
        # Skeletonize to get single-pixel-width edges
        edges_skeleton = skeletonize_board(edges_cleaned)
        edges_final = edges_skeleton
    else:
        edges_final = edges_cleaned
    
    # Step 8: Final cleanup - remove isolated pixels
    kernel_final = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2, 2))
    edges_final = cv.morphologyEx(edges_final, cv.MORPH_OPEN, kernel_final)
    
    # Step 9: Invert so edges are white on black background
    edges_inverted = cv.bitwise_not(edges_final)
    
    return edges_inverted

def skeletonize_board(image):
    """
    Apply morphological skeletonization to get single-pixel-width edges for board processing.
    """
    # Convert to binary if needed
    if len(image.shape) > 2:
        binary = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    else:
        binary = image.copy()
    
    # Ensure binary (0 or 255)
    _, binary = cv.threshold(binary, 127, 255, cv.THRESH_BINARY)
    
    # Morphological skeletonization
    skeleton = np.zeros(binary.shape, np.uint8)
    element = cv.getStructuringElement(cv.MORPH_CROSS, (3, 3))
    
    iteration_count = 0
    max_iterations = 20  # Prevent infinite loops
    
    while iteration_count < max_iterations:
        # Open operation
        opened = cv.morphologyEx(binary, cv.MORPH_OPEN, element)
        # Subtract opened from original
        temp = cv.subtract(binary, opened)
        # Add to skeleton
        skeleton = cv.bitwise_or(skeleton, temp)
        # Erode the original
        binary = cv.erode(binary, element)
        
        iteration_count += 1
        
        # Break if no more pixels to process
        if cv.countNonZero(binary) == 0:
            break
    
    return skeleton

def create_multi_precision_edge_template(image):
    """
    Create template with multiple precision levels for robustness while maintaining elegance.
    Same function as used in precise template processor.
    """
    # Highly selective edges (strict thresholds, skeletonized)
    ultra_selective = create_precise_edge_template(image, blur_kernel=5, canny_low=100, canny_high=200, use_skeleton=True)
    
    # Moderately selective edges (medium thresholds, cleaned)
    selective = create_precise_edge_template(image, blur_kernel=5, canny_low=60, canny_high=140, use_skeleton=False)
    
    # Combine both - ultra-selective provides precision, selective provides coverage
    combined = cv.bitwise_or(ultra_selective, selective)
    
    # Final aggressive cleanup - ensure we only keep strong, connected edges
    kernel_final = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    final_edges = cv.morphologyEx(combined, cv.MORPH_CLOSE, kernel_final)
    
    # Remove tiny components again after combination
    num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(cv.bitwise_not(final_edges), connectivity=8)
    
    final_cleaned = final_edges.copy()
    min_final_size = 15  # Larger minimum size for final template
    
    for i in range(1, num_labels):  # Skip background
        if stats[i, cv.CC_STAT_AREA] < min_final_size:
            final_cleaned[labels == i] = 0  # Remove small components
    
    return final_cleaned

def process_board_for_precise_edge_matching(board_image):
    """
    Convert the live board to precise edge format matching the templates.
    """
    global edge_processed_board
    print("✨ Processing board with precise edge detection...")
    edge_processed_board = create_multi_precision_edge_template(board_image)
    
    # Save debug image to see the precise edge detection results
    debug_path = "../../precise_edge_board_capture.png"
    cv.imwrite(debug_path, edge_processed_board)
    print(f"📸 Debug: Precise edge-processed board saved to {debug_path}")
    
    return edge_processed_board

def detect_board_boundary(board_image):
    """
    Detect the actual board boundary using line detection.
    Returns the bounding box coordinates of the game board.
    """
    print("🔍 Detecting board boundary using line detection...")
    
    # Convert to grayscale for line detection
    gray = cv.cvtColor(board_image, cv.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv.GaussianBlur(gray, (5, 5), 0)
    
    # Use adaptive threshold to handle varying lighting
    adaptive_thresh = cv.adaptiveThreshold(blurred, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 2)
    
    # Detect edges with Canny
    edges = cv.Canny(blurred, 50, 150, apertureSize=3)
    
    # Use morphological operations to connect broken lines
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    edges_closed = cv.morphologyEx(edges, cv.MORPH_CLOSE, kernel)
    
    # Find contours to detect the board boundary
    contours, _ = cv.findContours(edges_closed, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        print("   ⚠️  No contours found, using full image area")
        return 0, 0, board_image.shape[1], board_image.shape[0]
    
    # Find the largest rectangular contour (likely the board boundary)
    board_contour = None
    max_area = 0
    
    for contour in contours:
        # Approximate contour to reduce points
        epsilon = 0.02 * cv.arcLength(contour, True)
        approx = cv.approxPolyDP(contour, epsilon, True)
        
        area = cv.contourArea(contour)
        
        # Look for rectangular-ish contours (3-6 vertices) with significant area
        if len(approx) >= 4 and len(approx) <= 8 and area > max_area:
            # Must be at least 20% of image area to be considered
            if area > (board_image.shape[0] * board_image.shape[1] * 0.2):
                max_area = area
                board_contour = contour
    
    if board_contour is None:
        print("   ⚠️  No suitable board boundary found, using full image")
        return 0, 0, board_image.shape[1], board_image.shape[0]
    
    # Get bounding rectangle of the board contour
    x, y, w, h = cv.boundingRect(board_contour)
    
    print(f"   ✅ Board boundary detected: ({x},{y}) size {w}x{h}")
    print(f"   Board area: {max_area:.0f} pixels ({max_area/(board_image.shape[0]*board_image.shape[1])*100:.1f}% of image)")
    
    # Save debug image showing detected boundary
    debug_image = board_image.copy()
    cv.drawContours(debug_image, [board_contour], -1, (0, 255, 0), 2)
    cv.rectangle(debug_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
    cv.imwrite("../../debug_board_boundary.png", debug_image)
    print(f"   📸 Board boundary debug saved to debug_board_boundary.png")
    
    return x, y, w, h

def find_reference_pieces_in_board(board_image, board_x, board_y, board_w, board_h, available_templates, optimal_thresholds):
    """
    Find actual game pieces within the detected board boundary.
    Returns positions of detected pieces for grid inference.
    """
    print("🎯 Finding reference pieces within board boundary...")
    
    # Extract board region
    board_region = board_image[board_y:board_y+board_h, board_x:board_x+board_w]
    
    # Process board region for edge detection
    edge_processed_region = create_multi_scale_edge_template(board_region)
    
    piece_positions = []  # List of (x, y, template_name, confidence)
    
    # Search for pieces using each template
    for template_name, template_path in available_templates.items():
        template = cv.imread(str(template_path), cv.IMREAD_GRAYSCALE)
        if template is None:
            continue
            
        # Scale template
        global optimal_template_scale
        if optimal_template_scale > 0:
            new_width = int(template.shape[1] * optimal_template_scale)
            new_height = int(template.shape[0] * optimal_template_scale)
            template = cv.resize(template, (new_width, new_height), interpolation=cv.INTER_CUBIC)
        
        # Template matching
        res = cv.matchTemplate(edge_processed_region, template, cv.TM_CCOEFF_NORMED)
        threshold = optimal_thresholds.get(template_name, 0.25)
        
        # Find matches above threshold
        locations = np.where(res >= threshold)
        
        for pt in zip(*locations[::-1]):
            match_center_x = pt[0] + template.shape[1] // 2
            match_center_y = pt[1] + template.shape[0] // 2
            confidence = res[pt[1], pt[0]]
            
            # Convert back to full image coordinates
            full_x = board_x + match_center_x
            full_y = board_y + match_center_y
            
            piece_positions.append((full_x, full_y, template_name, confidence))
    
    # Remove duplicate detections (same position, different templates)
    filtered_positions = []
    for pos in piece_positions:
        is_duplicate = False
        for existing in filtered_positions:
            distance = np.sqrt((pos[0] - existing[0])**2 + (pos[1] - existing[1])**2)
            if distance < 20:  # If within 20 pixels, consider duplicate
                if pos[3] > existing[3]:  # Keep higher confidence
                    filtered_positions.remove(existing)
                else:
                    is_duplicate = True
                    break
        if not is_duplicate:
            filtered_positions.append(pos)
    
    print(f"   ✅ Found {len(filtered_positions)} distinct pieces in board region")
    
    # Sort by confidence and return top matches
    filtered_positions.sort(key=lambda x: x[3], reverse=True)
    
    return filtered_positions[:20]  # Return top 20 pieces for grid inference

def infer_grid_structure_from_pieces(piece_positions, board_x, board_y, board_w, board_h):
    """
    Infer the grid structure from actual piece positions.
    Returns grid parameters: cell_size, grid_origin_x, grid_origin_y, detected_cols, detected_rows
    """
    print("🧮 Inferring grid structure AND dimensions from piece positions...")
    
    if len(piece_positions) < 3:  # Need more pieces to detect dimensions
        print("   ⚠️  Need at least 3 pieces to infer grid structure and dimensions")
        return None, None, None, None, None
    
    # Extract just the x,y coordinates relative to board
    positions = [(pos[0] - board_x, pos[1] - board_y) for pos in piece_positions]
    
    # Find horizontal and vertical spacing patterns
    x_coords = [pos[0] for pos in positions]
    y_coords = [pos[1] for pos in positions]
    
    x_coords.sort()
    y_coords.sort()
    
    # Calculate differences between consecutive positions
    x_diffs = [x_coords[i+1] - x_coords[i] for i in range(len(x_coords)-1)]
    y_diffs = [y_coords[i+1] - y_coords[i] for i in range(len(y_coords)-1)]
    
    # Filter out very small differences (likely same cell) and very large ones (likely skipped cells)
    x_diffs_filtered = [d for d in x_diffs if 20 < d < 100]
    y_diffs_filtered = [d for d in y_diffs if 20 < d < 100]
    
    if not x_diffs_filtered or not y_diffs_filtered:
        print("   ⚠️  Could not find valid spacing patterns")
        return None, None, None, None, None
    
    # Find the most common spacing (mode)
    def find_most_common_spacing(diffs):
        # Group similar spacings together (within 5 pixels)
        groups = []
        for diff in diffs:
            added_to_group = False
            for group in groups:
                if abs(diff - np.mean(group)) <= 5:
                    group.append(diff)
                    added_to_group = True
                    break
            if not added_to_group:
                groups.append([diff])
        
        # Return the mean of the largest group
        largest_group = max(groups, key=len)
        return np.mean(largest_group)
    
    cell_width = find_most_common_spacing(x_diffs_filtered)
    cell_height = find_most_common_spacing(y_diffs_filtered)
    
    # For match-3 games, enforce square cells
    cell_size = int((cell_width + cell_height) / 2)
    
    print(f"   Detected spacing: width={cell_width:.1f}, height={cell_height:.1f}")
    print(f"   Enforced square cell size: {cell_size}x{cell_size}")
    
    # DETECT GRID DIMENSIONS from board size and cell size
    detected_cols = round(board_w / cell_size)
    detected_rows = round(board_h / cell_size)
    
    print(f"   🎯 DETECTED GRID DIMENSIONS: {detected_cols} cols × {detected_rows} rows")
    print(f"   Board area: {board_w}x{board_h}, Cell size: {cell_size}x{cell_size}")
    
    # Find grid origin by finding the piece closest to top-left of board
    min_distance = float('inf')
    origin_piece = None
    
    for pos in positions:
        distance = np.sqrt(pos[0]**2 + pos[1]**2)
        if distance < min_distance:
            min_distance = distance
            origin_piece = pos
    
    if origin_piece is None:
        print("   ⚠️  Could not determine grid origin")
        return None, None, None, None, None
    
    # Calculate grid origin by snapping to likely grid positions
    grid_origin_x = origin_piece[0] % cell_size
    grid_origin_y = origin_piece[1] % cell_size
    
    # Adjust origin to center of first cell
    grid_origin_x = grid_origin_x - (cell_size // 2)
    grid_origin_y = grid_origin_y - (cell_size // 2)
    
    # Make sure origin is positive
    while grid_origin_x < 0:
        grid_origin_x += cell_size
    while grid_origin_y < 0:
        grid_origin_y += cell_size
    
    print(f"   Grid origin within board: ({grid_origin_x}, {grid_origin_y})")
    print(f"   Cell size: {cell_size}x{cell_size}")
    
    return cell_size, grid_origin_x, grid_origin_y, detected_cols, detected_rows

def calculate_robust_grid_coordinates(board_x, board_y, cell_size, grid_origin_x, grid_origin_y):
    """
    Calculate final grid coordinates using the inferred structure.
    """
    cell_centers_x = []
    cell_centers_y = []
    
    for i in range(cols):
        center_x = board_x + grid_origin_x + (cell_size * i)
        cell_centers_x.append(center_x)
    
    for j in range(rows):
        center_y = board_y + grid_origin_y + (cell_size * j)
        cell_centers_y.append(center_y)
    
    return cell_centers_x, cell_centers_y

def validate_robust_grid_detection(board_image, board_x, board_y, board_w, board_h, cell_centers_x, cell_centers_y, piece_positions):
    """
    Validate the robust grid detection by showing piece positions vs inferred grid.
    """
    print("   🔍 Validating robust grid detection...")
    
    # Create validation image
    debug_image = board_image.copy()
    
    # Draw board boundary
    cv.rectangle(debug_image, (board_x, board_y), (board_x + board_w, board_y + board_h), (0, 255, 0), 2)
    
    # Draw inferred grid lines
    for x in cell_centers_x:
        cv.line(debug_image, (x, board_y), (x, board_y + board_h), (255, 0, 0), 1)
    
    for y in cell_centers_y:
        cv.line(debug_image, (board_x, y), (board_x + board_w, y), (255, 0, 0), 1)
    
    # Draw grid centers
    for i, x in enumerate(cell_centers_x):
        for j, y in enumerate(cell_centers_y):
            if board_y <= y <= board_y + board_h and board_x <= x <= board_x + board_w:
                cv.circle(debug_image, (x, y), 2, (255, 0, 0), -1)
                cv.putText(debug_image, f"{i},{j}", (x-8, y-8), cv.FONT_HERSHEY_SIMPLEX, 0.25, (255, 255, 255), 1)
    
    # Draw detected pieces
    for pos in piece_positions:
        cv.circle(debug_image, (pos[0], pos[1]), 5, (0, 255, 255), 2)
        cv.putText(debug_image, pos[2], (pos[0]+5, pos[1]-5), cv.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1)
    
    # Save debug image
    cv.imwrite("../../debug_robust_grid_overlay.png", debug_image)
    print(f"   📸 Robust grid validation saved to debug_robust_grid_overlay.png")
    
    return True

def find_high_confidence_anchor_pieces(board_image, available_templates):
    """
    Find high-confidence template matches to use as grid anchors.
    Try multiple scales if needed to find reliable matches.
    Returns list of (x, y, template_name, confidence, scale_used)
    """
    print("🎯 Finding high-confidence anchor pieces...")
    
    # Process board for edge detection
    edge_processed_board = create_multi_scale_edge_template(board_image)
    
    anchor_pieces = []
    scales_to_try = [0.4, 0.5, 0.6, 0.7, 0.8]  # Multiple scales to try
    min_confidence = 0.35  # Lowered from 0.7 - based on actual confidence levels we see
    
    for template_name, template_path in available_templates.items():
        template = cv.imread(str(template_path), cv.IMREAD_GRAYSCALE)
        if template is None:
            continue
            
        print(f"   Testing template {template_name} at multiple scales...")
        
        best_matches_for_template = []
        
        for scale in scales_to_try:
            # Scale template
            new_width = int(template.shape[1] * scale)
            new_height = int(template.shape[0] * scale)
            
            if new_width < 8 or new_height < 8 or new_width > 100 or new_height > 100:
                continue
                
            scaled_template = cv.resize(template, (new_width, new_height), interpolation=cv.INTER_CUBIC)
            
            # Template matching
            res = cv.matchTemplate(edge_processed_board, scaled_template, cv.TM_CCOEFF_NORMED)
            
            # Find high-confidence matches
            locations = np.where(res >= min_confidence)
            
            for pt in zip(*locations[::-1]):
                match_center_x = pt[0] + scaled_template.shape[1] // 2
                match_center_y = pt[1] + scaled_template.shape[0] // 2
                confidence = res[pt[1], pt[0]]
                
                best_matches_for_template.append((match_center_x, match_center_y, template_name, confidence, scale))
        
        # Sort by confidence and take top matches
        best_matches_for_template.sort(key=lambda x: x[3], reverse=True)
        anchor_pieces.extend(best_matches_for_template[:5])  # Top 5 per template instead of 3
    
    # Remove duplicates (same position, different templates/scales)
    filtered_anchors = []
    for piece in anchor_pieces:
        is_duplicate = False
        for existing in filtered_anchors:
            distance = np.sqrt((piece[0] - existing[0])**2 + (piece[1] - existing[1])**2)
            if distance < 15:  # If within 15 pixels, consider duplicate
                if piece[3] > existing[3]:  # Keep higher confidence
                    filtered_anchors.remove(existing)
                else:
                    is_duplicate = True
                    break
        if not is_duplicate:
            filtered_anchors.append(piece)
    
    # Sort by confidence
    filtered_anchors.sort(key=lambda x: x[3], reverse=True)
    
    print(f"   ✅ Found {len(filtered_anchors)} high-confidence anchor pieces (min confidence: {min_confidence})")
    for i, (x, y, name, conf, scale) in enumerate(filtered_anchors[:8]):  # Show top 8
        print(f"      {i+1}. {name} at ({x},{y}) confidence={conf:.3f} scale={scale:.2f}")
    
    return filtered_anchors

def find_adjacent_piece(anchor_piece, all_anchor_pieces, expected_distance_range=(30, 50)):
    """
    Find the nearest adjacent piece to establish grid spacing.
    Returns (adjacent_piece, distance, direction) or None if not found.
    """
    anchor_x, anchor_y = anchor_piece[0], anchor_piece[1]
    
    best_adjacent = None
    best_distance = float('inf')
    best_direction = None
    
    print(f"   Looking for adjacent pieces to anchor at ({anchor_x},{anchor_y})...")
    
    for other_piece in all_anchor_pieces:
        if other_piece == anchor_piece:
            continue
            
        other_x, other_y = other_piece[0], other_piece[1]
        
        # Calculate horizontal and vertical distances
        dx = abs(other_x - anchor_x)
        dy = abs(other_y - anchor_y)
        
        print(f"      Checking {other_piece[2]} at ({other_x},{other_y}): dx={dx}, dy={dy}")
        
        # Check if this could be an adjacent piece (either horizontal or vertical neighbor)
        # Expanded search range and more flexible tolerance
        if expected_distance_range[0] <= dx <= expected_distance_range[1] and dy < 20:
            # Horizontal neighbor
            distance = dx
            direction = 'horizontal'
            print(f"         → Potential horizontal neighbor (dx={dx}, dy={dy})")
        elif expected_distance_range[0] <= dy <= expected_distance_range[1] and dx < 20:
            # Vertical neighbor  
            distance = dy
            direction = 'vertical'
            print(f"         → Potential vertical neighbor (dx={dx}, dy={dy})")
        else:
            # Also check for slightly diagonal pieces that might be grid-aligned
            total_distance = np.sqrt(dx*dx + dy*dy)
            if expected_distance_range[0] <= total_distance <= expected_distance_range[1] * 1.4:
                # Diagonal but within reasonable range - use the larger component
                if dx > dy:
                    distance = dx
                    direction = 'horizontal'
                    print(f"         → Potential diagonal-horizontal neighbor (total={total_distance:.1f})")
                else:
                    distance = dy
                    direction = 'vertical'
                    print(f"         → Potential diagonal-vertical neighbor (total={total_distance:.1f})")
            else:
                continue
            
        if distance < best_distance:
            best_distance = distance
            best_adjacent = other_piece
            best_direction = direction
            print(f"         → NEW BEST: distance={distance:.1f}, direction={direction}")
    
    if best_adjacent:
        print(f"   Found best adjacent: {best_adjacent[2]} at distance {best_distance:.1f} ({best_direction})")
        return best_adjacent, best_distance, best_direction
    else:
        print(f"   No adjacent pieces found within range {expected_distance_range}")
        return None

def infer_grid_from_anchor_pieces(board_image, anchor_pieces):
    """
    Use high-confidence anchor pieces to infer the complete grid structure.
    Returns grid parameters: cell_size, grid_origin_x, grid_origin_y, detected_cols, detected_rows
    """
    print("🧮 Inferring grid structure from high-confidence anchor pieces...")
    
    if len(anchor_pieces) < 2:
        print("   ⚠️  Need at least 2 anchor pieces to infer grid")
        return None, None, None, None, None
    
    # Use the highest confidence piece as primary anchor
    primary_anchor = anchor_pieces[0]
    print(f"   Primary anchor: {primary_anchor[2]} at ({primary_anchor[0]},{primary_anchor[1]}) conf={primary_anchor[3]:.3f}")
    
    # Find adjacent piece to establish cell size
    adjacent_info = find_adjacent_piece(primary_anchor, anchor_pieces)
    
    if adjacent_info is None:
        print("   ⚠️  Could not find adjacent piece to primary anchor")
        return None, None, None, None, None
    
    adjacent_piece, cell_size, direction = adjacent_info
    print(f"   Adjacent piece: {adjacent_piece[2]} at ({adjacent_piece[0]},{adjacent_piece[1]}) distance={cell_size:.1f} direction={direction}")
    
    # Use the cell size to determine grid dimensions
    board_height, board_width = board_image.shape[:2]
    detected_cols = round(board_width / cell_size)
    detected_rows = round(board_height / cell_size)
    
    print(f"   🎯 ANCHOR-BASED GRID DIMENSIONS: {detected_cols} cols × {detected_rows} rows")
    print(f"   Board size: {board_width}x{board_height}, Inferred cell size: {cell_size:.1f}x{cell_size:.1f}")
    
    # FIXED: Calculate grid origin to ensure positive coordinates
    # Find the piece closest to top-left corner to use as reference
    min_distance = float('inf')
    reference_piece = None
    
    for piece in anchor_pieces:
        distance = np.sqrt(piece[0]**2 + piece[1]**2)  # Distance from top-left
        if distance < min_distance:
            min_distance = distance
            reference_piece = piece
    
    if reference_piece is None:
        print("   ⚠️  Could not find reference piece")
        return None, None, None, None, None
    
    ref_x, ref_y = reference_piece[0], reference_piece[1]
    print(f"   Reference piece: {reference_piece[2]} at ({ref_x},{ref_y}) (closest to top-left)")
    
    # FIXED: Calculate grid origin based on actual piece positions
    # Use multiple anchor pieces to find the best grid origin
    best_origin_x = 0
    best_origin_y = 0
    min_total_error = float('inf')
    
    print(f"   🔍 Testing grid origins to find best fit for {len(anchor_pieces)} anchor pieces...")
    print(f"   Board dimensions: {board_width}x{board_height}, Cell size: {cell_size}")
    
    # Try different grid origins within a reasonable range
    for test_origin_x in range(0, int(cell_size), 3):  # Step by 3 for faster search
        for test_origin_y in range(0, int(cell_size), 3):
            total_error = 0
            valid_pieces = 0
            
            # Test how well this origin fits all anchor pieces
            for piece in anchor_pieces[:8]:  # Test with first 8 pieces
                piece_x, piece_y = piece[0], piece[1]
                
                # Calculate which grid cell this piece should be in
                estimated_col = round((piece_x - test_origin_x - cell_size // 2) / cell_size)
                estimated_row = round((piece_y - test_origin_y - cell_size // 2) / cell_size)
                
                # Ensure within bounds
                if 0 <= estimated_col < detected_cols and 0 <= estimated_row < detected_rows:
                    # Calculate expected center position
                    expected_x = test_origin_x + estimated_col * cell_size + cell_size // 2
                    expected_y = test_origin_y + estimated_row * cell_size + cell_size // 2
                    
                    # Calculate error
                    error = np.sqrt((piece_x - expected_x)**2 + (piece_y - expected_y)**2)
                    total_error += error
                    valid_pieces += 1
            
            # Average error for this origin
            if valid_pieces > 0:
                avg_error = total_error / valid_pieces
                if avg_error < min_total_error:
                    min_total_error = avg_error
                    best_origin_x = test_origin_x
                    best_origin_y = test_origin_y
    
    grid_origin_x = best_origin_x
    grid_origin_y = best_origin_y
    
    print(f"   ✅ Optimized grid origin: ({grid_origin_x}, {grid_origin_y}) with avg error: {min_total_error:.1f}")
    
    # Validate the optimized grid with a few pieces
    print(f"   🔍 Validation with optimized grid:")
    for i, piece in enumerate(anchor_pieces[:3]):
        piece_x, piece_y = piece[0], piece[1]
        estimated_col = round((piece_x - grid_origin_x - cell_size // 2) / cell_size)
        estimated_row = round((piece_y - grid_origin_y - cell_size // 2) / cell_size)
        
        if 0 <= estimated_col < detected_cols and 0 <= estimated_row < detected_rows:
            expected_x = grid_origin_x + estimated_col * cell_size + cell_size // 2
            expected_y = grid_origin_y + estimated_row * cell_size + cell_size // 2
            error = np.sqrt((piece_x - expected_x)**2 + (piece_y - expected_y)**2)
            print(f"      {piece[2]} at ({piece_x},{piece_y}) → grid({estimated_col},{estimated_row}) error={error:.1f}")
    
    # Ensure grid doesn't extend beyond board boundaries
    max_grid_width = detected_cols * cell_size
    max_grid_height = detected_rows * cell_size
    
    if grid_origin_x + max_grid_width > board_width:
        grid_origin_x = board_width - max_grid_width
        print(f"   📏 Adjusted grid_origin_x to {grid_origin_x} to fit within board width")
    if grid_origin_y + max_grid_height > board_height:
        grid_origin_y = board_height - max_grid_height
        print(f"   📏 Adjusted grid_origin_y to {grid_origin_y} to fit within board height")
    
    # Final check: ensure grid origin is still positive after adjustments
    grid_origin_x = max(0, grid_origin_x)
    grid_origin_y = max(0, grid_origin_y)
    
    print(f"   📍 Final grid origin: ({grid_origin_x}, {grid_origin_y})")
    
    return int(cell_size), int(grid_origin_x), int(grid_origin_y), detected_cols, detected_rows

def calibrate_grid_from_pieces(board_image, available_templates):
    """
    New approach: Start with high-confidence pieces, find adjacent pieces, infer grid.
    """
    print("\n🎯 ANCHOR-BASED GRID DETECTION SYSTEM")
    print("   Step 1: Find high-confidence template matches") 
    print("   Step 2: Identify adjacent pieces to establish spacing")
    print("   Step 3: Use anchor pieces to infer complete grid structure")
    print("   This approach prioritizes piece detection over boundary detection")
    
    # Step 1: Find high-confidence anchor pieces
    anchor_pieces = find_high_confidence_anchor_pieces(board_image, available_templates)
    
    if len(anchor_pieces) < 2:
        print("   ⚠️  Insufficient high-confidence pieces found")
        return None, None, None, None, None
    
    # Step 2 & 3: Use anchor pieces to infer grid structure
    cell_size, grid_origin_x, grid_origin_y, detected_cols, detected_rows, anchor_pieces = infer_grid_from_anchor_pieces(
        board_image, anchor_pieces
    )
    
    if cell_size is None:
        print("   ⚠️  Could not infer grid structure from anchor pieces")
        return None, None, None, None, None
    
    return cell_size, grid_origin_x, grid_origin_y, detected_cols, detected_rows, anchor_pieces

def validate_anchor_based_grid_detection(board_image, cell_size, grid_origin_x, grid_origin_y, anchor_pieces):
    """
    Create debug visualization showing anchor-based grid detection.
    """
    print("   🔍 Validating anchor-based grid detection...")
    
    debug_image = board_image.copy()
    board_height, board_width = board_image.shape[:2]
    
    # Draw inferred grid
    cols = round(board_width / cell_size)
    rows = round(board_height / cell_size)
    
    # Draw vertical grid lines
    for i in range(cols + 1):
        x = int(grid_origin_x + i * cell_size)
        if 0 <= x < board_width:
            cv.line(debug_image, (x, 0), (x, board_height), (255, 0, 0), 1)
    
    # Draw horizontal grid lines  
    for j in range(rows + 1):
        y = int(grid_origin_y + j * cell_size)
        if 0 <= y < board_height:
            cv.line(debug_image, (0, y), (board_width, y), (255, 0, 0), 1)
    
    # Draw cell centers
    for i in range(cols):
        for j in range(rows):
            center_x = int(grid_origin_x + i * cell_size + cell_size // 2)
            center_y = int(grid_origin_y + j * cell_size + cell_size // 2)
            if 0 <= center_x < board_width and 0 <= center_y < board_height:
                cv.circle(debug_image, (center_x, center_y), 2, (255, 0, 0), -1)
                cv.putText(debug_image, f"{i},{j}", (center_x-8, center_y-8), cv.FONT_HERSHEY_SIMPLEX, 0.25, (255, 255, 255), 1)
    
    # Highlight anchor pieces
    for i, (x, y, name, conf, scale) in enumerate(anchor_pieces[:10]):
        color = (0, 255, 255) if i == 0 else (0, 255, 0)  # Yellow for primary, green for others
        cv.circle(debug_image, (int(x), int(y)), 8, color, 2)
        cv.putText(debug_image, f"{name}({conf:.2f})", (int(x)+10, int(y)-10), cv.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    # Save debug image
    cv.imwrite("../../debug_anchor_based_grid.png", debug_image)
    print(f"   📸 Anchor-based grid validation saved to debug_anchor_based_grid.png")
    
    return True

def find_two_adjacent_pieces_with_scaling(board_image, available_templates):
    """
    Historically successful methodology: Find exactly two adjacent pieces with extremely high confidence
    using iterative template scaling on edge-detected board and templates.
    OPTIMIZED to prevent hanging.
    """
    print("🎯 PROVEN METHODOLOGY: Finding two adjacent pieces with iterative scaling")
    
    # SIMPLIFIED: Test fewer scales to prevent hanging
    scale_range = np.arange(0.4, 0.7, 0.1)  # Reduced range: 0.4, 0.5, 0.6
    best_adjacent_pair = None
    best_confidence_sum = 0
    
    for scale in scale_range:
        print(f"   🔍 Testing scale {scale:.1f}...")
        
        # Find high-confidence pieces at this scale
        high_confidence_pieces = []
        confidence_threshold = 0.35  # LOWERED - more realistic for candy crush
        
        for template_name, template_path in available_templates.items():
            template = cv.imread(str(template_path), cv.IMREAD_GRAYSCALE)
            if template is None:
                continue
                
            # Scale template
            new_width = int(template.shape[1] * scale)
            new_height = int(template.shape[0] * scale)
            if new_width < 5 or new_height < 5:
                continue
                
            scaled_template = cv.resize(template, (new_width, new_height))
            
            # Match against edge-processed board
            board_gray = cv.cvtColor(board_image, cv.COLOR_BGR2GRAY) if len(board_image.shape) == 3 else board_image
            result = cv.matchTemplate(board_gray, scaled_template, cv.TM_CCOEFF_NORMED)
            
            # Find locations above threshold
            locations = np.where(result >= confidence_threshold)
            
            for pt in zip(*locations[::-1]):
                confidence = result[pt[1], pt[0]]
                center_x = pt[0] + scaled_template.shape[1] // 2
                center_y = pt[1] + scaled_template.shape[0] // 2
                high_confidence_pieces.append((center_x, center_y, template_name, confidence, scale))
        
        print(f"      Found {len(high_confidence_pieces)} pieces at scale {scale:.2f}")
        
        # Look for adjacent pairs (horizontal or vertical) with improved logic
        for i, piece1 in enumerate(high_confidence_pieces):
            for j, piece2 in enumerate(high_confidence_pieces[i+1:], i+1):
                x1, y1, name1, conf1, scale1 = piece1
                x2, y2, name2, conf2, scale2 = piece2
                
                # Calculate distance
                dx = abs(x2 - x1)
                dy = abs(y2 - y1)
                distance = np.sqrt(dx**2 + dy**2)
                
                # More flexible adjacency detection for candy crush
                # Board is 261x251, expect around 7x6 grid, so cell size ~37
                expected_cell_size = min(261/7, 251/6)  # ~37
                tolerance = expected_cell_size * 0.3  # 30% tolerance
                
                # Check if pieces are adjacent (horizontal or vertical)
                is_horizontal = dy <= tolerance and (expected_cell_size - tolerance) <= dx <= (expected_cell_size + tolerance)
                is_vertical = dx <= tolerance and (expected_cell_size - tolerance) <= dy <= (expected_cell_size + tolerance)
                
                if is_horizontal or is_vertical:
                    confidence_sum = conf1 + conf2
                    direction = "horizontal" if is_horizontal else "vertical"
                    
                    if confidence_sum > best_confidence_sum:
                        best_confidence_sum = confidence_sum
                        best_adjacent_pair = (piece1, piece2, distance, direction, scale)
                        print(f"      ✨ Found adjacent pair: {name1}({conf1:.3f}) + {name2}({conf2:.3f}) = {confidence_sum:.3f}")
                        print(f"         Distance: {distance:.1f}, Direction: {direction}, Expected: ~{expected_cell_size:.1f}")
    
    if best_adjacent_pair is None:
        print("   ❌ Could not find two adjacent pieces with sufficient confidence")
        print(f"   💡 Try lowering confidence threshold or adjusting scale range")
        return None
        
    piece1, piece2, distance, direction, optimal_scale = best_adjacent_pair
    print(f"   ✅ BEST ADJACENT PAIR: {piece1[2]}+{piece2[2]} confidence={best_confidence_sum:.3f}")
    print(f"      Scale: {optimal_scale:.2f}, Distance: {distance:.1f}, Direction: {direction}")
    
    return piece1, piece2, distance, direction, optimal_scale

def calculate_initial_grid_from_adjacent_pieces(board_image, piece1, piece2, distance, direction):
    """
    Calculate initial grid parameters from two adjacent pieces.
    """
    print("📐 Calculating initial grid from adjacent pieces...")
    
    board_height, board_width = board_image.shape[:2]
    cell_size = distance  # Distance between adjacent pieces = cell size
    
    # Use the leftmost/topmost piece as reference
    if direction == "horizontal":
        ref_piece = piece1 if piece1[0] < piece2[0] else piece2
        grid_cols = round(board_width / cell_size)
        grid_rows = round(board_height / cell_size)
    else:  # vertical
        ref_piece = piece1 if piece1[1] < piece2[1] else piece2
        grid_cols = round(board_width / cell_size)
        grid_rows = round(board_height / cell_size)
    
    # Calculate initial grid origin
    ref_x, ref_y = ref_piece[0], ref_piece[1]
    
    # Estimate which grid cell the reference piece is in
    estimated_col = round(ref_x / cell_size)
    estimated_row = round(ref_y / cell_size)
    
    # Calculate grid origin to place reference piece in estimated cell
    grid_origin_x = ref_x - (estimated_col * cell_size + cell_size // 2)
    grid_origin_y = ref_y - (estimated_row * cell_size + cell_size // 2)
    
    # Ensure positive origin
    grid_origin_x = max(0, grid_origin_x)
    grid_origin_y = max(0, grid_origin_y)
    
    print(f"   Initial grid: {grid_cols}x{grid_rows}, cell_size: {cell_size:.1f}")
    print(f"   Initial origin: ({grid_origin_x:.1f}, {grid_origin_y:.1f})")
    
    return int(cell_size), int(grid_origin_x), int(grid_origin_y), grid_cols, grid_rows

def iteratively_refine_grid_to_fit_all_pieces(board_image, initial_grid, available_templates, optimal_scale):
    """
    CRITICAL: Iteratively refine the grid until all high-confidence pieces fit uniformly.
    This is the historically successful step that was missing.
    """
    print("🔄 ITERATIVE GRID REFINEMENT: Adjusting grid to fit all high-confidence pieces")
    
    cell_size, grid_origin_x, grid_origin_y, grid_cols, grid_rows = initial_grid
    board_height, board_width = board_image.shape[:2]
    
    # Get all high-confidence pieces at optimal scale
    all_high_confidence_pieces = []
    confidence_threshold = 0.35
    
    for template_name, template_path in available_templates.items():
        template = cv.imread(str(template_path), cv.IMREAD_GRAYSCALE)
        if template is None:
            continue
            
        # Scale template to optimal scale
        new_width = int(template.shape[1] * optimal_scale)
        new_height = int(template.shape[0] * optimal_scale)
        scaled_template = cv.resize(template, (new_width, new_height))
        
        # Match against board
        board_gray = cv.cvtColor(board_image, cv.COLOR_BGR2GRAY) if len(board_image.shape) == 3 else board_image
        result = cv.matchTemplate(board_gray, scaled_template, cv.TM_CCOEFF_NORMED)
        
        locations = np.where(result >= confidence_threshold)
        for pt in zip(*locations[::-1]):
            confidence = result[pt[1], pt[0]]
            center_x = pt[0] + scaled_template.shape[1] // 2
            center_y = pt[1] + scaled_template.shape[0] // 2
            all_high_confidence_pieces.append((center_x, center_y, template_name, confidence))
    
    print(f"   Found {len(all_high_confidence_pieces)} high-confidence pieces to fit")
    
    # Iterative refinement loop
    max_iterations = 20
    best_grid = (cell_size, grid_origin_x, grid_origin_y, grid_cols, grid_rows)
    best_fit_score = 0
    
    for iteration in range(max_iterations):
        # Test current grid against all pieces
        pieces_fit = 0
        total_error = 0
        
        for piece_x, piece_y, piece_name, piece_conf in all_high_confidence_pieces:
            # Find which grid cell this piece should be in
            col = round((piece_x - grid_origin_x - cell_size // 2) / cell_size)
            row = round((piece_y - grid_origin_y - cell_size // 2) / cell_size)
            
            # Check if within grid bounds
            if 0 <= col < grid_cols and 0 <= row < grid_rows:
                # Calculate expected position
                expected_x = grid_origin_x + col * cell_size + cell_size // 2
                expected_y = grid_origin_y + row * cell_size + cell_size // 2
                
                # Calculate error
                error = np.sqrt((piece_x - expected_x)**2 + (piece_y - expected_y)**2)
                
                if error <= cell_size * 0.3:  # Within 30% of cell size
                    pieces_fit += 1
                    total_error += error
        
        fit_score = pieces_fit - (total_error / 100)  # Pieces fit minus penalty for error
        
        if fit_score > best_fit_score:
            best_fit_score = fit_score
            best_grid = (cell_size, grid_origin_x, grid_origin_y, grid_cols, grid_rows)
        
        # Calculate adjustment for next iteration
        if iteration < max_iterations - 1:
            # Collect displacement vectors for pieces that don't fit well
            displacement_x = []
            displacement_y = []
            
            for piece_x, piece_y, piece_name, piece_conf in all_high_confidence_pieces:
                col = round((piece_x - grid_origin_x - cell_size // 2) / cell_size)
                row = round((piece_y - grid_origin_y - cell_size // 2) / cell_size)
                
                if 0 <= col < grid_cols and 0 <= row < grid_rows:
                    expected_x = grid_origin_x + col * cell_size + cell_size // 2
                    expected_y = grid_origin_y + row * cell_size + cell_size // 2
                    
                    displacement_x.append(piece_x - expected_x)
                    displacement_y.append(piece_y - expected_y)
            
            # Apply small adjustment based on average displacement
            if displacement_x and displacement_y:
                avg_disp_x = np.mean(displacement_x)
                avg_disp_y = np.mean(displacement_y)
                
                # Apply gradual adjustment
                adjustment_factor = 0.1
                grid_origin_x += avg_disp_x * adjustment_factor
                grid_origin_y += avg_disp_y * adjustment_factor
                
                # Keep within bounds
                grid_origin_x = max(0, min(grid_origin_x, board_width - grid_cols * cell_size))
                grid_origin_y = max(0, min(grid_origin_y, board_height - grid_rows * cell_size))
        
        if iteration % 5 == 0:
            print(f"   Iteration {iteration}: {pieces_fit}/{len(all_high_confidence_pieces)} pieces fit (score: {fit_score:.1f})")
    
    final_cell_size, final_origin_x, final_origin_y, final_cols, final_rows = best_grid
    fitted_pieces = sum(1 for piece_x, piece_y, _, _ in all_high_confidence_pieces 
                       if 0 <= round((piece_x - final_origin_x - final_cell_size // 2) / final_cell_size) < final_cols
                       and 0 <= round((piece_y - final_origin_y - final_cell_size // 2) / final_cell_size) < final_rows)
    
    print(f"   ✅ REFINEMENT COMPLETE: {fitted_pieces}/{len(all_high_confidence_pieces)} pieces fit uniformly")
    print(f"   Final grid: {final_cols}x{final_rows}, cell_size: {final_cell_size}")
    print(f"   Final origin: ({final_origin_x}, {final_origin_y})")
    
    return int(final_cell_size), int(final_origin_x), int(final_origin_y), final_cols, final_rows, all_high_confidence_pieces

def proven_grid_detection_methodology(board_image, available_templates):
    """
    Implement the historically successful grid detection methodology:
    1. Find two adjacent pieces with iterative scaling for extremely high confidence
    2. Calculate initial grid from these adjacent pieces  
    3. Iteratively refine grid until all high-confidence pieces fit uniformly
    """
    print("\n🎯 PROVEN GRID DETECTION METHODOLOGY")
    print("=" * 60)
    
    # Step 1: Find two adjacent pieces with scaling
    adjacent_result = find_two_adjacent_pieces_with_scaling(board_image, available_templates)
    if adjacent_result is None:
        return None, None, None, None, None, None
    
    piece1, piece2, distance, direction, optimal_scale = adjacent_result
    
    # Step 2: Calculate initial grid
    initial_grid = calculate_initial_grid_from_adjacent_pieces(board_image, piece1, piece2, distance, direction)
    
    # Step 3: Iteratively refine grid to fit all pieces
    final_grid_result = iteratively_refine_grid_to_fit_all_pieces(board_image, initial_grid, available_templates, optimal_scale)
    
    if final_grid_result is None:
        return None, None, None, None, None, None
        
    cell_size, grid_origin_x, grid_origin_y, grid_cols, grid_rows, all_pieces = final_grid_result
    
    return cell_size, grid_origin_x, grid_origin_y, grid_cols, grid_rows, all_pieces

def validate_proven_grid_detection(board_image, cell_size, grid_origin_x, grid_origin_y, all_pieces):
    """
    Create debug visualization showing proven grid detection results.
    """
    print("   🔍 Validating proven grid detection...")
    
    debug_image = board_image.copy()
    board_height, board_width = board_image.shape[:2]
    
    # Draw inferred grid
    cols = round(board_width / cell_size)
    rows = round(board_height / cell_size)
    
    # Draw vertical grid lines
    for i in range(cols + 1):
        x = int(grid_origin_x + i * cell_size)
        if 0 <= x < board_width:
            cv.line(debug_image, (x, 0), (x, board_height), (0, 255, 0), 2)  # Green lines
    
    # Draw horizontal grid lines  
    for j in range(rows + 1):
        y = int(grid_origin_y + j * cell_size)
        if 0 <= y < board_height:
            cv.line(debug_image, (0, y), (board_width, y), (0, 255, 0), 2)  # Green lines
    
    # Draw cell centers and grid coordinates
    for i in range(cols):
        for j in range(rows):
            center_x = int(grid_origin_x + i * cell_size + cell_size // 2)
            center_y = int(grid_origin_y + j * cell_size + cell_size // 2)
            if 0 <= center_x < board_width and 0 <= center_y < board_height:
                cv.circle(debug_image, (center_x, center_y), 3, (0, 255, 0), -1)  # Green dots
                cv.putText(debug_image, f"{i},{j}", (center_x-10, center_y-8), cv.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
    
    # Highlight all high-confidence pieces that fit in the grid
    pieces_fit = 0
    pieces_outside = 0
    
    for piece_x, piece_y, piece_name, piece_conf in all_pieces:
        # Determine grid position
        col = round((piece_x - grid_origin_x - cell_size // 2) / cell_size)
        row = round((piece_y - grid_origin_y - cell_size // 2) / cell_size)
        
        if 0 <= col < cols and 0 <= row < rows:
            # Piece fits in grid - green circle
            cv.circle(debug_image, (int(piece_x), int(piece_y)), 8, (0, 255, 0), 2)
            cv.putText(debug_image, f"{piece_name}({piece_conf:.2f})", (int(piece_x)+10, int(piece_y)-10), cv.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            pieces_fit += 1
            
            # Draw line to expected grid center
            expected_x = grid_origin_x + col * cell_size + cell_size // 2
            expected_y = grid_origin_y + row * cell_size + cell_size // 2
            cv.line(debug_image, (int(piece_x), int(piece_y)), (int(expected_x), int(expected_y)), (255, 255, 0), 1)  # Cyan line
        else:
            # Piece outside grid - red circle
            cv.circle(debug_image, (int(piece_x), int(piece_y)), 8, (0, 0, 255), 2)
            cv.putText(debug_image, f"{piece_name}({piece_conf:.2f})", (int(piece_x)+10, int(piece_y)-10), cv.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            pieces_outside += 1
    
    # Add text overlay with statistics
    fit_percentage = (pieces_fit / len(all_pieces)) * 100 if all_pieces else 0
    cv.putText(debug_image, f"PROVEN GRID: {cols}x{rows}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv.putText(debug_image, f"Cell Size: {cell_size}x{cell_size}", (10, 55), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv.putText(debug_image, f"Origin: ({grid_origin_x}, {grid_origin_y})", (10, 80), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv.putText(debug_image, f"Pieces Fit: {pieces_fit}/{len(all_pieces)} ({fit_percentage:.1f}%)", (10, 105), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Save debug image
    cv.imwrite("../../debug_proven_grid.png", debug_image)
    print(f"   📸 Proven grid validation saved to debug_proven_grid.png")
    print(f"   📊 Grid fit: {pieces_fit}/{len(all_pieces)} pieces ({fit_percentage:.1f}%)")
    
    return True

def detect_actual_board_boundary_from_edges(board_image):
    """
    Detect the actual game board boundary using the excellent edge detection.
    Returns (board_x, board_y, board_width, board_height) of the actual game area.
    """
    print("🔍 Detecting actual board boundary from edge detection...")
    
    # Use our excellent hue-focused edge detection
    edges = detect_hue_based_edges(board_image)
    
    # Find the largest rectangular region with strong edges (the game board)
    # Apply morphological operations to connect game board edges
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    edges_cleaned = cv.morphologyEx(edges, cv.MORPH_CLOSE, kernel)
    
    # Find contours
    contours, _ = cv.findContours(edges_cleaned, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        print("   ⚠️  No contours found, using full image")
        return 0, 0, board_image.shape[1], board_image.shape[0]
    
    # Find the largest contour that looks like a game board
    board_contour = None
    max_area = 0
    
    for contour in contours:
        # Approximate contour to reduce points
        epsilon = 0.02 * cv.arcLength(contour, True)
        approx = cv.approxPolyDP(contour, epsilon, True)
        
        area = cv.contourArea(contour)
        
        # Look for rectangular-ish contours (3-6 vertices) with significant area
        if len(approx) >= 4 and len(approx) <= 8 and area > max_area:
            # Must be at least 20% of image area to be considered
            if area > (board_image.shape[0] * board_image.shape[1] * 0.2):
                max_area = area
                board_contour = contour
    
    if board_contour is None:
        print("   ⚠️  No suitable board boundary found, using full image")
        return 0, 0, board_image.shape[1], board_image.shape[0]
    
    # Get bounding rectangle of the game board
    x, y, w, h = cv.boundingRect(board_contour)
    
    # Add small margin to ensure we don't cut off pieces at edges
    margin = 10
    x = max(0, x - margin)
    y = max(0, y - margin) 
    w = min(board_image.shape[1] - x, w + 2*margin)
    h = min(board_image.shape[0] - y, h + 2*margin)
    
    print(f"   ✅ Detected board boundary: ({x},{y}) size {w}x{h}")
    print(f"   Board area: {max_area:.0f} pixels ({max_area/(board_image.shape[0]*board_image.shape[1])*100:.1f}% of screenshot)")
    
    # Save debug image
    debug_image = board_image.copy()
    cv.drawContours(debug_image, [board_contour], -1, (0, 255, 0), 3)
    cv.rectangle(debug_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
    cv.imwrite("../../debug_board_boundary_detection.png", debug_image)
    print(f"   📸 Board boundary debug saved to debug_board_boundary_detection.png")
    
    return x, y, w, h

def find_extremely_high_confidence_pieces(board_image, board_x, board_y, board_w, board_h, available_templates):
    """
    Find high confidence pieces within the detected board boundary.
    Uses more reasonable thresholds for actual detection.
    """
    print("🎯 Finding high confidence pieces within detected board boundary...")
    
    # Extract board region
    board_region = board_image[board_y:board_y+board_h, board_x:board_x+board_w]
    
    # Process with excellent edge detection
    edge_board = create_multi_scale_edge_template(board_region)
    
    high_confidence_pieces = []
    scales_to_try = [0.4, 0.5, 0.6, 0.7, 0.8]  # Broader scale range
    HIGH_CONFIDENCE = 0.35  # LOWERED: More realistic threshold
    
    for template_name, template_path in available_templates.items():
        template = cv.imread(str(template_path), cv.IMREAD_GRAYSCALE)
        if template is None:
            continue
            
        print(f"   Testing template {template_name} with confidence threshold {HIGH_CONFIDENCE}")
        
        best_matches_for_template = []
        
        for scale in scales_to_try:
            # Scale template
            new_width = int(template.shape[1] * scale)
            new_height = int(template.shape[0] * scale)
            
            if new_width < 8 or new_height < 8 or new_width > 150 or new_height > 150:
                continue
                
            scaled_template = cv.resize(template, (new_width, new_height), interpolation=cv.INTER_CUBIC)
            
            # Template matching against edge-processed board region
            res = cv.matchTemplate(edge_board, scaled_template, cv.TM_CCOEFF_NORMED)
            
            # Find high confidence matches
            locations = np.where(res >= HIGH_CONFIDENCE)
            
            for pt in zip(*locations[::-1]):
                match_center_x = pt[0] + scaled_template.shape[1] // 2
                match_center_y = pt[1] + scaled_template.shape[0] // 2
                confidence = res[pt[1], pt[0]]
                
                # Convert back to full image coordinates
                full_x = board_x + match_center_x
                full_y = board_y + match_center_y
                
                best_matches_for_template.append((full_x, full_y, template_name, confidence, scale))
        
        # Sort by confidence and take reasonable number
        best_matches_for_template.sort(key=lambda x: x[3], reverse=True)
        high_confidence_pieces.extend(best_matches_for_template[:10])  # Top 10 per template
    
    # Remove duplicates with reasonable distance checking
    filtered_pieces = []
    for piece in high_confidence_pieces:
        is_duplicate = False
        for existing in filtered_pieces:
            distance = np.sqrt((piece[0] - existing[0])**2 + (piece[1] - existing[1])**2)
            if distance < 15:  # Reasonable duplicate detection
                if piece[3] > existing[3]:  # Keep higher confidence
                    filtered_pieces.remove(existing)
                else:
                    is_duplicate = True
                    break
        if not is_duplicate:
            filtered_pieces.append(piece)
    
    # Sort by confidence and take reasonable number
    filtered_pieces.sort(key=lambda x: x[3], reverse=True)
    final_pieces = filtered_pieces[:20]  # Top 20 pieces maximum
    
    print(f"   ✅ Found {len(final_pieces)} high-confidence pieces (threshold: {HIGH_CONFIDENCE})")
    for i, (x, y, name, conf, scale) in enumerate(final_pieces[:8]):  # Show top 8
        print(f"      {i+1}. {name} at ({x},{y}) confidence={conf:.3f} scale={scale:.2f}")
    
    return final_pieces

def analyze_board_shape_from_pieces(high_confidence_pieces, board_x, board_y, board_w, board_h):
    """
    Analyze the actual board shape from piece positions to handle irregular boards.
    """
    print("🔍 Analyzing actual board shape from piece positions...")
    
    if len(high_confidence_pieces) < 5:
        print("   ⚠️  Not enough pieces to analyze board shape")
        return None
    
    # Get piece positions relative to board
    piece_positions = [(p[0] - board_x, p[1] - board_y) for p in high_confidence_pieces]
    
    # Find the actual occupied area
    x_coords = [pos[0] for pos in piece_positions]
    y_coords = [pos[1] for pos in piece_positions]
    
    # Find min/max bounds of actual pieces
    min_x, max_x = min(x_coords), max(x_coords)
    min_y, max_y = min(y_coords), max(y_coords)
    
    # Calculate spacing between pieces to estimate cell size
    x_diffs = []
    y_diffs = []
    
    for i, pos1 in enumerate(piece_positions):
        for j, pos2 in enumerate(piece_positions[i+1:], i+1):
            dx = abs(pos2[0] - pos1[0])
            dy = abs(pos2[1] - pos1[1])
            
            # Look for adjacent pieces
            if dy < 10 and 20 < dx < 80:  # Horizontal neighbors
                x_diffs.append(dx)
            elif dx < 10 and 20 < dy < 80:  # Vertical neighbors
                y_diffs.append(dy)
    
    if not x_diffs and not y_diffs:
        print("   ⚠️  Could not find adjacent pieces to estimate cell size")
        return None
    
    # Estimate cell size
    if x_diffs and y_diffs:
        cell_size = (np.median(x_diffs) + np.median(y_diffs)) / 2
    elif x_diffs:
        cell_size = np.median(x_diffs)
    else:
        cell_size = np.median(y_diffs)
    
    # Estimate grid dimensions from occupied area
    occupied_width = max_x - min_x + cell_size
    occupied_height = max_y - min_y + cell_size
    
    estimated_cols = round(occupied_width / cell_size)
    estimated_rows = round(occupied_height / cell_size)
    
    print(f"   📐 Occupied area: {occupied_width:.1f}x{occupied_height:.1f}")
    print(f"   📐 Estimated cell size: {cell_size:.1f}")
    print(f"   📐 Estimated grid: {estimated_cols}x{estimated_rows}")
    
    # Calculate grid origin
    grid_origin_x = board_x + min_x - (min_x % cell_size)
    grid_origin_y = board_y + min_y - (min_y % cell_size)
    
    return {
        'cell_size': int(cell_size),
        'grid_origin_x': int(grid_origin_x),
        'grid_origin_y': int(grid_origin_y), 
        'estimated_cols': estimated_cols,
        'estimated_rows': estimated_rows,
        'occupied_bounds': (min_x, min_y, max_x, max_y)
    }

def create_adaptive_grid_debug_visualization(board_image, board_x, board_y, board_w, board_h, 
                                           grid_info, high_confidence_pieces):
    """
    Create comprehensive debug visualization showing board boundary, detected pieces, and adaptive grid.
    """
    print("   🔍 Creating adaptive grid debug visualization...")
    
    debug_image = board_image.copy()
    
    # Draw board boundary in green
    cv.rectangle(debug_image, (board_x, board_y), (board_x + board_w, board_y + board_h), (0, 255, 0), 3)
    
    if grid_info is not None:
        cell_size = grid_info['cell_size']
        grid_origin_x = grid_info['grid_origin_x']
        grid_origin_y = grid_info['grid_origin_y']
        estimated_cols = grid_info['estimated_cols']
        estimated_rows = grid_info['estimated_rows']
        
        # Draw adaptive grid lines in blue
        for i in range(estimated_cols + 1):
            x = grid_origin_x + i * cell_size
            if board_x <= x <= board_x + board_w:
                cv.line(debug_image, (x, board_y), (x, board_y + board_h), (255, 0, 0), 2)
        
        for j in range(estimated_rows + 1):
            y = grid_origin_y + j * cell_size
            if board_y <= y <= board_y + board_h:
                cv.line(debug_image, (board_x, y), (board_x + board_w, y), (255, 0, 0), 2)
        
        # Draw cell centers and grid coordinates
        for i in range(estimated_cols):
            for j in range(estimated_rows):
                center_x = grid_origin_x + i * cell_size + cell_size // 2
                center_y = grid_origin_y + j * cell_size + cell_size // 2
                
                # Only draw if within board boundary
                if (board_x <= center_x <= board_x + board_w and 
                    board_y <= center_y <= board_y + board_h):
                    cv.circle(debug_image, (center_x, center_y), 3, (255, 0, 0), -1)
                    cv.putText(debug_image, f"{i},{j}", (center_x-8, center_y-8), 
                             cv.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
    
    # Draw detected high-confidence pieces
    for i, (x, y, name, conf, scale) in enumerate(high_confidence_pieces):
        # Color based on confidence: high=green, medium=yellow, low=red
        if conf > 0.6:
            color = (0, 255, 0)  # Green
        elif conf > 0.4:
            color = (0, 255, 255)  # Yellow
        else:
            color = (0, 0, 255)  # Red
            
        cv.circle(debug_image, (int(x), int(y)), 8, color, 2)
        cv.putText(debug_image, f"{name}({conf:.2f})", (int(x)+10, int(y)-10), 
                  cv.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    # Add info overlay
    overlay_y = 30
    if grid_info is not None:
        cv.putText(debug_image, f"ADAPTIVE GRID: {grid_info['estimated_cols']}x{grid_info['estimated_rows']}", 
                  (10, overlay_y), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        overlay_y += 25
        cv.putText(debug_image, f"Cell Size: {grid_info['cell_size']}x{grid_info['cell_size']}", 
                  (10, overlay_y), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        overlay_y += 25
        cv.putText(debug_image, f"Origin: ({grid_info['grid_origin_x']}, {grid_info['grid_origin_y']})", 
                  (10, overlay_y), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        overlay_y += 25
    
    cv.putText(debug_image, f"High-Conf Pieces: {len(high_confidence_pieces)}", 
              (10, overlay_y), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Save debug image
    cv.imwrite("../../debug_adaptive_grid_detection.png", debug_image)
    print(f"   📸 Adaptive grid debug saved to debug_adaptive_grid_detection.png")
    
    return True

def improved_grid_detection_methodology(board_image, available_templates):
    """
    IMPROVED: Adaptive grid detection that handles irregular board shapes.
    1. Detect actual board boundary using excellent edge detection
    2. Find high confidence pieces within board area (reasonable threshold)
    3. Analyze actual board shape from piece positions
    4. Create adaptive grid that fits the detected pieces
    5. Generate comprehensive debug visualization
    """
    print("\n🎯 IMPROVED ADAPTIVE GRID DETECTION METHODOLOGY")
    print("=" * 60)
    
    # Step 1: Detect actual board boundary
    board_x, board_y, board_w, board_h = detect_actual_board_boundary_from_edges(board_image)
    
    # Step 2: Find high confidence pieces within board area (reasonable threshold)
    high_conf_pieces = find_extremely_high_confidence_pieces(
        board_image, board_x, board_y, board_w, board_h, available_templates
    )
    
    if len(high_conf_pieces) < 5:
        print("   ❌ Insufficient high confidence pieces found for adaptive grid")
        return None, None, None, None, None, None
    
    # Step 3: Analyze actual board shape from piece positions
    grid_info = analyze_board_shape_from_pieces(high_conf_pieces, board_x, board_y, board_w, board_h)
    
    if grid_info is None:
        print("   ❌ Could not analyze board shape from pieces")
        return None, None, None, None, None, None
    
    # Step 4: Extract grid parameters
    cell_size = grid_info['cell_size']
    grid_origin_x = grid_info['grid_origin_x'] 
    grid_origin_y = grid_info['grid_origin_y']
    detected_cols = grid_info['estimated_cols']
    detected_rows = grid_info['estimated_rows']
    
    # Step 5: Generate comprehensive debug visualization
    create_adaptive_grid_debug_visualization(
        board_image, board_x, board_y, board_w, board_h, grid_info, high_conf_pieces
    )
    
    print(f"   ✅ ADAPTIVE GRID DETECTION SUCCESSFUL!")
    print(f"   📐 Detected grid: {detected_cols}x{detected_rows}")
    print(f"   📐 Cell size: {cell_size}x{cell_size}")  
    print(f"   📍 Grid origin: ({grid_origin_x}, {grid_origin_y})")
    print(f"   🎯 Based on {len(high_conf_pieces)} high-confidence pieces")
    
    return cell_size, grid_origin_x, grid_origin_y, detected_cols, detected_rows, high_conf_pieces

def calculate_grid_dimensions_from_board_and_pieces(board_w, board_h, high_confidence_pieces):
    """
    Calculate grid dimensions from actual board size and piece spacing.
    This is the correct approach - use board area, not screenshot area.
    """
    print("📐 Calculating grid dimensions from actual board area and piece spacing...")
    print(f"   Actual board area: {board_w}x{board_h}")
    
    if len(high_confidence_pieces) < 3:
        print("   ⚠️  Need at least 3 pieces to determine grid dimensions")
        return None, None
    
    # Analyze piece spacing to determine cell size
    pieces_positions = [(p[0], p[1]) for p in high_confidence_pieces]
    
    # Calculate distances between all piece pairs
    horizontal_distances = []
    vertical_distances = []
    
    for i, pos1 in enumerate(pieces_positions):
        for j, pos2 in enumerate(pieces_positions[i+1:], i+1):
            dx = abs(pos2[0] - pos1[0])
            dy = abs(pos2[1] - pos1[1])
            
            # Look for likely adjacent pieces (horizontal or vertical)
            if dy < 15 and 30 < dx < 80:  # Horizontal neighbors
                horizontal_distances.append(dx)
            elif dx < 15 and 30 < dy < 80:  # Vertical neighbors
                vertical_distances.append(dy)
    
    if not horizontal_distances and not vertical_distances:
        print("   ⚠️  Could not find adjacent pieces to determine cell size")
        return None, None
    
    # Find most common spacing (cell size)
    def find_most_common_distance(distances):
        if not distances:
            return None
        
        # Group similar distances
        groups = []
        for dist in distances:
            added = False
            for group in groups:
                if abs(dist - np.mean(group)) <= 3:  # Within 3 pixels
                    group.append(dist)
                    added = True
                    break
            if not added:
                groups.append([dist])
        
        # Return average of largest group
        largest_group = max(groups, key=len)
        return np.mean(largest_group)
    
    cell_width = find_most_common_distance(horizontal_distances)
    cell_height = find_most_common_distance(vertical_distances)
    
    # Use available spacing to estimate cell size
    if cell_width is not None and cell_height is not None:
        cell_size = (cell_width + cell_height) / 2
        print(f"   Cell size from spacing: width={cell_width:.1f}, height={cell_height:.1f}, average={cell_size:.1f}")
    elif cell_width is not None:
        cell_size = cell_width
        print(f"   Cell size from horizontal spacing: {cell_size:.1f}")
    elif cell_height is not None:
        cell_size = cell_height
        print(f"   Cell size from vertical spacing: {cell_size:.1f}")
    else:
        print("   ⚠️  Could not determine cell size from piece spacing")
        return None, None
    
    # Calculate grid dimensions from BOARD SIZE, not screenshot size
    detected_cols = round(board_w / cell_size)
    detected_rows = round(board_h / cell_size)
    
    print(f"   🎯 CALCULATED GRID DIMENSIONS: {detected_cols} cols × {detected_rows} rows")
    print(f"   Based on: board size {board_w}x{board_h} ÷ cell size {cell_size:.1f}")
    
    # Validate reasonable dimensions
    if detected_cols < 3 or detected_cols > 12 or detected_rows < 3 or detected_rows > 15:
        print(f"   ⚠️  Unreasonable grid dimensions: {detected_cols}x{detected_rows}")
        return None, None
    
    return detected_cols, detected_rows, int(cell_size)

def proven_grid_detection_methodology(board_image, available_templates):
    """
    CORRECTED: Proven methodology with proper board boundary detection and high confidence matching.
    1. Detect actual board boundary using excellent edge detection
    2. Find extremely high confidence pieces within board area
    3. Calculate grid dimensions from board size and piece spacing
    4. Find adjacent pieces for grid origin
    5. Iteratively refine grid
    """
    print("\n🎯 CORRECTED PROVEN GRID DETECTION METHODOLOGY")
    print("=" * 60)
    
    # Step 1: Detect actual board boundary
    board_x, board_y, board_w, board_h = detect_actual_board_boundary_from_edges(board_image)
    
    # Step 2: Find extremely high confidence pieces within board area
    high_conf_pieces = find_extremely_high_confidence_pieces(
        board_image, board_x, board_y, board_w, board_h, available_templates
    )
    
    if len(high_conf_pieces) < 3:
        print("   ❌ Insufficient extremely high confidence pieces found")
        return None, None, None, None, None, None
    
    # Step 3: Calculate grid dimensions from BOARD area and piece spacing
    grid_result = calculate_grid_dimensions_from_board_and_pieces(board_w, board_h, high_conf_pieces)
    if grid_result is None:
        print("   ❌ Could not calculate grid dimensions")
        return None, None, None, None, None, None
    
    detected_cols, detected_rows, cell_size = grid_result
    
    # Step 4: Find adjacent pieces to establish grid origin
    piece1, piece2 = None, None
    min_distance = float('inf')
    
    for i, p1 in enumerate(high_conf_pieces):
        for j, p2 in enumerate(high_conf_pieces[i+1:], i+1):
            dx = abs(p2[0] - p1[0])
            dy = abs(p2[1] - p1[1])
            distance = np.sqrt(dx**2 + dy**2)
            
            # Look for pieces approximately cell_size apart
            tolerance = cell_size * 0.2  # 20% tolerance
            if abs(distance - cell_size) < tolerance and distance < min_distance:
                min_distance = distance
                piece1, piece2 = p1, p2
    
    if piece1 is None:
        print("   ⚠️  Could not find adjacent pieces for grid origin")
        # Use top-left piece as reference
        piece1 = min(high_conf_pieces, key=lambda p: p[0] + p[1])
        grid_origin_x = board_x + (piece1[0] - board_x) % cell_size - cell_size // 2
        grid_origin_y = board_y + (piece1[1] - board_y) % cell_size - cell_size // 2
    else:
        print(f"   ✅ Found adjacent pieces: {piece1[2]} and {piece2[2]} (distance: {min_distance:.1f})")
        # Calculate grid origin from adjacent pieces
        ref_piece = piece1 if piece1[0] + piece1[1] < piece2[0] + piece2[1] else piece2
        rel_x = ref_piece[0] - board_x  # Position relative to board
        rel_y = ref_piece[1] - board_y
        
        # Calculate grid origin
        grid_origin_x = board_x + (rel_x % cell_size) - cell_size // 2
        grid_origin_y = board_y + (rel_y % cell_size) - cell_size // 2
    
    # Ensure grid origin is within board
    grid_origin_x = max(board_x, min(grid_origin_x, board_x + board_w - detected_cols * cell_size))
    grid_origin_y = max(board_y, min(grid_origin_y, board_y + board_h - detected_rows * cell_size))
    
    print(f"   📍 Grid origin: ({grid_origin_x}, {grid_origin_y})")
    print(f"   📏 Final grid: {detected_cols}x{detected_rows}, cell size: {cell_size}")
    
    return cell_size, grid_origin_x, grid_origin_y, detected_cols, detected_rows, high_conf_pieces

def detect_actual_board_cells_from_edge_image(edge_processed_board, cell_size, grid_origin_x, grid_origin_y, detected_cols, detected_rows):
    """
    Analyze the edge-processed board to determine which grid cells actually contain content.
    Returns a boolean array indicating which cells exist.
    """
    print("🔍 Analyzing edge-detected board to find which cells actually exist...")
    
    board_height, board_width = edge_processed_board.shape[:2]
    cell_exists = [[False for _ in range(detected_cols)] for _ in range(detected_rows)]
    content_threshold = 10  # Minimum edge pixels to consider a cell as having content
    
    for row in range(detected_rows):
        for col in range(detected_cols):
            # Calculate cell boundaries
            cell_left = grid_origin_x + col * cell_size
            cell_top = grid_origin_y + row * cell_size
            cell_right = min(cell_left + cell_size, board_width)
            cell_bottom = min(cell_top + cell_size, board_height)
            
            # Ensure cell is within image bounds
            if (cell_left >= 0 and cell_top >= 0 and 
                cell_right <= board_width and cell_bottom <= board_height):
                
                # Extract cell region from edge-processed board
                cell_region = edge_processed_board[cell_top:cell_bottom, cell_left:cell_right]
                
                # Count edge pixels (white pixels in edge image)
                edge_pixels = np.sum(cell_region == 255)
                
                # Cell exists if it has enough edge content
                if edge_pixels >= content_threshold:
                    cell_exists[row][col] = True
                    print(f"   Cell ({col},{row}): {edge_pixels} edge pixels - EXISTS")
                else:
                    print(f"   Cell ({col},{row}): {edge_pixels} edge pixels - EMPTY")
    
    # Count existing cells
    total_existing = sum(sum(1 for exists in row if exists) for row in cell_exists)
    total_possible = detected_rows * detected_cols
    
    print(f"   ✅ Found {total_existing}/{total_possible} cells that actually exist")
    
    return cell_exists

def ensure_identical_edge_detection_for_templates_and_board(image, template_mode=False):
    """
    CRITICAL: Use identical edge detection parameters for both templates and board.
    This ensures template matching confidence is reliable.
    """
    if template_mode:
        print("   🎨 Processing template with IDENTICAL edge detection...")
    else:
        print("   🎨 Processing board with IDENTICAL edge detection...")
    
    # IDENTICAL PARAMETERS for both templates and board
    BLUR_KERNEL = 5
    CANNY_LOW = 80  
    CANNY_HIGH = 160
    DILATE_ITERATIONS = 1
    
    # Step 1: Enhanced color contrast (identical for both)
    enhanced = enhance_color_contrast(image)
    
    # Step 2: Hue-based edge detection (identical parameters)
    edges = detect_hue_based_edges(enhanced, BLUR_KERNEL, CANNY_LOW, CANNY_HIGH)
    
    # Step 3: Morphological operations (identical)
    if DILATE_ITERATIONS > 0:
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2, 2))
        edges_processed = cv.dilate(edges, kernel, iterations=DILATE_ITERATIONS)
    else:
        edges_processed = edges
    
    # Step 4: Invert so edges are white on black background (identical)
    edges_inverted = cv.bitwise_not(edges_processed)
    
    return edges_inverted

def process_board_for_identical_edge_matching(board_image):
    """
    Convert the live board using IDENTICAL edge detection as templates.
    """
    global edge_processed_board
    print("🎨 Processing board with IDENTICAL edge detection as templates...")
    edge_processed_board = ensure_identical_edge_detection_for_templates_and_board(board_image, template_mode=False)
    
    # Save debug image
    debug_path = "../../identical_edge_board_capture.png"
    cv.imwrite(debug_path, edge_processed_board)
    print(f"📸 Debug: Board with identical edge processing saved to {debug_path}")
    
    return edge_processed_board

def improved_grid_detection_with_cell_analysis(board_image, available_templates):
    """
    IMPROVED: Detect grid AND analyze which cells actually exist.
    1. Use boundary detection and piece analysis to get grid structure
    2. Use edge-processed board to determine which cells actually contain content
    3. Only consider existing cells for moves and template matching
    """
    print("\n🎯 IMPROVED GRID DETECTION WITH CELL EXISTENCE ANALYSIS")
    print("=" * 70)
    
    # Step 1: Detect actual board boundary
    board_x, board_y, board_w, board_h = detect_actual_board_boundary_from_edges(board_image)
    
    # Step 2: Process board with IDENTICAL edge detection as templates
    edge_processed_board = process_board_for_identical_edge_matching(board_image)
    
    # Step 3: Find high confidence pieces within board area
    high_conf_pieces = find_extremely_high_confidence_pieces(
        board_image, board_x, board_y, board_w, board_h, available_templates
    )
    
    if len(high_conf_pieces) < 3:
        print("   ❌ Insufficient high confidence pieces found")
        return None
    
    # Step 4: Calculate grid dimensions from board size and piece spacing  
    grid_result = calculate_grid_dimensions_from_board_and_pieces(board_w, board_h, high_conf_pieces)
    if grid_result is None:
        print("   ❌ Could not calculate grid dimensions")
        return None
    
    detected_cols, detected_rows, cell_size = grid_result
    
    # Step 5: Calculate grid origin
    piece1 = min(high_conf_pieces, key=lambda p: p[0] + p[1])  # Top-left piece
    rel_x = piece1[0] - board_x
    rel_y = piece1[1] - board_y
    grid_origin_x = board_x + (rel_x % cell_size) - cell_size // 2
    grid_origin_y = board_y + (rel_y % cell_size) - cell_size // 2
    
    # Ensure grid origin is within board
    grid_origin_x = max(board_x, min(grid_origin_x, board_x + board_w - detected_cols * cell_size))
    grid_origin_y = max(board_y, min(grid_origin_y, board_y + board_h - detected_rows * cell_size))
    
    # Step 6: CRITICAL - Analyze which cells actually exist
    cell_exists = detect_actual_board_cells_from_edge_image(
        edge_processed_board, cell_size, grid_origin_x, grid_origin_y, detected_cols, detected_rows
    )
    
    # Step 7: Generate comprehensive debug visualization
    create_cell_existence_debug_visualization(
        board_image, edge_processed_board, board_x, board_y, board_w, board_h,
        cell_size, grid_origin_x, grid_origin_y, detected_cols, detected_rows,
        cell_exists, high_conf_pieces
    )
    
    print(f"   ✅ GRID DETECTION WITH CELL ANALYSIS SUCCESSFUL!")
    print(f"   📐 Detected grid: {detected_cols}x{detected_rows}")
    print(f"   📐 Cell size: {cell_size}x{cell_size}")
    print(f"   📍 Grid origin: ({grid_origin_x}, {grid_origin_y})")
    
    return {
        'cell_size': cell_size,
        'grid_origin_x': grid_origin_x, 
        'grid_origin_y': grid_origin_y,
        'detected_cols': detected_cols,
        'detected_rows': detected_rows,
        'cell_exists': cell_exists,
        'high_conf_pieces': high_conf_pieces
    }

def create_cell_existence_debug_visualization(board_image, edge_processed_board, board_x, board_y, board_w, board_h,
                                           cell_size, grid_origin_x, grid_origin_y, detected_cols, detected_rows,
                                           cell_exists, high_conf_pieces):
    """
    Create debug visualization showing which cells actually exist vs empty space.
    """
    print("   🔍 Creating cell existence debug visualization...")
    
    debug_image = board_image.copy()
    
    # Draw board boundary in green
    cv.rectangle(debug_image, (board_x, board_y), (board_x + board_w, board_y + board_h), (0, 255, 0), 3)
    
    # Draw grid with different colors for existing vs non-existing cells
    for row in range(detected_rows):
        for col in range(detected_cols):
            cell_left = grid_origin_x + col * cell_size
            cell_top = grid_origin_y + row * cell_size
            cell_right = cell_left + cell_size
            cell_bottom = cell_top + cell_size
            center_x = cell_left + cell_size // 2
            center_y = cell_top + cell_size // 2
            
            if cell_exists[row][col]:
                # Existing cell - blue rectangle and white center dot
                cv.rectangle(debug_image, (cell_left, cell_top), (cell_right, cell_bottom), (255, 0, 0), 2)
                cv.circle(debug_image, (center_x, center_y), 3, (255, 255, 255), -1)
                cv.putText(debug_image, f"{col},{row}", (center_x-8, center_y-8), 
                          cv.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
            else:
                # Non-existing cell - red X
                cv.rectangle(debug_image, (cell_left, cell_top), (cell_right, cell_bottom), (0, 0, 255), 1)
                cv.line(debug_image, (cell_left, cell_top), (cell_right, cell_bottom), (0, 0, 255), 2)
                cv.line(debug_image, (cell_right, cell_top), (cell_left, cell_bottom), (0, 0, 255), 2)
    
    # Draw detected high-confidence pieces
    for i, (x, y, name, conf, scale) in enumerate(high_conf_pieces):
        color = (0, 255, 255)  # Yellow for pieces
        cv.circle(debug_image, (int(x), int(y)), 8, color, 2)
        cv.putText(debug_image, f"{name}({conf:.2f})", (int(x)+10, int(y)-10), 
                  cv.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    # Add info overlay
    existing_cells = sum(sum(1 for exists in row if exists) for row in cell_exists)
    total_cells = detected_rows * detected_cols
    
    cv.putText(debug_image, f"CELL ANALYSIS: {existing_cells}/{total_cells} cells exist", 
              (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv.putText(debug_image, f"Grid: {detected_cols}x{detected_rows}, Cell Size: {cell_size}x{cell_size}", 
              (10, 55), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv.putText(debug_image, f"Origin: ({grid_origin_x}, {grid_origin_y})", 
              (10, 80), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv.putText(debug_image, f"High-Conf Pieces: {len(high_conf_pieces)}", 
              (10, 105), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Save debug images
    cv.imwrite("../../debug_cell_existence_analysis.png", debug_image)
    
    # Also save the edge-processed board for comparison
    cv.imwrite("../../debug_edge_processed_for_cell_analysis.png", edge_processed_board)
    
    print(f"   📸 Cell existence analysis saved to debug_cell_existence_analysis.png")
    print(f"   📸 Edge-processed board saved to debug_edge_processed_for_cell_analysis.png")
    
    return True

def find_all_occurences_into_mainArray_with_cell_analysis(filename,template_name,color=(0,0,255),custom_threshold=None):
    """
    Find template matches using IDENTICAL EDGE DETECTION and CELL EXISTENCE constraints.
    Only places pieces in cells that actually exist on the board.
    """
    #assume template is smaller than single "block"/rectangle with desired object
    #modifies given image
    FullGridImage = FullGridImageOriginal #given this image  | full grid

    global dim
    dim = FullGridImage.shape
    
    # Use IDENTICAL edge-processed board
    global edge_processed_board
    if edge_processed_board is None:
        edge_processed_board = ensure_identical_edge_detection_for_templates_and_board(FullGridImage, template_mode=False)
    
    # Use CELL EXISTENCE constraints
    global true_cell_size, grid_offset_x, grid_offset_y, cell_exists_grid
    
    if true_cell_size > 0:
        # Use cell-analysis-aware grid coordinates
        tempArrayW, tempArrayH = calculate_anchor_based_grid_coordinates(grid_offset_x, grid_offset_y, true_cell_size)
        grid_status = "CELL-ANALYSIS-BASED"
        cell_info = f"cell_size: {true_cell_size} (from cell existence analysis)"
    else:
        print("⚠️  Grid not calibrated - using fallback square calculation")
        # Fallback to square grid calculation
        cellW = dim[1]//cols
        cellH = dim[0]//rows
        cell_size = min(cellW, cellH)  # Force square cells
        
        tempArrayW = []
        tempArrayH = []
        for i in range(cols):
            tempArrayW.append( (cell_size * i) + (cell_size//2) )
        for j in range(rows):
             tempArrayH.append( (cell_size * j) + (cell_size//2) )
        
        grid_status = "FALLBACK-SQUARE"
        cell_info = f"cell_size: {cell_size} (fallback calculation)"

    # Load template - handle both color and grayscale templates
    template_image = cv.imread(filename, cv.IMREAD_COLOR)  # Load as color
    if template_image is None:
        print(f"⚠️  Could not load template: {filename}")
        return
    
    # Check if template is effectively grayscale (edge-processed templates)
    if len(template_image.shape) == 3:
        # Check if all channels are the same (grayscale saved as color)
        b, g, r = cv.split(template_image)
        if np.array_equal(b, g) and np.array_equal(g, r):
            # Template is grayscale, use it directly
            template_processed = b  # Use one channel
        else:
            # Template is color, apply edge detection
            template_processed = ensure_identical_edge_detection_for_templates_and_board(template_image, template_mode=True)
    else:
        # Template is already grayscale
        template_processed = template_image
    
    # Use the globally determined optimal scale for this puzzle
    global optimal_template_scale
    if optimal_template_scale > 0:
        new_width = int(template_processed.shape[1] * optimal_template_scale)
        new_height = int(template_processed.shape[0] * optimal_template_scale)
        template_processed = cv.resize(template_processed, (new_width, new_height), interpolation=cv.INTER_CUBIC)
    
    w, h = template_processed.shape[::-1]
    
    # IDENTICAL Edge vs Edge matching
    res = cv.matchTemplate(edge_processed_board, template_processed, cv.TM_CCOEFF_NORMED)
    
    # Use custom threshold if provided, otherwise use lower default for identical edge matching
    threshold = custom_threshold if custom_threshold is not None else 0.25
    loc = np.where( res >= threshold)
    
    # Count matches and get confidence scores
    match_count = len(loc[0])
    max_confidence = np.max(res) if res.size > 0 else 0
    template_size = f"{template_processed.shape[1]}x{template_processed.shape[0]}"
    
    # Show grid calibration status in output
    print(f"   Template {template_name}: {match_count} matches (max confidence: {max_confidence:.3f}, threshold: {threshold:.3f}) [size: {template_size}] {grid_status}")
    
    if match_count > 0:
        print(f"   🔍 CELL-ANALYSIS grid mapping debug for {template_name}:")
        print(f"      Cell centers W: {tempArrayW[:3]}...{tempArrayW[-3:]} ({len(tempArrayW)} total)")
        print(f"      Cell centers H: {tempArrayH[:3]}...{tempArrayH[-3:]} ({len(tempArrayH)} total)")
        print(f"      {cell_info}, Grid offsets: ({grid_offset_x}, {grid_offset_y})")
    
    match_debug_count = 0
    successful_mappings = 0
    
    for pt in zip(*loc[::-1]):
        if(debug):
            cv.rectangle(FullGridImage, pt, (pt[0] + w, pt[1] + h), color, 2)
            cv.rectangle(FullGridImage, ((pt[0] + w//2)-1, (pt[1] + h//2)-1), ((pt[0] + w//2)+1, (pt[1] + h//2)+1), color, 2)
        
        #pt is top left point eg.(106, 5)
        match_center_x = pt[0] + w//2
        match_center_y = pt[1] + h//2
        nearestW = find_nearest(tempArrayW, match_center_x)
        nearestH = find_nearest(tempArrayH, match_center_y)
        
        # Debug first few matches with distance info
        if match_debug_count < 3 and match_count <= 50:  # Only debug if reasonable match count
            expected_x = tempArrayW[nearestW]
            expected_y = tempArrayH[nearestH]
            distance_x = abs(match_center_x - expected_x)
            distance_y = abs(match_center_y - expected_y)
            print(f"      Match {match_debug_count+1}: center({match_center_x},{match_center_y}) → grid({nearestW},{nearestH}) [distance: ({distance_x},{distance_y})]")
            match_debug_count += 1
        
        if(debug):
            print(match_center_x, match_center_y, "    |   ",nearestW,nearestH,"   ",template_name)
        
        # CRITICAL: Only assign if cell exists AND is empty
        if (nearestH < len(cell_exists_grid) and nearestW < len(cell_exists_grid[0]) and
            cell_exists_grid[nearestH][nearestW] and mainArray[nearestH][nearestW] == ""):
            mainArray[nearestH][nearestW] = template_name  # Store color letter instead of number
            successful_mappings += 1
        elif nearestH < len(cell_exists_grid) and nearestW < len(cell_exists_grid[0]) and not cell_exists_grid[nearestH][nearestW]:
            # Skip - cell doesn't exist
            pass
    
    if match_count > 0:
        print(f"      ✅ Successfully mapped {successful_mappings}/{match_count} matches to existing grid cells")
        
    if(debug):
        cv.imwrite('./screenshots/ress.png',FullGridImage)

def printMainArrayWithCellAnalysis():
    """Print the board state showing which cells exist vs don't exist."""
    global cell_exists_grid
    print("📋 Current board (X = non-existent cell):")
    for i in range(rows):
        row_str = "   "
        for j in range(cols):
            if not cell_exists_grid[i][j]:
                row_str += "X "  # Non-existent cell
            elif mainArray[i][j] == "":
                row_str += ". "  # Empty but existing cell
            else:
                row_str += f"{mainArray[i][j]} "  # Piece in existing cell
        print(row_str)

# Initialize global variable for cell existence tracking
cell_exists_grid = []

def systematic_grid_detection(board_image, available_templates, available_powerups=None):
    """
    Systematic grid detection following 8-step process (using ONLY piece templates):
    1. Use screenshot boundaries as exact grid bounds
    2. Search for high confidence match starting at 60% scale
    3. Increment scale by 10% if no matches found
    4. Optimize scale for found match
    5. Find adjacent match to establish grid spacing
    6. Infer complete grid structure
    7. Find distant matches to refine positioning
    8. Iterate until 30%+ cells have matches
    """
    print("🔍 SYSTEMATIC GRID DETECTION:")
    
    # Step 1: Use screenshot boundaries as exact grid bounds
    board_h, board_w = board_image.shape[:2]
    board_x, board_y = 0, 0
    print(f"   📐 Board boundaries: {board_w}x{board_h} pixels")
    
    # Use ONLY piece templates for grid detection (exclude powerups to avoid confusion with UI elements)
    all_templates = available_templates.copy()
    print(f"   🎯 Using {len(all_templates)} piece templates for grid detection (excluding powerups)")
    
    if not all_templates:
        print("   ❌ No piece templates available for detection")
        return None
    
    # Step 2-4: Find first high confidence match with dynamic scaling
    first_match = find_first_high_confidence_match_systematic(board_image, all_templates)
    if not first_match:
        print("   ❌ No high confidence matches found")
        return None
    
    template_name, (match_x, match_y), confidence, optimal_scale, template_size = first_match
    print(f"   ✅ First match: {template_name} at ({match_x},{match_y}) confidence={confidence:.3f} scale={optimal_scale:.3f}")
    
    # Step 5: Find adjacent match to establish grid spacing
    adjacent_match = find_adjacent_match_systematic(board_image, all_templates, first_match)
    if not adjacent_match:
        print("   ❌ No adjacent match found to establish grid")
        return None
    
    adj_template_name, (adj_x, adj_y), adj_confidence, adj_scale, direction = adjacent_match
    grid_spacing = calculate_spacing_from_matches_systematic(first_match, adjacent_match)
    print(f"   ✅ Adjacent match: {adj_template_name} at ({adj_x},{adj_y}) direction={direction}")
    print(f"   📏 Grid spacing: {grid_spacing} pixels")
    
    # Step 6: Infer complete grid structure
    grid_info = infer_complete_grid_structure_systematic(board_w, board_h, first_match, adjacent_match, grid_spacing)
    if not grid_info:
        print("   ❌ Failed to infer grid structure")
        return None
    
    cell_size, grid_cols, grid_rows, grid_origin_x, grid_origin_y = grid_info
    print(f"   📊 Grid: {grid_cols}x{grid_rows}, cell_size={cell_size}, origin=({grid_origin_x},{grid_origin_y})")
    
    # Step 7-8: Find distant matches and refine grid iteratively
    distant_matches = []  # Initialize for debug visualization
    refined_grid = refine_grid_with_distant_matches_systematic(board_image, all_templates, grid_info, optimal_scale)
    if refined_grid:
        cell_size, grid_cols, grid_rows, grid_origin_x, grid_origin_y, distant_matches = refined_grid
        print(f"   🔧 Refined grid: {grid_cols}x{grid_rows}, cell_size={cell_size}, origin=({grid_origin_x},{grid_origin_y})")
    
    # Validate final grid has sufficient matches
    # Use refined grid coordinates (first 5 values) if available, otherwise original grid_info
    final_grid_info = refined_grid[:5] if refined_grid else grid_info
    final_matches = count_matches_in_grid_systematic(board_image, all_templates, final_grid_info, optimal_scale)
    match_percentage = (final_matches / (grid_cols * grid_rows)) * 100
    print(f"   📊 Final validation: {final_matches}/{grid_cols * grid_rows} cells ({match_percentage:.1f}%) have matches")
    
    if match_percentage < 30:
        print("   ⚠️  Less than 30% cells have matches - grid may be inaccurate")
    
    # Create debug visualization ONLY on first successful detection
    global systematic_debug_created
    if not systematic_debug_created and match_percentage >= 30:
        systematic_debug_created = True
        grid_data = {
            'cell_size': cell_size,
            'grid_cols': grid_cols,
            'grid_rows': grid_rows, 
            'grid_origin_x': grid_origin_x,
            'grid_origin_y': grid_origin_y,
            'match_percentage': match_percentage
        }
        create_systematic_grid_debug_visualization(board_image, grid_data, first_match, adjacent_match, 
                                                 distant_matches, optimal_scale) 
    
    # Return in the same format as the old function for compatibility
    return {
        'cell_size': cell_size,
        'grid_cols': grid_cols, 
        'grid_rows': grid_rows,
        'grid_origin_x': grid_origin_x,
        'grid_origin_y': grid_origin_y,
        'detected_cols': grid_cols,  # For compatibility
        'detected_rows': grid_rows,  # For compatibility
        'cell_exists': [[True for _ in range(grid_cols)] for _ in range(grid_rows)],  # Assume all exist for now
        'high_conf_pieces': [],  # For compatibility
        'optimal_scale': optimal_scale,
        'match_percentage': match_percentage,
        'board_w': board_w,
        'board_h': board_h
    }

def find_first_high_confidence_match_systematic(board_image, all_templates, min_confidence=0.5):
    """Steps 2-4: Find first high confidence match with dynamic scaling"""
    print("   🔍 Searching for first high confidence match...")
    
    best_match = None
    best_confidence = 0
    
    # Process board for template matching
    processed_board = ensure_identical_edge_detection_for_templates_and_board(board_image, template_mode=False)
    
    for template_name, template_path in all_templates.items():
        # Load and process template
        template = cv.imread(str(template_path), cv.IMREAD_GRAYSCALE)
        if template is None:
            continue
            
        processed_template = ensure_identical_edge_detection_for_templates_and_board(template, template_mode=True)
        original_h, original_w = processed_template.shape
        
        # Start with 60% scale, increment by 10%
        for scale_percent in range(60, 101, 10):
            scale = scale_percent / 100.0
            new_w = int(original_w * scale)
            new_h = int(original_h * scale)
            
            if new_w < 5 or new_h < 5:  # Too small
                continue
                
            scaled_template = cv.resize(processed_template, (new_w, new_h))
            
            # Template matching
            result = cv.matchTemplate(processed_board, scaled_template, cv.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)
            
            if max_val > best_confidence and max_val >= min_confidence:
                # Optimize scale around this match
                optimized_scale, optimized_confidence, optimized_pos = optimize_scale_for_match_systematic(
                    processed_board, processed_template, max_loc, scale, max_val
                )
                
                if optimized_confidence > best_confidence:
                    best_match = (template_name, optimized_pos, optimized_confidence, optimized_scale, (new_w, new_h))
                    best_confidence = optimized_confidence
                    
                    print(f"      🎯 Better match: {template_name} confidence={optimized_confidence:.3f} scale={optimized_scale:.3f}")
    
    return best_match

def optimize_scale_for_match_systematic(processed_board, original_template, base_pos, base_scale, base_confidence):
    """Optimize scale around a found match to maximize confidence"""
    best_scale = base_scale
    best_confidence = base_confidence
    best_pos = base_pos
    
    # Test scales around the base scale (±20% in 2% increments)
    for scale_offset in range(-20, 21, 2):
        test_scale = base_scale * (1 + scale_offset / 100.0)
        if test_scale <= 0.2 or test_scale >= 1.5:  # Reasonable scale bounds
            continue
            
        original_h, original_w = original_template.shape
        new_w = int(original_w * test_scale)
        new_h = int(original_h * test_scale)
        
        if new_w < 5 or new_h < 5:
            continue
            
        scaled_template = cv.resize(original_template, (new_w, new_h))
        
        # Search in small area around base position
        search_x1 = max(0, base_pos[0] - 10)
        search_y1 = max(0, base_pos[1] - 10) 
        search_x2 = min(processed_board.shape[1] - new_w, base_pos[0] + 10)
        search_y2 = min(processed_board.shape[0] - new_h, base_pos[1] + 10)
        
        if search_x2 <= search_x1 or search_y2 <= search_y1:
            continue
            
        search_area = processed_board[search_y1:search_y2+new_h, search_x1:search_x2+new_w]
        result = cv.matchTemplate(search_area, scaled_template, cv.TM_CCOEFF_NORMED)
        
        if result.size > 0:
            min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)
            actual_pos = (search_x1 + max_loc[0], search_y1 + max_loc[1])
            
            if max_val > best_confidence:
                best_scale = test_scale
                best_confidence = max_val
                best_pos = actual_pos
    
    return best_scale, best_confidence, best_pos

def find_adjacent_match_systematic(board_image, all_templates, first_match):
    """Step 5: Find adjacent match to establish grid spacing"""
    template_name, (match_x, match_y), confidence, scale, template_size = first_match
    processed_board = ensure_identical_edge_detection_for_templates_and_board(board_image, template_mode=False)
    
    # Expected cell size based on template size and scale
    expected_cell_size = int(max(template_size) * 1.2)  # Assume some padding
    
    # Search directions: right, down, left, up
    search_directions = [
        ('right', (expected_cell_size, 0)),
        ('down', (0, expected_cell_size)),
        ('left', (-expected_cell_size, 0)),
        ('up', (0, -expected_cell_size))
    ]
    
    best_adjacent = None
    best_confidence = 0
    
    for direction, (dx, dy) in search_directions:
        search_x = match_x + dx
        search_y = match_y + dy
        
        # Search in area around expected position
        for template_name2, template_path in all_templates.items():
            template = cv.imread(str(template_path), cv.IMREAD_GRAYSCALE)
            if template is None:
                continue
                
            processed_template = ensure_identical_edge_detection_for_templates_and_board(template, template_mode=True)
            original_h, original_w = processed_template.shape
            
            # Use similar scale as first match
            new_w = int(original_w * scale)
            new_h = int(original_h * scale)
            
            if new_w < 5 or new_h < 5:
                continue
                
            scaled_template = cv.resize(processed_template, (new_w, new_h))
            
            # Search in area around expected position
            search_radius = expected_cell_size // 2
            search_x1 = max(0, search_x - search_radius)
            search_y1 = max(0, search_y - search_radius)
            search_x2 = min(processed_board.shape[1] - new_w, search_x + search_radius)
            search_y2 = min(processed_board.shape[0] - new_h, search_y + search_radius)
            
            if search_x2 <= search_x1 or search_y2 <= search_y1:
                continue
                
            search_area = processed_board[search_y1:search_y2+new_h, search_x1:search_x2+new_w]
            result = cv.matchTemplate(search_area, scaled_template, cv.TM_CCOEFF_NORMED)
            
            if result.size > 0:
                min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)
                
                if max_val > best_confidence and max_val >= 0.45:  # Higher threshold for adjacent matches
                    actual_pos = (search_x1 + max_loc[0], search_y1 + max_loc[1])
                    best_adjacent = (template_name2, actual_pos, max_val, scale, direction)
                    best_confidence = max_val
    
    return best_adjacent

def calculate_spacing_from_matches_systematic(first_match, adjacent_match):
    """Calculate grid spacing from two adjacent matches"""
    _, (x1, y1), _, _, _ = first_match
    _, (x2, y2), _, _, direction = adjacent_match
    
    if direction in ['right', 'left']:
        return abs(x2 - x1)
    else:  # up, down
        return abs(y2 - y1)

def infer_complete_grid_structure_systematic(board_w, board_h, first_match, adjacent_match, grid_spacing):
    """Step 6: Infer complete grid structure from two matches"""
    _, (x1, y1), _, scale, template_size = first_match
    _, (x2, y2), _, _, direction = adjacent_match
    
    # Use grid spacing as cell size
    cell_size = grid_spacing
    
    # VALIDATION: Reject grids with cells smaller than 30 pixels (likely detecting noise/obstacles)
    if cell_size < 30:
        print(f"   ❌ Rejected grid: cell size {cell_size} too small (minimum 30 pixels)")
        return None
    
    # Determine initial grid origin from first match position
    if direction == 'right':
        initial_grid_origin_x = x1
        initial_grid_origin_y = min(y1, y2)
    elif direction == 'down':
        initial_grid_origin_x = min(x1, x2)
        initial_grid_origin_y = y1
    elif direction == 'left':
        initial_grid_origin_x = x2
        initial_grid_origin_y = min(y1, y2)
    else:  # up
        initial_grid_origin_x = min(x1, x2)
        initial_grid_origin_y = y2
    
    # Calculate how many cells fit in board dimensions from initial position
    initial_grid_cols = max(1, (board_w - initial_grid_origin_x) // cell_size)
    initial_grid_rows = max(1, (board_h - initial_grid_origin_y) // cell_size)
    
    # Calculate total grid dimensions that would fit in the board
    max_possible_cols = board_w // cell_size
    max_possible_rows = board_h // cell_size
    
    # Use the smaller of initial calculation and maximum possible
    grid_cols = min(initial_grid_cols, max_possible_cols)
    grid_rows = min(initial_grid_rows, max_possible_rows)
    
    # GRID SCALING & CENTERING: Scale grid to perfectly align with at least one axis, then center
    total_grid_width = grid_cols * cell_size
    total_grid_height = grid_rows * cell_size
    
    # Calculate scale factors to fit perfectly on each axis
    width_scale = board_w / total_grid_width
    height_scale = board_h / total_grid_height
    
    # Use the smaller scale to ensure grid fits in bounds, but at least one axis will be perfectly aligned
    final_scale = min(width_scale, height_scale)
    
    # Apply scaling to cell size and recalculate grid dimensions
    scaled_cell_size = int(cell_size * final_scale)
    
    # Recalculate how many cells fit with the new cell size
    final_grid_cols = min(grid_cols, board_w // scaled_cell_size)
    final_grid_rows = min(grid_rows, board_h // scaled_cell_size)
    
    # Calculate final grid dimensions
    final_grid_width = final_grid_cols * scaled_cell_size
    final_grid_height = final_grid_rows * scaled_cell_size
    
    # Center the scaled grid within the board bounds
    centered_grid_origin_x = (board_w - final_grid_width) // 2
    centered_grid_origin_y = (board_h - final_grid_height) // 2
    
    # Ensure centering doesn't push grid out of bounds
    centered_grid_origin_x = max(0, centered_grid_origin_x)
    centered_grid_origin_y = max(0, centered_grid_origin_y)
    
    print(f"   📐 Grid scaling: {cell_size}px → {scaled_cell_size}px (scale: {final_scale:.3f})")
    print(f"   📐 Grid centering: {initial_grid_origin_x},{initial_grid_origin_y} → {centered_grid_origin_x},{centered_grid_origin_y}")
    print(f"   📐 Final grid: {final_grid_cols}x{final_grid_rows} filling {final_grid_width}x{final_grid_height} of {board_w}x{board_h}")
    
    return scaled_cell_size, final_grid_cols, final_grid_rows, centered_grid_origin_x, centered_grid_origin_y

def refine_grid_with_distant_matches_systematic(board_image, all_templates, grid_info, optimal_scale):
    """Steps 7-8: Find distant matches and refine grid positioning"""
    cell_size, grid_cols, grid_rows, grid_origin_x, grid_origin_y = grid_info
    processed_board = ensure_identical_edge_detection_for_templates_and_board(board_image, template_mode=False)
    
    # Find matches in grid cells, focusing on corners and edges
    distant_positions = [
        (0, 0),  # top-left
        (grid_cols-1, 0),  # top-right  
        (0, grid_rows-1),  # bottom-left
        (grid_cols-1, grid_rows-1),  # bottom-right
        (grid_cols//2, grid_rows//2)  # center
    ]
    
    found_matches = []
    
    for grid_col, grid_row in distant_positions:
        if grid_col >= grid_cols or grid_row >= grid_rows:
            continue
            
        cell_center_x = grid_origin_x + grid_col * cell_size + cell_size // 2
        cell_center_y = grid_origin_y + grid_row * cell_size + cell_size // 2
        
        # Search for matches in this cell
        best_match = find_best_match_in_area_systematic(
            processed_board, all_templates, 
            cell_center_x, cell_center_y, cell_size, optimal_scale
        )
        
        if best_match:
            template_name, (match_x, match_y), confidence = best_match
            found_matches.append({
                'grid_pos': (grid_col, grid_row),
                'expected_pos': (cell_center_x, cell_center_y),
                'actual_pos': (match_x + cell_size//2, match_y + cell_size//2),  # Center of match
                'confidence': confidence,
                'template': template_name
            })
    
    if len(found_matches) >= 2:
        print(f"   🔧 Found {len(found_matches)} distant matches for grid refinement")
        
        # Calculate average offset between expected and actual positions
        x_offsets = [match['actual_pos'][0] - match['expected_pos'][0] for match in found_matches]
        y_offsets = [match['actual_pos'][1] - match['expected_pos'][1] for match in found_matches]
        
        avg_x_offset = sum(x_offsets) / len(x_offsets)
        avg_y_offset = sum(y_offsets) / len(y_offsets)
        
        # Adjust grid origin
        refined_origin_x = grid_origin_x + int(avg_x_offset)
        refined_origin_y = grid_origin_y + int(avg_y_offset)
        
        print(f"      📍 Grid origin adjustment: ({avg_x_offset:.1f}, {avg_y_offset:.1f})")
        
        return cell_size, grid_cols, grid_rows, refined_origin_x, refined_origin_y, found_matches
    
    return None  # No refinement

def find_best_match_in_area_systematic(processed_board, all_templates, center_x, center_y, search_radius, scale):
    """Find best template match in a specific area"""
    best_match = None
    best_confidence = 0
    
    for template_name, template_path in all_templates.items():
        template = cv.imread(str(template_path), cv.IMREAD_GRAYSCALE)
        if template is None:
            continue
            
        processed_template = ensure_identical_edge_detection_for_templates_and_board(template, template_mode=True)
        original_h, original_w = processed_template.shape
        
        new_w = int(original_w * scale)
        new_h = int(original_h * scale)
        
        if new_w < 5 or new_h < 5:
            continue
            
        scaled_template = cv.resize(processed_template, (new_w, new_h))
        
        # Define search area
        search_x1 = max(0, center_x - search_radius//2)
        search_y1 = max(0, center_y - search_radius//2)
        search_x2 = min(processed_board.shape[1] - new_w, center_x + search_radius//2)
        search_y2 = min(processed_board.shape[0] - new_h, center_y + search_radius//2)
        
        if search_x2 <= search_x1 or search_y2 <= search_y1:
            continue
            
        search_area = processed_board[search_y1:search_y2+new_h, search_x1:search_x2+new_w]
        result = cv.matchTemplate(search_area, scaled_template, cv.TM_CCOEFF_NORMED)
        
        if result.size > 0:
            min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)
            
            if max_val > best_confidence and max_val >= 0.45:
                actual_pos = (search_x1 + max_loc[0], search_y1 + max_loc[1])
                best_match = (template_name, actual_pos, max_val)
                best_confidence = max_val
    
    return best_match

def count_matches_in_grid_systematic(board_image, all_templates, grid_info, optimal_scale):
    """Count how many grid cells contain template matches"""
    cell_size, grid_cols, grid_rows, grid_origin_x, grid_origin_y = grid_info
    processed_board = ensure_identical_edge_detection_for_templates_and_board(board_image, template_mode=False)
    
    match_count = 0
    
    for grid_row in range(grid_rows):
        for grid_col in range(grid_cols):
            cell_center_x = grid_origin_x + grid_col * cell_size + cell_size // 2
            cell_center_y = grid_origin_y + grid_row * cell_size + cell_size // 2
            
            match = find_best_match_in_area_systematic(
                processed_board, all_templates,
                cell_center_x, cell_center_y, cell_size, optimal_scale
            )
            
            if match and match[2] >= 0.45:  # 45% confidence threshold
                match_count += 1
    
    return match_count

def create_systematic_grid_debug_visualization(board_image, grid_info, first_match, adjacent_match, distant_matches, optimal_scale):
    """
    Create debug visualization showing systematic grid detection results.
    """
    print("   🔍 Creating systematic grid debug visualization...")
    
    debug_image = board_image.copy()
    cell_size = grid_info['cell_size']
    grid_cols = grid_info['grid_cols'] 
    grid_rows = grid_info['grid_rows']
    grid_origin_x = grid_info['grid_origin_x']
    grid_origin_y = grid_info['grid_origin_y']
    
    # Draw grid overlay
    for row in range(grid_rows):
        for col in range(grid_cols):
            cell_left = grid_origin_x + col * cell_size
            cell_top = grid_origin_y + row * cell_size
            cell_right = cell_left + cell_size
            cell_bottom = cell_top + cell_size
            center_x = cell_left + cell_size // 2
            center_y = cell_top + cell_size // 2
            
            # Draw grid cell - blue rectangles
            cv.rectangle(debug_image, (cell_left, cell_top), (cell_right, cell_bottom), (255, 0, 0), 1)
            
            # Draw cell coordinates
            cv.putText(debug_image, f"{col},{row}", (center_x-8, center_y+3), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.25, (255, 255, 255), 1)
    
    # Highlight the first match that established the grid (bright green)
    if first_match:
        template_name, (match_x, match_y), confidence, scale, template_size = first_match
        cv.circle(debug_image, (int(match_x + template_size[0]//2), int(match_y + template_size[1]//2)), 12, (0, 255, 0), 3)
        cv.putText(debug_image, f"1st: {template_name}({confidence:.2f})", 
                  (int(match_x), int(match_y)-15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Highlight the adjacent match that established spacing (bright yellow)
    if adjacent_match:
        adj_template_name, (adj_x, adj_y), adj_confidence, adj_scale, direction = adjacent_match
        # Calculate template size from scale
        template_size = (int(64 * adj_scale), int(64 * adj_scale))  # Assume 64x64 base template
        cv.circle(debug_image, (int(adj_x + template_size[0]//2), int(adj_y + template_size[1]//2)), 12, (0, 255, 255), 3)
        cv.putText(debug_image, f"2nd: {adj_template_name}({adj_confidence:.2f})", 
                  (int(adj_x), int(adj_y)-15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        # Draw line showing the spacing relationship
        if first_match:
            first_center = (int(first_match[1][0] + first_match[4][0]//2), int(first_match[1][1] + first_match[4][1]//2))
            adj_center = (int(adj_x + template_size[0]//2), int(adj_y + template_size[1]//2))
            cv.line(debug_image, first_center, adj_center, (255, 255, 0), 2)
            
            # Add spacing text
            mid_x = (first_center[0] + adj_center[0]) // 2
            mid_y = (first_center[1] + adj_center[1]) // 2
            cv.putText(debug_image, f"spacing: {cell_size}px", (mid_x-30, mid_y-5), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
    
    # Show distant matches used for refinement (orange)
    if distant_matches:
        for i, match in enumerate(distant_matches):
            if isinstance(match, dict) and 'actual_pos' in match:
                x, y = match['actual_pos']
                conf = match['confidence']
                cv.circle(debug_image, (int(x), int(y)), 8, (0, 165, 255), 2)
                cv.putText(debug_image, f"R{i+1}({conf:.2f})", (int(x)+10, int(y)-10), 
                          cv.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1)
    
    # Add comprehensive info overlay
    match_percentage = grid_info.get('match_percentage', 0)
    cv.putText(debug_image, f"SYSTEMATIC GRID DETECTION", 
              (10, 25), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv.putText(debug_image, f"Grid: {grid_cols}x{grid_rows} ({grid_cols * grid_rows} cells)", 
              (10, 50), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv.putText(debug_image, f"Cell Size: {cell_size}x{cell_size} pixels", 
              (10, 75), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv.putText(debug_image, f"Origin: ({grid_origin_x}, {grid_origin_y})", 
              (10, 100), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv.putText(debug_image, f"Template Scale: {optimal_scale:.3f}", 
              (10, 125), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv.putText(debug_image, f"Validation: {match_percentage:.1f}% cells matched", 
              (10, 150), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if match_percentage >= 30 else (0, 255, 255), 2)
    
    # Add legend
    cv.putText(debug_image, "Legend:", (10, 190), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv.circle(debug_image, (20, 210), 8, (0, 255, 0), 3)
    cv.putText(debug_image, "First Match", (35, 215), cv.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv.circle(debug_image, (20, 230), 8, (0, 255, 255), 3)
    cv.putText(debug_image, "Adjacent Match", (35, 235), cv.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv.circle(debug_image, (20, 250), 6, (0, 165, 255), 2)
    cv.putText(debug_image, "Refinement Matches", (35, 255), cv.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    # Save debug image
    cv.imwrite("../../debug_systematic_grid_detection.png", debug_image)
    print(f"   📸 Systematic grid detection saved to debug_systematic_grid_detection.png")
    
    return True

def find_all_occurences_into_mainArray_with_cell_analysis(filename,template_name,color=(0,0,255),custom_threshold=None):
    """
    Find template matches using IDENTICAL EDGE DETECTION and CELL EXISTENCE constraints.
    Only places pieces in cells that actually exist on the board.
    """
    #assume template is smaller than single "block"/rectangle with desired object
    #modifies given image
    FullGridImage = FullGridImageOriginal #given this image  | full grid

    global dim
    dim = FullGridImage.shape
    
    # Use IDENTICAL edge-processed board
    global edge_processed_board
    if edge_processed_board is None:
        edge_processed_board = ensure_identical_edge_detection_for_templates_and_board(FullGridImage, template_mode=False)
    
    # Use CELL EXISTENCE constraints
    global true_cell_size, grid_offset_x, grid_offset_y, cell_exists_grid
    
    if true_cell_size > 0:
        # Use cell-analysis-aware grid coordinates
        tempArrayW, tempArrayH = calculate_anchor_based_grid_coordinates(grid_offset_x, grid_offset_y, true_cell_size)
        grid_status = "CELL-ANALYSIS-BASED"
        cell_info = f"cell_size: {true_cell_size} (from cell existence analysis)"
    else:
        print("⚠️  Grid not calibrated - using fallback square calculation")
        # Fallback to square grid calculation
        cellW = dim[1]//cols
        cellH = dim[0]//rows
        cell_size = min(cellW, cellH)  # Force square cells
        
        tempArrayW = []
        tempArrayH = []
        for i in range(cols):
            tempArrayW.append( (cell_size * i) + (cell_size//2) )
        for j in range(rows):
             tempArrayH.append( (cell_size * j) + (cell_size//2) )
        
        grid_status = "FALLBACK-SQUARE"
        cell_info = f"cell_size: {cell_size} (fallback calculation)"

    # Load template - handle both color and grayscale templates
    template_image = cv.imread(filename, cv.IMREAD_COLOR)  # Load as color
    if template_image is None:
        print(f"⚠️  Could not load template: {filename}")
        return
    
    # Check if template is effectively grayscale (edge-processed templates)
    if len(template_image.shape) == 3:
        # Check if all channels are the same (grayscale saved as color)
        b, g, r = cv.split(template_image)
        if np.array_equal(b, g) and np.array_equal(g, r):
            # Template is grayscale, use it directly
            template_processed = b  # Use one channel
        else:
            # Template is color, apply edge detection
            template_processed = ensure_identical_edge_detection_for_templates_and_board(template_image, template_mode=True)
    else:
        # Template is already grayscale
        template_processed = template_image
    
    # Use the globally determined optimal scale for this puzzle
    global optimal_template_scale
    if optimal_template_scale > 0:
        new_width = int(template_processed.shape[1] * optimal_template_scale)
        new_height = int(template_processed.shape[0] * optimal_template_scale)
        template_processed = cv.resize(template_processed, (new_width, new_height), interpolation=cv.INTER_CUBIC)
    
    w, h = template_processed.shape[::-1]
    
    # IDENTICAL Edge vs Edge matching
    res = cv.matchTemplate(edge_processed_board, template_processed, cv.TM_CCOEFF_NORMED)
    
    # Use custom threshold if provided, otherwise use lower default for identical edge matching
    threshold = custom_threshold if custom_threshold is not None else 0.25
    loc = np.where( res >= threshold)
    
    # Count matches and get confidence scores
    match_count = len(loc[0])
    max_confidence = np.max(res) if res.size > 0 else 0
    template_size = f"{template_processed.shape[1]}x{template_processed.shape[0]}"
    
    # Show grid calibration status in output
    print(f"   Template {template_name}: {match_count} matches (max confidence: {max_confidence:.3f}, threshold: {threshold:.3f}) [size: {template_size}] {grid_status}")
    
    if match_count > 0:
        print(f"   🔍 CELL-ANALYSIS grid mapping debug for {template_name}:")
        print(f"      Cell centers W: {tempArrayW[:3]}...{tempArrayW[-3:]} ({len(tempArrayW)} total)")
        print(f"      Cell centers H: {tempArrayH[:3]}...{tempArrayH[-3:]} ({len(tempArrayH)} total)")
        print(f"      {cell_info}, Grid offsets: ({grid_offset_x}, {grid_offset_y})")
    
    match_debug_count = 0
    successful_mappings = 0
    
    for pt in zip(*loc[::-1]):
        if(debug):
            cv.rectangle(FullGridImage, pt, (pt[0] + w, pt[1] + h), color, 2)
            cv.rectangle(FullGridImage, ((pt[0] + w//2)-1, (pt[1] + h//2)-1), ((pt[0] + w//2)+1, (pt[1] + h//2)+1), color, 2)
        
        #pt is top left point eg.(106, 5)
        match_center_x = pt[0] + w//2
        match_center_y = pt[1] + h//2
        nearestW = find_nearest(tempArrayW, match_center_x)
        nearestH = find_nearest(tempArrayH, match_center_y)
        
        # Debug first few matches with distance info
        if match_debug_count < 3 and match_count <= 50:  # Only debug if reasonable match count
            expected_x = tempArrayW[nearestW]
            expected_y = tempArrayH[nearestH]
            distance_x = abs(match_center_x - expected_x)
            distance_y = abs(match_center_y - expected_y)
            print(f"      Match {match_debug_count+1}: center({match_center_x},{match_center_y}) → grid({nearestW},{nearestH}) [distance: ({distance_x},{distance_y})]")
            match_debug_count += 1
        
        if(debug):
            print(match_center_x, match_center_y, "    |   ",nearestW,nearestH,"   ",template_name)
        
        # CRITICAL: Only assign if cell exists AND is empty
        if (nearestH < len(cell_exists_grid) and nearestW < len(cell_exists_grid[0]) and
            cell_exists_grid[nearestH][nearestW] and mainArray[nearestH][nearestW] == ""):
            mainArray[nearestH][nearestW] = template_name  # Store color letter instead of number
            successful_mappings += 1
        elif nearestH < len(cell_exists_grid) and nearestW < len(cell_exists_grid[0]) and not cell_exists_grid[nearestH][nearestW]:
            # Skip - cell doesn't exist
            pass
    
    if match_count > 0:
        print(f"      ✅ Successfully mapped {successful_mappings}/{match_count} matches to existing grid cells")
        
    if(debug):
        cv.imwrite('./screenshots/ress.png',FullGridImage)

def printMainArrayWithCellAnalysis():
    """Print the board state showing which cells exist vs don't exist."""
    global cell_exists_grid
    print("📋 Current board (X = non-existent cell):")
    for i in range(rows):
        row_str = "   "
        for j in range(cols):
            if not cell_exists_grid[i][j]:
                row_str += "X "  # Non-existent cell
            elif mainArray[i][j] == "":
                row_str += ". "  # Empty but existing cell
            else:
                row_str += f"{mainArray[i][j]} "  # Piece in existing cell
        print(row_str)

# Initialize global variable for cell existence tracking
cell_exists_grid = []

def main():
    """Main game loop with improved UX."""
    global cols, rows, mainArray, failed_moves_history  # Declare globals at start
    
    print("🎮 Enhanced Match-3 Bot")
    print("=" * 50)
    
    # Load templates first
    game_name = get_target_game()
    if not game_name:
        print("\n❌ Game selection failed!")
        sys.exit(1)
    
    available_templates = load_templates(game_name)
    if not available_templates:
        print("\n❌ Template loading failed!")
        sys.exit(1)
    
    # Load powerup templates
    available_powerups = load_powerup_templates(game_name)
    
    print(f"\n🎯 Bot configured for: {game_name.upper()}")
    print(f"📋 Loaded {len(available_templates)} piece templates: {', '.join(sorted(available_templates.keys()))}")
    if available_powerups:
        powerup_priorities = sorted(available_powerups.keys(), key=lambda x: int(x) if x.isdigit() else 0, reverse=True)
        print(f"⚡ Loaded {len(available_powerups)} powerup templates: {', '.join(powerup_priorities)} (priority order)")
    else:
        print("ℹ️  No powerup templates available")
    
    # Get coordinates
    coords = getGridLocation(game_name)
    if None in coords:
        print("❌ Coordinate setup failed!")
        sys.exit(1)
    
    gridX1, gridY1, gridX2, gridY2 = coords
    
    # Take test screenshot
    print("\n🧪 Taking test screenshot...")
    test_img = gridScreenshot("test.png", gridX1, gridY1, gridX2, gridY2)
    print("📸 Test screenshot taken")
    
    # Confirm before starting
    if not input("Does the setup look correct? Start bot? (y/n): ").lower().startswith('y'):
        print("❌ Setup cancelled")
        sys.exit(1)
    
    print("\n🚀 Starting automated gameplay!")
    print("🛡️  SAFETY: Move mouse to top-left corner to emergency stop")
    print("⏰ Bot analyzes board every 7 seconds")
    print()
    
    # Define colors for template visualization (cycling through these)
    debug_colors = [
        (0,0,255),    # Red
        (0,255,0),    # Green  
        (255,0,255),  # Magenta
        (255,0,0),    # Blue
        (255,255,0),  # Cyan
        (0,255,255),  # Yellow
        (128,0,128),  # Purple
        (255,165,0),  # Orange
    ]
    
    move_count = 0
    global previous_board_state, stuck_move_count, optimal_template_scale
    
    # Grid detection will run every cycle for fresh accuracy
    
    puzzle_scale_determined = False  # Only determine scale once per puzzle
    optimal_thresholds = {}  # Store optimal thresholds per template
    
    try:
        while True:
            print(f"\n🔄 Cycle {move_count + 1}")
            
            # Take screenshot and setup
            gridImage = gridScreenshot("current.png", gridX1, gridY1, gridX2, gridY2)
            global FullGridImageOriginal
            FullGridImageOriginal = cv.cvtColor(np.array(gridImage), cv.COLOR_RGB2BGR)

            # CRITICAL: Process board for IDENTICAL edge detection before template matching
            process_board_for_identical_edge_matching(FullGridImageOriginal)

            # Determine optimal template scale for this puzzle (ONLY ONCE per puzzle)
            if not puzzle_scale_determined:
                print("🎯 Determining optimal template scale for this puzzle...")
                print("   This only happens once per puzzle session...")
                
                # Test with the first available template to find best scale
                first_template_name = min(available_templates.keys())
                test_template_path = available_templates[first_template_name]
                optimal_template_scale, best_confidence = find_optimal_template_scale(
                    str(test_template_path), FullGridImageOriginal
                )
                
                print(f"✅ Optimal template scale determined: {optimal_template_scale:.2f} (confidence: {best_confidence:.3f})")
                
                # Now find optimal thresholds for each template using IDENTICAL EDGE DETECTION
                # Combine piece and powerup templates for threshold optimization
                all_templates = available_templates.copy()
                all_templates.update(available_powerups)
                optimal_thresholds = find_optimal_thresholds(all_templates, FullGridImageOriginal)
                
                print("   🎯 Identical edge-based scale and threshold optimization complete!")
                puzzle_scale_determined = True  # Never run scale detection again for this puzzle

            # CRITICAL: SYSTEMATIC Grid Detection following 8-step process (EVERY CYCLE for accuracy)
            global true_cell_size, grid_offset_x, grid_offset_y, cell_exists_grid
            # Use the new systematic methodology from user requirements
            grid_result = systematic_grid_detection(FullGridImageOriginal, available_templates, available_powerups)
                
            if grid_result is not None:
                    cell_size = grid_result['cell_size']
                    grid_origin_x = grid_result['grid_origin_x']
                    grid_origin_y = grid_result['grid_origin_y']
                    detected_cols = grid_result['detected_cols']
                    detected_rows = grid_result['detected_rows']
                    cell_exists = grid_result['cell_exists']
                    all_pieces = grid_result['high_conf_pieces']
                    
                    # Set global grid dimensions from detection
                    cols, rows = detected_cols, detected_rows
                    
                    print(f"   🎯 GRID WITH CELL ANALYSIS SET: {cols} cols × {rows} rows")
                    
                    # Count existing cells
                    existing_cells = sum(sum(1 for exists in row if exists) for row in cell_exists)
                    print(f"   📊 {existing_cells}/{cols*rows} cells actually exist on the board")
                    
                    # Reinitialize mainArray with detected dimensions
                    mainArray = []
                    for i in range(rows):
                        row = []
                        for j in range(cols):
                            row.append("")
                        mainArray.append(row)
                    
                    # Store the grid parameters
                    true_cell_size = cell_size
                    grid_offset_x = grid_origin_x
                    grid_offset_y = grid_origin_y
                    cell_exists_grid = cell_exists  # Store which cells exist
                    
                    print("✅ GRID DETECTION WITH CELL ANALYSIS SUCCESSFUL!")
                    print(f"   Found {len(all_pieces)} high-confidence pieces")
                    print(f"   Detected grid: {cols}x{rows}, {existing_cells} cells exist")
                    print(f"   Cell size: {cell_size}x{cell_size}")
                    print(f"   Grid origin: ({grid_origin_x}, {grid_origin_y})")
                    print("   All subsequent template matching uses this ANALYZED grid")
            else:
                    print("⚠️  Grid detection with cell analysis failed - using fallback")
                    # Fallback: assume common match-3 dimensions
                    cols, rows = 7, 6  # Updated fallback for candy crush
                    print(f"   🔧 FALLBACK GRID: {cols} cols × {rows} rows")
                    
                    # Initialize mainArray with fallback dimensions
                    mainArray = []
                    for i in range(rows):
                        row = []
                        for j in range(cols):
                            row.append("")
                        mainArray.append(row)
                    
                    cell_exists_grid = [[True for _ in range(cols)] for _ in range(rows)]  # Assume all exist
                
            print()

            # Analyze board dimensions with CELL EXISTENCE info
            board_height, board_width = FullGridImageOriginal.shape[:2]
            fallback_cellW = board_width // cols
            fallback_cellH = board_height // rows
            
            print(f"🔍 GRID WITH CELL ANALYSIS:")
            print(f"   Board dimensions: {board_width}x{board_height}")
            print(f"   Grid: {cols}x{rows} = {cols*rows} total possible cells")
            if true_cell_size > 0:
                existing_cells = sum(sum(1 for exists in row if exists) for row in cell_exists_grid)
                print(f"   🎯 CELL ANALYSIS: {existing_cells}/{cols*rows} cells actually exist")
                print(f"   Cell size {true_cell_size}x{true_cell_size} pixels")
                print(f"   Grid origin: ({grid_offset_x}, {grid_offset_y})")
                print(f"   Using IDENTICAL edge detection for templates and board")
                expected_grid_area = f"{true_cell_size * cols}x{true_cell_size * rows}"
                print(f"   Expected grid area: {expected_grid_area}")
            else:
                square_cell_size = min(fallback_cellW, fallback_cellH)
                print(f"   FALLBACK: cell size {square_cell_size}x{square_cell_size} pixels (assumed uniform)")
            print(f"   Template scale: {optimal_template_scale:.3f}")
            print()

            global LegalMoves
            LegalMoves = []

            # Clear board state
            for i in range(rows):
                for j in range(cols):
                    mainArray[i][j] = ""

            # PRIORITY 1: POWERUP DETECTION (BEFORE PIECES)
            powerup_positions = []
            if available_powerups:
                powerup_positions = find_powerups_on_board(available_powerups, optimal_thresholds)
                if powerup_positions:
                    create_powerup_moves(powerup_positions)
                    print(f"⚡ POWERUP PRIORITY: {len(powerup_positions)} powerups detected, {len([m for m in LegalMoves if m[4].startswith('POWERUP_')])} moves created")
            
            # PRIORITY 2: PIECE DETECTION (NORMAL GAMEPLAY)
            # Process all available templates using IDENTICAL edge detection and optimal thresholds
            for template_name, template_path in available_templates.items():
                color = debug_colors[(ord(template_name[0]) - ord('A')) % len(debug_colors)]  # Use first letter for color cycling
                template_threshold = optimal_thresholds.get(template_name, 0.25)  # Use optimal threshold or fallback
                
                # Load and show template info before processing
                template = cv.imread(str(template_path), cv.IMREAD_GRAYSCALE)
                if template is not None:
                    orig_template_size = f"{template.shape[1]}x{template.shape[0]}"
                    if optimal_template_scale > 0:
                        new_width = int(template.shape[1] * optimal_template_scale)
                        new_height = int(template.shape[0] * optimal_template_scale)
                        scaled_template_size = f"{new_width}x{new_height}"
                        print(f"📏 Template {template_name}: {orig_template_size} → {scaled_template_size} (scale: {optimal_template_scale:.3f})")
                
                find_all_occurences_into_mainArray_with_cell_analysis(str(template_path), template_name, color, template_threshold)

            # Get current board state for comparison
            current_board_state = get_board_state_hash()

            # Validate board state - only count pieces in existing cells
            total_pieces = 0
            empty_existing_cells = 0
            for i in range(rows):
                for j in range(cols):
                    if cell_exists_grid[i][j]:  # Only count existing cells
                        if mainArray[i][j] != "":
                            total_pieces += 1
                        else:
                            empty_existing_cells += 1
            
            print(f"📊 Board state: {total_pieces} pieces detected in existing cells, {empty_existing_cells} existing cells empty")
            
            # Show current board prominently with cell existence markers
            print(f"\n" + "="*60)
            print(f"🎮 DETECTED BOARD STATE WITH CELL ANALYSIS (Cycle {move_count + 1}):")
            print(f"📊 {total_pieces} pieces detected in existing cells")
            if powerup_positions:
                print(f"⚡ {len(powerup_positions)} powerups detected on board")
                for x, y, name, conf in powerup_positions[:3]:  # Show top 3 powerups
                    priority = int(name) if name.isdigit() else 0
                    print(f"   ⚡ Powerup {name} at ({x},{y}) - Priority {priority}")
            print("="*60)
            printMainArrayWithCellAnalysis()
            print("="*60 + "\n")
            
            # Validation: A normal board should have 45-64 pieces (most cells filled)
            if total_pieces < 20:
                print("⚠️  Warning: Very few pieces detected - template matching may be too strict")
            elif total_pieces > 55:
                print("⚠️  Warning: Too many pieces detected - template matching may be too loose")

            # Search for additional piece moves (powerup moves already added)
            piece_moves_before = len(LegalMoves)
            searchMoves()
            piece_moves_added = len(LegalMoves) - piece_moves_before
            
            powerup_moves_count = len([m for m in LegalMoves if m[4].startswith('POWERUP_')])
            print(f"🎯 Total moves: {len(LegalMoves)} ({powerup_moves_count} powerups + {len(LegalMoves) - powerup_moves_count} pieces)")
            
            # Validation: Normal board should have 5-30 moves
            if len(LegalMoves) > 100:
                print("⚠️  Warning: Excessive moves detected - likely template matching issues")
                print("📋 Reducing move list randomly with spread for safety...")
                import random
                
                # Keep powerup moves (highest priority)
                powerup_moves = [m for m in LegalMoves if m[4].startswith('POWERUP_')]
                piece_moves = [m for m in LegalMoves if not m[4].startswith('POWERUP_')]
                
                # Randomly sample from piece moves with spread across the board
                if len(piece_moves) > 20:
                    # Sort piece moves by grid position to ensure spread
                    piece_moves.sort(key=lambda m: (m[0], m[1]))  # Sort by x, then y
                    
                    # Take every nth move to get spread, then random sample from remainder
                    spread_moves = piece_moves[::max(1, len(piece_moves)//10)]  # Every nth move
                    remaining_moves = [m for m in piece_moves if m not in spread_moves]
                    random_moves = random.sample(remaining_moves, min(15, len(remaining_moves)))
                    
                    piece_moves = spread_moves + random_moves
                
                LegalMoves = powerup_moves + piece_moves[:20]
                print(f"   📊 Kept {len(powerup_moves)} powerups + {len(LegalMoves) - len(powerup_moves)} spread piece moves")
            
            if LegalMoves:
                best_move = chooseBestMove(BlockIDsPreference)
                print(f"✨ Best move: {best_move}")
                
                if makeMove(best_move, gridX1, gridY1):
                    # Wait a moment for the move to complete and screen to update
                    time.sleep(0.5)
                    
                    # Take another screenshot to check if board state changed
                    check_image = gridScreenshot("check.png", gridX1, gridY1, gridX2, gridY2)
                    FullGridImageOriginal = cv.cvtColor(np.array(check_image), cv.COLOR_RGB2BGR)
                    
                    # Clear and rebuild board state for comparison
                    for i in range(rows):
                        for j in range(cols):
                            mainArray[i][j] = ""
                    
                    # Re-detect pieces
                    for template_name, template_path in available_templates.items():
                        color = debug_colors[(ord(template_name[0]) - ord('A')) % len(debug_colors)]
                        template_threshold = optimal_thresholds.get(template_name, 0.5)
                        find_all_occurences_into_mainArray(str(template_path), template_name, color, template_threshold)
                    
                    new_board_state = get_board_state_hash()
                    
                    # Check if board state changed
                    if board_states_equal(current_board_state, new_board_state):
                        # Board didn't change - move failed
                        if handle_stuck_move(best_move):
                            print("⏳ Skipping this cycle due to repeated failures...")
                            time.sleep(1.5)
                            continue
                    else:
                        # Board changed - move succeeded
                        move_count += 1
                        stuck_move_count = 0  # Reset stuck counter
                        failed_moves_history.clear()  # Clear failed moves on successful move
                        last_failed_move = None  # Reset last failed move tracking
                        print(f"✅ Move {move_count} completed successfully!")
                        print(f"   🧹 Cleared failed moves history after successful move")
                        
                    # Update previous board state
                    previous_board_state = new_board_state
                else:
                    print("❌ Move execution failed")
            else:
                print("🤔 No moves found")
                stuck_move_count = 0  # Reset stuck counter when no moves available

            # Clear moves and wait
            LegalMoves.clear()
            print("⏳ Waiting 1.5 seconds...")
            time.sleep(1.0)  # Wait 1.0 second since we already waited 0.5 seconds for move check = 1.5 total
            
    except KeyboardInterrupt:
        print("\n⏹️  Bot stopped by user (Ctrl+C)")
    except pyautogui.FailSafeException:
        print("\n🛡️  Emergency stop activated (mouse moved to corner)")
    except Exception as e:
        print(f"\n❌ Error: {e}")
    
    print("\n✋ You now have control of your mouse back!")

if __name__ == "__main__":
    main()