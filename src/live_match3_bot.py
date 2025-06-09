"""
Live Match3-Bot for Royal Match via iPhone Mirroring

This script runs the match3-bot against live Royal Match gameplay.
It will ask you to define the game board boundaries, then continuously
analyze the board and make optimal moves.

Setup:
1. Open iPhone Mirroring with Royal Match ready
2. Position the game window where you can see the full board
3. Run this script and follow the coordinate setup prompts
4. Watch the bot play!

Usage:
    python -m src.live_match3_bot
"""

import cv2 as cv
import numpy as np
import pyautogui
import time
import sys
from pathlib import Path

class LiveMatch3Bot:
    def __init__(self):
        self.templates_dir = Path("vendor/match3-bot/screenshots")
        self.debug_dir = Path("data/debug")
        self.debug_dir.mkdir(exist_ok=True)
        
        # Game settings - will be set during calibration
        self.grid_x1 = None
        self.grid_y1 = None
        self.grid_x2 = None
        self.grid_y2 = None
        
        # Grid configuration (Royal Match is typically 8x8)
        self.rows = 8
        self.cols = 8
        
        # Template matching threshold
        self.match_threshold = 0.75  # Slightly lower for real screenshots
        
        # Move preferences (template IDs in order of preference)
        self.piece_preferences = [1, 2, 3, 4, 5]  # Adjust based on your piece priority
        
        # Safety settings
        pyautogui.FAILSAFE = True  # Move mouse to corner to stop
        pyautogui.PAUSE = 0.1  # Small pause between actions
        
    def setup_coordinates(self):
        """Interactive setup to get game board coordinates."""
        print("🎯 Game Board Coordinate Setup")
        print("=" * 40)
        print("📱 Make sure Royal Match is visible on your screen")
        print("🎮 Position your iPhone Mirroring window so the game board is clearly visible")
        print()
        print("We need to define the exact boundaries of the game board.")
        print("You can either:")
        print("  1. Click on corners to auto-detect")
        print("  2. Enter coordinates manually")
        print()
        
        choice = input("Choose method (1 for click, 2 for manual): ").strip()
        
        if choice == "1":
            self._setup_by_clicking()
        else:
            self._setup_manually()
            
        print(f"✅ Game board set to: ({self.grid_x1}, {self.grid_y1}) → ({self.grid_x2}, {self.grid_y2})")
        print(f"📐 Board size: {self.grid_x2 - self.grid_x1} x {self.grid_y2 - self.grid_y1} pixels")
        
    def _setup_by_clicking(self):
        """Setup coordinates by clicking on screen."""
        print("\n🖱️  Click Setup Mode")
        print("Instructions:")
        print("1. Click on the TOP-LEFT corner of the game board")
        print("2. Then click on the BOTTOM-RIGHT corner of the game board")
        print("3. Press ENTER after each click")
        print()
        
        input("Position your mouse over the TOP-LEFT corner of the game board, then press ENTER...")
        pos1 = pyautogui.position()
        self.grid_x1, self.grid_y1 = pos1.x, pos1.y
        print(f"✅ Top-left: ({self.grid_x1}, {self.grid_y1})")
        
        input("Now position your mouse over the BOTTOM-RIGHT corner of the game board, then press ENTER...")
        pos2 = pyautogui.position()
        self.grid_x2, self.grid_y2 = pos2.x, pos2.y
        print(f"✅ Bottom-right: ({self.grid_x2}, {self.grid_y2})")
        
    def _setup_manually(self):
        """Setup coordinates by manual input."""
        print("\n⌨️  Manual Coordinate Entry")
        print("Tip: Use a screenshot tool to find exact pixel coordinates")
        print()
        
        self.grid_x1 = int(input("Enter X coordinate of top-left corner: "))
        self.grid_y1 = int(input("Enter Y coordinate of top-left corner: "))
        self.grid_x2 = int(input("Enter X coordinate of bottom-right corner: "))
        self.grid_y2 = int(input("Enter Y coordinate of bottom-right corner: "))
        
    def take_board_screenshot(self):
        """Take a screenshot of just the game board area."""
        width = self.grid_x2 - self.grid_x1
        height = self.grid_y2 - self.grid_y1
        
        region = (self.grid_x1, self.grid_y1, width, height)
        screenshot = pyautogui.screenshot(region=region)
        
        # Convert PIL to OpenCV format
        board_img = cv.cvtColor(np.array(screenshot), cv.COLOR_RGB2BGR)
        
        # Save debug copy
        debug_path = self.debug_dir / f"board_capture_{int(time.time())}.png"
        cv.imwrite(str(debug_path), board_img)
        
        return board_img
        
    def analyze_board(self, board_img):
        """Analyze the board using template matching to identify pieces."""
        h, w = board_img.shape[:2]
        
        # Initialize board state
        board_state = np.zeros((self.rows, self.cols), dtype=int)
        
        # Calculate cell dimensions
        cell_w = w // self.cols
        cell_h = h // self.rows
        
        # Get center points for each cell (for debugging)
        center_points_x = []
        center_points_y = []
        for i in range(self.cols):
            center_points_x.append((cell_w * i) + (cell_w // 2))
        for j in range(self.rows):
            center_points_y.append((cell_h * j) + (cell_h // 2))
            
        # Convert to grayscale for template matching
        board_gray = cv.cvtColor(board_img, cv.COLOR_BGR2GRAY)
        
        # Test each template
        for template_id in range(1, 6):  # Templates 1-5
            template_path = self.templates_dir / f"{template_id}.png"
            if not template_path.exists():
                continue
                
            template = cv.imread(str(template_path), 0)  # Grayscale
            if template is None:
                continue
                
            # Perform template matching
            result = cv.matchTemplate(board_gray, template, cv.TM_CCOEFF_NORMED)
            locations = np.where(result >= self.match_threshold)
            
            # Map matches to grid positions
            for pt in zip(*locations[::-1]):  # pt is (x, y) of top-left corner
                template_h, template_w = template.shape
                center_x = pt[0] + template_w // 2
                center_y = pt[1] + template_h // 2
                
                # Find closest grid cell
                grid_col = self._find_nearest(center_points_x, center_x)
                grid_row = self._find_nearest(center_points_y, center_y)
                
                # Only assign if cell is empty (avoid overlaps)
                if 0 <= grid_row < self.rows and 0 <= grid_col < self.cols:
                    if board_state[grid_row][grid_col] == 0:
                        board_state[grid_row][grid_col] = template_id
                        
        return board_state, center_points_x, center_points_y
        
    def _find_nearest(self, array, value):
        """Find index of nearest value in array."""
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx
        
    def find_possible_moves(self, board_state):
        """Find all possible 3+ match moves on the current board."""
        legal_moves = []
        
        for row in range(self.rows):
            for col in range(self.cols):
                piece_type = board_state[row][col]
                if piece_type == 0:  # Empty cell
                    continue
                    
                # Check all possible moves for this position
                legal_moves.extend(self._check_3_matches(board_state, col, row, piece_type))
                legal_moves.extend(self._check_4_matches(board_state, col, row, piece_type))
                legal_moves.extend(self._check_5_matches(board_state, col, row, piece_type))
                
        return legal_moves
        
    def _check_3_matches(self, board, x, y, piece_type):
        """Check for possible 3-matches from this position."""
        moves = []
        
        # Horizontal 3-match by moving down
        if (y + 1 < self.rows and x - 1 >= 0 and x + 1 < self.cols and
            board[y + 1][x - 1] == board[y + 1][x + 1] == piece_type):
            moves.append((x, y, "down", 3, piece_type))
            
        # Horizontal 3-match by moving up  
        if (y - 1 >= 0 and x - 1 >= 0 and x + 1 < self.cols and
            board[y - 1][x - 1] == board[y - 1][x + 1] == piece_type):
            moves.append((x, y, "up", 3, piece_type))
            
        # Vertical 3-match by moving right
        if (x + 1 < self.cols and y - 1 >= 0 and y + 1 < self.rows and
            board[y - 1][x + 1] == board[y + 1][x + 1] == piece_type):
            moves.append((x, y, "right", 3, piece_type))
            
        # Vertical 3-match by moving left
        if (x - 1 >= 0 and y - 1 >= 0 and y + 1 < self.rows and
            board[y - 1][x - 1] == board[y + 1][x - 1] == piece_type):
            moves.append((x, y, "left", 3, piece_type))
            
        return moves
        
    def _check_4_matches(self, board, x, y, piece_type):
        """Check for possible 4-matches from this position."""
        moves = []
        # Simplified 4-match detection - add more patterns as needed
        return moves
        
    def _check_5_matches(self, board, x, y, piece_type):
        """Check for possible 5-matches from this position."""
        moves = []
        # Simplified 5-match detection - add more patterns as needed  
        return moves
        
    def choose_best_move(self, legal_moves):
        """Choose the best move based on preferences."""
        if not legal_moves:
            return None
            
        # Sort by match size (5 > 4 > 3) then by piece preference
        def move_priority(move):
            x, y, direction, match_size, piece_type = move
            piece_pref = self.piece_preferences.index(piece_type) if piece_type in self.piece_preferences else 999
            return (-match_size, piece_pref)  # Negative for descending order
            
        legal_moves.sort(key=move_priority)
        return legal_moves[0]
        
    def make_move(self, move, center_points_x, center_points_y):
        """Execute a move by dragging on screen."""
        if move is None:
            return False
            
        x, y, direction, match_size, piece_type = move
        
        # Get screen coordinates of the piece
        screen_x = center_points_x[x] + self.grid_x1
        screen_y = center_points_y[y] + self.grid_y1
        
        # Calculate drag destination
        if direction == "down":
            dest_x, dest_y = screen_x, center_points_y[y + 1] + self.grid_y1
        elif direction == "up":
            dest_x, dest_y = screen_x, center_points_y[y - 1] + self.grid_y1
        elif direction == "right":
            dest_x, dest_y = center_points_x[x + 1] + self.grid_x1, screen_y
        elif direction == "left":
            dest_x, dest_y = center_points_x[x - 1] + self.grid_x1, screen_y
        else:
            return False
            
        print(f"🎯 Making move: {piece_type}-match {direction} from ({x},{y})")
        print(f"   Dragging from ({screen_x}, {screen_y}) to ({dest_x}, {dest_y})")
        
        # Perform the drag
        pyautogui.moveTo(screen_x, screen_y)
        time.sleep(0.2)
        pyautogui.dragTo(dest_x, dest_y, duration=0.5, button='left')
        
        return True
        
    def run_game_loop(self):
        """Main game loop - continuously analyze and make moves."""
        print("\n🤖 Starting Live Match3-Bot")
        print("=" * 40)
        print("🛡️  SAFETY: Move mouse to top-left corner to emergency stop")
        print("⏰ Bot will analyze board every 3 seconds")
        print()
        
        move_count = 0
        
        try:
            while True:
                print(f"\n🔄 Analysis cycle {move_count + 1}")
                
                # Take screenshot
                board_img = self.take_board_screenshot()
                print("📸 Board screenshot captured")
                
                # Analyze board
                board_state, center_x, center_y = self.analyze_board(board_img)
                print("🧠 Board state analyzed")
                
                # Debug: Print board state
                print("📋 Current board:")
                for row in board_state:
                    print("   " + " ".join(str(cell) if cell != 0 else "." for cell in row))
                
                # Find possible moves
                legal_moves = self.find_possible_moves(board_state)
                print(f"🎯 Found {len(legal_moves)} possible moves")
                
                if legal_moves:
                    # Choose and make best move
                    best_move = self.choose_best_move(legal_moves)
                    print(f"✨ Best move: {best_move}")
                    
                    if self.make_move(best_move, center_x, center_y):
                        move_count += 1
                        print(f"✅ Move {move_count} completed!")
                    else:
                        print("❌ Failed to execute move")
                else:
                    print("🤔 No moves found - board may have changed or templates need adjustment")
                
                # Wait before next analysis
                print("⏳ Waiting 3 seconds for board to settle...")
                time.sleep(3)
                
        except KeyboardInterrupt:
            print("\n⏹️  Bot stopped by user")
        except pyautogui.FailSafeException:
            print("\n🛡️  Emergency stop activated!")
            
    def test_setup(self):
        """Test the coordinate setup by taking a screenshot."""
        print("\n🧪 Testing coordinate setup...")
        
        board_img = self.take_board_screenshot()
        test_path = self.debug_dir / "test_board_capture.png"
        cv.imwrite(str(test_path), board_img)
        
        print(f"✅ Test screenshot saved: {test_path}")
        print(f"📐 Captured image size: {board_img.shape[1]}x{board_img.shape[0]}")
        print("👀 Please check the image to ensure it captures the full game board")
        
        return input("\nDoes the test screenshot look correct? (y/n): ").lower().startswith('y')

def main():
    print("🎮 Live Match3-Bot for Royal Match")
    print("=" * 50)
    
    bot = LiveMatch3Bot()
    
    # Step 1: Setup coordinates
    bot.setup_coordinates()
    
    # Step 2: Test the setup
    if not bot.test_setup():
        print("❌ Please adjust coordinates and try again")
        return
        
    # Step 3: Start the game loop
    print("\n🚀 Ready to start bot!")
    input("Press ENTER when you're ready to begin automated play...")
    
    bot.run_game_loop()

if __name__ == "__main__":
    main() 