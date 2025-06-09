"""
Test Match3-Bot with Extracted Templates

Tests the match3-bot using our manually extracted Royal Match piece templates.
This will help validate that our template extraction is working correctly.

Usage:
    python -m src.test_match3_bot
"""

import cv2 as cv
import numpy as np
import sys
import os
from pathlib import Path

# Add the match3-bot directory to Python path
sys.path.append(str(Path("vendor/match3-bot")))

# Import the match3-bot functionality
try:
    from Python_match_3_bot_test import match_templates, take_screenshot
except ImportError as e:
    print(f"❌ Could not import match3-bot: {e}")
    print("Make sure vendor/match3-bot/Python_match_3_bot_test.py exists")
    sys.exit(1)

class Match3BotTester:
    def __init__(self):
        self.templates_dir = Path("vendor/match3-bot/screenshots")
        self.test_images_dir = Path("data/test-boards")
        
        # Ensure test directory exists
        self.test_images_dir.mkdir(exist_ok=True)
        
    def validate_templates(self):
        """Check that our extracted templates are properly set up."""
        print("🔍 Validating extracted templates...")
        
        template_files = list(self.templates_dir.glob("*.png"))
        if not template_files:
            print("❌ No template files found in vendor/match3-bot/screenshots/")
            return False
            
        print(f"✅ Found {len(template_files)} template files:")
        for template_file in sorted(template_files):
            img = cv.imread(str(template_file))
            if img is not None:
                h, w = img.shape[:2]
                print(f"  📋 {template_file.name}: {w}x{h} pixels")
            else:
                print(f"  ❌ {template_file.name}: Could not load")
                return False
                
        return True
        
    def create_simple_test_board(self):
        """Create a simple test board with known patterns for validation."""
        print("🎨 Creating simple test board...")
        
        # Load our piece templates
        templates = []
        for i in range(1, 6):  # We have 5 piece templates
            template_path = self.templates_dir / f"{i}.png"
            if template_path.exists():
                template = cv.imread(str(template_path))
                if template is not None:
                    templates.append(template)
                    
        if len(templates) < 3:
            print("❌ Need at least 3 piece templates to create test board")
            return None
            
        print(f"✅ Loaded {len(templates)} piece templates")
        
        # Create a simple 6x6 grid with our templates
        tile_size = 64  # Our templates are 64x64
        board_size = 6
        board_img = np.zeros((board_size * tile_size, board_size * tile_size, 3), dtype=np.uint8)
        
        # Fill with a simple pattern that has obvious matches
        pattern = [
            [0, 1, 2, 0, 1, 2],
            [1, 2, 0, 1, 2, 0],
            [2, 0, 1, 2, 0, 1],
            [0, 1, 2, 0, 1, 2],
            [1, 2, 0, 1, 2, 0],
            [2, 0, 1, 2, 0, 1]
        ]
        
        for row in range(board_size):
            for col in range(board_size):
                template_idx = pattern[row][col] % len(templates)
                template = templates[template_idx]
                
                y1 = row * tile_size
                x1 = col * tile_size
                y2 = y1 + tile_size
                x2 = x1 + tile_size
                
                # Resize template to exactly tile_size if needed
                if template.shape[:2] != (tile_size, tile_size):
                    template = cv.resize(template, (tile_size, tile_size))
                    
                board_img[y1:y2, x1:x2] = template
                
        # Save test board
        test_board_path = self.test_images_dir / "simple_test_board.png"
        cv.imwrite(str(test_board_path), board_img)
        print(f"💾 Saved test board: {test_board_path}")
        
        return str(test_board_path)
        
    def test_template_matching(self, test_board_path):
        """Test template matching on our created board."""
        print("🔬 Testing template matching...")
        
        if not test_board_path or not Path(test_board_path).exists():
            print("❌ Test board not found")
            return False
            
        # Load test board
        board_img = cv.imread(test_board_path)
        if board_img is None:
            print("❌ Could not load test board")
            return False
            
        print(f"✅ Loaded test board: {board_img.shape[1]}x{board_img.shape[0]}")
        
        # Test each template individually
        results = {}
        for i in range(1, 6):
            template_path = self.templates_dir / f"{i}.png"
            if not template_path.exists():
                continue
                
            template = cv.imread(str(template_path))
            if template is None:
                continue
                
            # Perform template matching
            result = cv.matchTemplate(board_img, template, cv.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)
            
            # Count matches above threshold
            threshold = 0.8  # High threshold for exact matches
            matches = np.where(result >= threshold)
            match_count = len(matches[0])
            
            results[i] = {
                'max_confidence': max_val,
                'match_count': match_count,
                'best_location': max_loc
            }
            
            print(f"  🔍 Template {i}: max_conf={max_val:.3f}, matches={match_count}")
            
        return results
        
    def run_full_test(self):
        """Run complete test suite."""
        print("🤖 Match3-Bot Template Test Suite")
        print("=" * 40)
        
        # Step 1: Validate templates
        if not self.validate_templates():
            return False
            
        # Step 2: Create test board
        test_board_path = self.create_simple_test_board()
        if not test_board_path:
            return False
            
        # Step 3: Test template matching
        results = self.test_template_matching(test_board_path)
        if not results:
            return False
            
        # Step 4: Analyze results
        print("\n📊 Test Results Summary:")
        successful_templates = 0
        for template_id, result in results.items():
            max_conf = result['max_confidence']
            match_count = result['match_count']
            
            if max_conf >= 0.8 and match_count > 0:
                status = "✅ GOOD"
                successful_templates += 1
            elif max_conf >= 0.6:
                status = "⚠️  FAIR"
            else:
                status = "❌ POOR"
                
            print(f"  Template {template_id}: {status} (conf: {max_conf:.3f}, matches: {match_count})")
            
        print(f"\n🎯 Overall Result: {successful_templates}/{len(results)} templates working well")
        
        if successful_templates >= len(results) * 0.6:  # 60% success rate
            print("✅ Template extraction appears successful!")
            print("🚀 Ready to test with real Royal Match screenshots")
            return True
        else:
            print("⚠️  Template extraction may need refinement")
            print("🔧 Consider adjusting crop parameters or trying different screenshots")
            return False

def main():
    tester = Match3BotTester()
    success = tester.run_full_test()
    
    if success:
        print(f"\n🎮 Next steps:")
        print(f"1. Take a screenshot of Royal Match game board")
        print(f"2. Save it as 'data/test-boards/royal-match-board.png'")
        print(f"3. Test template matching against real game")
    else:
        print(f"\n🔧 Troubleshooting needed:")
        print(f"1. Check template image quality")
        print(f"2. Verify crop parameters in manual_template_processor.py")
        print(f"3. Consider taking new screenshots with better lighting/contrast")

if __name__ == "__main__":
    main() 