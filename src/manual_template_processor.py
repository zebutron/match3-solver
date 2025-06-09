"""
Manual Template Processor for Royal Match

Processes manually categorized screenshots into normalized templates:
- Pieces: Basic matchable game pieces
- Obstacles: Non-swipeable board entities
- Powerups: Special tiles numbered by priority (1=lowest, 5=highest)
- Layers: Elements layered atop/beneath pieces

Usage:
    python -m src.manual_template_processor
"""

import cv2 as cv
import numpy as np
import os
from pathlib import Path
from PIL import Image
import shutil

class ManualTemplateProcessor:
    def __init__(self, game_name):
        self.game_name = game_name
        self.inputs_base = Path("data/templates") / game_name / "inputs"
        self.outputs_base = Path("data/templates") / game_name / "extracted"
        
        # Template categories to process
        self.categories = ["pieces", "obstacles", "powerups", "layers"]
        
        # Target template size (square for consistency)
        self.target_size = 64  # 64x64 pixels - good balance of detail and performance
        
    def process_all_categories(self):
        """Process all template categories."""
        print(f"🎮 Processing manual templates for: {self.game_name}")
        print("=" * 50)
        
        # Create output directories
        self._setup_output_directories()
        
        # Process each category
        for category in self.categories:
            input_dir = self.inputs_base / category
            output_dir = self.outputs_base / category
            
            if not input_dir.exists():
                print(f"⚠️  Category '{category}' not found, skipping...")
                continue
                
            print(f"\n📂 Processing {category.upper()}...")
            self._process_category(input_dir, output_dir, category)
            
        print(f"\n✅ Manual template processing complete!")
        print(f"📁 Templates saved to: {self.outputs_base}")
        
    def _setup_output_directories(self):
        """Create clean output directory structure."""
        # Remove existing extracted directory
        if self.outputs_base.exists():
            shutil.rmtree(self.outputs_base)
            
        # Create fresh directories for each category
        for category in self.categories:
            (self.outputs_base / category).mkdir(parents=True, exist_ok=True)
            
        print(f"🧹 Cleaned and created output directories in: {self.outputs_base}")
        
    def _process_category(self, input_dir, output_dir, category):
        """Process all images in a specific category."""
        # Find all image files
        image_files = []
        for ext in ["*.png", "*.jpg", "*.PNG", "*.JPG"]:
            image_files.extend(input_dir.glob(ext))
            
        # Filter out system files
        image_files = [f for f in image_files if not f.name.startswith('.')]
        
        if not image_files:
            print(f"  ❌ No images found in {category}")
            return
            
        print(f"  📸 Found {len(image_files)} images")
        
        # Sort powerups by number for priority ordering
        if category == "powerups":
            image_files.sort(key=lambda x: self._extract_number_from_filename(x.name))
            
        # Process each image
        for i, img_file in enumerate(image_files, 1):
            try:
                processed_template = self._process_single_image(img_file, category)
                
                if processed_template is not None:
                    # Generate output filename
                    if category == "powerups":
                        # Keep original numbering for powerups
                        original_num = self._extract_number_from_filename(img_file.name)
                        output_name = f"{original_num}.png"
                    else:
                        # Sequential numbering for other categories
                        output_name = f"{i}.png"
                        
                    output_path = output_dir / output_name
                    cv.imwrite(str(output_path), processed_template)
                    print(f"  💾 Saved: {output_name}")
                else:
                    print(f"  ❌ Failed to process: {img_file.name}")
                    
            except Exception as e:
                print(f"  ❌ Error processing {img_file.name}: {e}")
                
    def _extract_number_from_filename(self, filename):
        """Extract number from filename for powerup ordering."""
        import re
        numbers = re.findall(r'\d+', filename)
        return int(numbers[0]) if numbers else 999
        
    def _process_single_image(self, img_file, category):
        """Process a single image into a normalized template."""
        print(f"    🔧 Processing: {img_file.name}")
        
        # Load image
        img = cv.imread(str(img_file))
        if img is None:
            return None
            
        # Auto-detect and crop the main game element
        cropped_element = self._auto_crop_element(img, category)
        
        if cropped_element is None:
            print(f"    ⚠️  Could not detect element in {img_file.name}")
            return None
            
        # Normalize to target size
        normalized = cv.resize(cropped_element, (self.target_size, self.target_size), 
                             interpolation=cv.INTER_CUBIC)
        
        # Optional: enhance contrast and sharpening for better matching
        enhanced = self._enhance_template(normalized)
        
        print(f"    ✅ Extracted {enhanced.shape[1]}x{enhanced.shape[0]} template")
        return enhanced
        
    def _auto_crop_element(self, img, category):
        """Automatically detect and crop the main game element from screenshot."""
        h, w = img.shape[:2]
        
        # Convert to different color spaces for better element detection
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        
        # Method 1: Contour detection for distinct shapes
        element = self._crop_by_contours(img, gray)
        if element is not None:
            return element
            
        # Method 2: Center crop with smart sizing based on category
        return self._smart_center_crop(img, category)
        
    def _crop_by_contours(self, img, gray):
        """Try to detect element boundaries using contour detection."""
        # Apply different preprocessing based on image characteristics
        blurred = cv.GaussianBlur(gray, (5, 5), 0)
        
        # Try edge detection with different thresholds
        for low_thresh in [30, 50, 80]:
            edges = cv.Canny(blurred, low_thresh, low_thresh * 2)
            contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Find the largest reasonable contour
                for contour in sorted(contours, key=cv.contourArea, reverse=True):
                    area = cv.contourArea(contour)
                    img_area = img.shape[0] * img.shape[1]
                    
                    # Element should be 5-70% of image area
                    if 0.05 * img_area <= area <= 0.7 * img_area:
                        x, y, w, h = cv.boundingRect(contour)
                        
                        # Element should be reasonably square-ish (0.5 to 2.0 aspect ratio)
                        aspect_ratio = w / h
                        if 0.5 <= aspect_ratio <= 2.0 and w > 20 and h > 20:
                            # Add some padding around the detected element
                            padding = max(10, min(w, h) // 10)
                            x1 = max(0, x - padding)
                            y1 = max(0, y - padding)
                            x2 = min(img.shape[1], x + w + padding)
                            y2 = min(img.shape[0], y + h + padding)
                            
                            return img[y1:y2, x1:x2]
        
        return None
        
    def _smart_center_crop(self, img, category):
        """Intelligent center cropping based on category expectations."""
        h, w = img.shape[:2]
        
        # Category-specific crop ratios
        if category == "pieces":
            # Basic pieces are usually smaller, centered
            crop_ratio = 0.4
        elif category == "obstacles":
            # Obstacles can be larger, more varied
            crop_ratio = 0.6
        elif category == "powerups":
            # Powerups are usually medium-sized, often with effects
            crop_ratio = 0.5
        elif category == "layers":
            # Layers might be subtle, use larger crop
            crop_ratio = 0.7
        else:
            crop_ratio = 0.5
            
        # Calculate crop dimensions (square)
        crop_size = int(min(h, w) * crop_ratio)
        
        # Center the crop
        start_x = (w - crop_size) // 2
        start_y = (h - crop_size) // 2
        
        return img[start_y:start_y+crop_size, start_x:start_x+crop_size]
        
    def _enhance_template(self, img):
        """Apply enhancement to make templates more distinctive for matching."""
        # Slight sharpening to make edges more distinct
        kernel = np.array([[-1,-1,-1],
                          [-1, 9,-1],
                          [-1,-1,-1]])
        sharpened = cv.filter2D(img, -1, kernel)
        
        # Blend original with sharpened (mild enhancement)
        enhanced = cv.addWeighted(img, 0.7, sharpened, 0.3, 0)
        
        return enhanced
        
    def generate_match3_bot_templates(self):
        """Copy pieces templates to match3-bot expected location for testing."""
        pieces_dir = self.outputs_base / "pieces"
        match3_bot_dir = Path("vendor/match3-bot/screenshots")
        
        if not pieces_dir.exists():
            print("❌ No extracted pieces found. Run processing first.")
            return
            
        # Clear existing match3-bot templates
        if match3_bot_dir.exists():
            for existing in match3_bot_dir.glob("*.png"):
                existing.unlink()
        else:
            match3_bot_dir.mkdir(parents=True, exist_ok=True)
            
        # Copy pieces as numbered templates (1.png, 2.png, etc.)
        piece_files = list(pieces_dir.glob("*.png"))
        piece_files.sort(key=lambda x: int(x.stem))
        
        for i, piece_file in enumerate(piece_files, 1):
            dest_path = match3_bot_dir / f"{i}.png"
            shutil.copy2(piece_file, dest_path)
            print(f"📋 Copied {piece_file.name} → {dest_path.name}")
            
        print(f"✅ Prepared {len(piece_files)} piece templates for match3-bot testing")

def main():
    print("🎮 Manual Template Processor for Royal Match")
    print("=" * 50)
    
    processor = ManualTemplateProcessor("royal-match")
    
    # Process all categories
    processor.process_all_categories()
    
    # Generate match3-bot compatible templates for initial testing
    print(f"\n🤖 Preparing templates for match3-bot testing...")
    processor.generate_match3_bot_templates()

if __name__ == "__main__":
    main() 