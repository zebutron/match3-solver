"""
Template Creator - Extract gem templates from match-3 game screenshots.

Usage:
1. Take a screenshot of a match-3 game with clear, unobstructed gems
2. Run this script to interactively crop gem templates
3. Templates will be saved to vendor/match3-bot/screenshots/
"""

import cv2 as cv
import numpy as np
import os
from pathlib import Path

class TemplateCreator:
    def __init__(self):
        self.templates_dir = Path("vendor/match3-bot/screenshots")
        self.templates_dir.mkdir(parents=True, exist_ok=True)
        self.current_template = 1
        self.crop_coords = []
        
    def create_from_screenshot(self, screenshot_path):
        """Extract templates from a full game screenshot."""
        if not os.path.exists(screenshot_path):
            print(f"Screenshot not found: {screenshot_path}")
            return
            
        img = cv.imread(screenshot_path)
        if img is None:
            print(f"Could not load image: {screenshot_path}")
            return
            
        print("Template Creator Instructions:")
        print("- Click and drag to select each gem type")
        print("- Press 's' to save current selection")
        print("- Press 'n' to skip to next template number")
        print("- Press 'q' to quit")
        print(f"- Currently creating template: {self.current_template}.png")
        
        cv.namedWindow('Template Creator', cv.WINDOW_NORMAL)
        cv.setMouseCallback('Template Creator', self._mouse_callback, img)
        
        while True:
            display_img = img.copy()
            
            # Draw current selection
            if len(self.crop_coords) == 2:
                cv.rectangle(display_img, self.crop_coords[0], self.crop_coords[1], (0, 255, 0), 2)
                cv.putText(display_img, f"Template {self.current_template}", 
                          (self.crop_coords[0][0], self.crop_coords[0][1] - 10),
                          cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Show current template number
            cv.putText(display_img, f"Creating Template: {self.current_template}.png", 
                      (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            cv.imshow('Template Creator', display_img)
            
            key = cv.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s') and len(self.crop_coords) == 2:
                self._save_template(img)
            elif key == ord('n'):
                self.current_template += 1
                self.crop_coords = []
                print(f"Now creating template: {self.current_template}.png")
                
        cv.destroyAllWindows()
        
    def _mouse_callback(self, event, x, y, flags, img):
        """Handle mouse events for cropping."""
        if event == cv.EVENT_LBUTTONDOWN:
            self.crop_coords = [(x, y)]
        elif event == cv.EVENT_LBUTTONUP and len(self.crop_coords) == 1:
            self.crop_coords.append((x, y))
            
    def _save_template(self, img):
        """Save the currently selected region as a template."""
        if len(self.crop_coords) != 2:
            return
            
        x1, y1 = self.crop_coords[0]
        x2, y2 = self.crop_coords[1]
        
        # Ensure coordinates are in correct order
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        
        # Crop the template
        template = img[y1:y2, x1:x2]
        
        # Save as grayscale (match3-bot loads templates in grayscale)
        template_gray = cv.cvtColor(template, cv.COLOR_BGR2GRAY)
        
        template_path = self.templates_dir / f"{self.current_template}.png"
        cv.imwrite(str(template_path), template_gray)
        
        print(f"✅ Saved template {self.current_template}.png ({x2-x1}x{y2-y1} pixels)")
        
        # Move to next template
        self.current_template += 1
        self.crop_coords = []

def main():
    creator = TemplateCreator()
    
    # Example usage
    screenshot_path = input("Enter path to game screenshot: ").strip()
    if screenshot_path:
        creator.create_from_screenshot(screenshot_path)
    else:
        print("Usage: python -m src.template_creator")
        print("Then provide path to a match-3 game screenshot")

if __name__ == "__main__":
    main() 