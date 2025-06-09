"""
Automated Grid Template Extractor for Match-3 Games

Takes screenshots of puzzle boards, automatically detects the game grid,
slices into tiles, clusters similar tiles using perceptual hashing,
and outputs template images for the match3-bot to use.

Usage:
    python -m src.template_creator royal-match

Requirements:
    - Input images in: data/templates/{game_name}/inputs/
    - Output templates to: data/templates/{game_name}/tiles/
"""

import cv2 as cv
import numpy as np
import os
import sys
from pathlib import Path
from collections import defaultdict, Counter
from PIL import Image
import imagehash

class AutomatedTemplateExtractor:
    def __init__(self, game_name):
        self.game_name = game_name
        self.base_dir = Path("data/templates") / game_name
        self.inputs_dir = self.base_dir / "inputs"
        self.tiles_dir = self.base_dir / "tiles"
        
        # Create directories if they don't exist
        self.tiles_dir.mkdir(parents=True, exist_ok=True)
        
        # Parameters
        self.hash_threshold = 4  # Max bit difference for clustering
        self.min_occurrences = 2  # Exclude tiles that appear only once
        
    def extract_templates(self):
        """Main pipeline: process all input images and generate template tiles."""
        print(f"🎮 Processing templates for game: {self.game_name}")
        
        if not self.inputs_dir.exists():
            print(f"❌ Input directory not found: {self.inputs_dir}")
            print(f"Please create the directory and add PNG/JPG screenshots.")
            return
            
        # Find all input images
        input_files = list(self.inputs_dir.glob("*.png")) + list(self.inputs_dir.glob("*.jpg"))
        if not input_files:
            print(f"❌ No PNG/JPG files found in {self.inputs_dir}")
            return
            
        print(f"📸 Found {len(input_files)} input images")
        
        # Process each input image
        all_tiles = []
        for img_file in input_files:
            print(f"🔍 Processing: {img_file.name}")
            tiles = self._process_single_image(img_file)
            all_tiles.extend(tiles)
            
        print(f"🧩 Extracted {len(all_tiles)} total tiles from all images")
        
        # Cluster similar tiles using perceptual hashing
        clusters = self._cluster_tiles(all_tiles)
        
        # Filter out tiles that appear only once (likely UI elements)
        filtered_clusters = self._filter_clusters(clusters)
        
        # Save exemplar templates
        self._save_templates(filtered_clusters)
        
        print(f"✅ Template extraction complete! Generated {len(filtered_clusters)} templates.")
        
    def _process_single_image(self, img_path):
        """Process a single input image and extract grid tiles."""
        img = cv.imread(str(img_path))
        if img is None:
            print(f"⚠️  Could not load image: {img_path}")
            return []
            
        # Detect board region
        board_region = self._detect_board_region(img)
        
        # Slice into grid tiles
        tiles = self._slice_into_tiles(board_region)
        
        return tiles
        
    def _detect_board_region(self, img):
        """Detect the game board region using contour detection with center-crop fallback."""
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        
        # Try contour detection first
        blurred = cv.GaussianBlur(gray, (5, 5), 0)
        edges = cv.Canny(blurred, 50, 150)
        
        # Find contours
        contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        
        # Look for large rectangular contours that could be the game board
        for contour in sorted(contours, key=cv.contourArea, reverse=True):
            area = cv.contourArea(contour)
            if area < img.shape[0] * img.shape[1] * 0.1:  # Must be at least 10% of image
                continue
                
            # Approximate contour to see if it's roughly rectangular
            epsilon = 0.02 * cv.arcLength(contour, True)
            approx = cv.approxPolyDP(contour, epsilon, True)
            
            if len(approx) >= 4:  # Roughly rectangular
                x, y, w, h = cv.boundingRect(contour)
                aspect_ratio = w / h
                
                # Game boards are usually square-ish (0.7 to 1.4 aspect ratio)
                if 0.7 <= aspect_ratio <= 1.4 and w > 200 and h > 200:
                    print(f"📐 Board detected via contours: {w}x{h} at ({x},{y})")
                    return img[y:y+h, x:x+w]
        
        # Fallback: center crop (assume board is in the center 60% of image)
        h, w = img.shape[:2]
        crop_size = min(h, w) * 0.6
        crop_size = int(crop_size)
        
        start_x = (w - crop_size) // 2
        start_y = (h - crop_size) // 2
        
        print(f"📐 Board detected via center-crop fallback: {crop_size}x{crop_size}")
        return img[start_y:start_y+crop_size, start_x:start_x+crop_size]
        
    def _slice_into_tiles(self, board_img):
        """Slice board image into equal square tiles (assumes 8x8 grid)."""
        h, w = board_img.shape[:2]
        
        # Assume 8x8 grid (standard for most match-3 games)
        rows, cols = 8, 8
        
        tile_height = h // rows
        tile_width = w // cols
        
        tiles = []
        for row in range(rows):
            for col in range(cols):
                y1 = row * tile_height
                x1 = col * tile_width
                y2 = y1 + tile_height
                x2 = x1 + tile_width
                
                tile = board_img[y1:y2, x1:x2]
                tiles.append({
                    'image': tile,
                    'position': (row, col),
                    'source_size': (h, w)
                })
                
        return tiles
        
    def _cluster_tiles(self, all_tiles):
        """Cluster tiles using perceptual hashing."""
        print(f"🔗 Clustering tiles using perceptual hashing...")
        
        # Compute hash for each tile
        tile_hashes = []
        for i, tile_data in enumerate(all_tiles):
            tile_img = tile_data['image']
            
            # Convert to PIL Image for imagehash
            pil_img = Image.fromarray(cv.cvtColor(tile_img, cv.COLOR_BGR2RGB))
            
            # Compute perceptual hash
            phash = imagehash.phash(pil_img)
            
            tile_hashes.append({
                'hash': phash,
                'tile_data': tile_data,
                'index': i
            })
        
        # Cluster tiles with similar hashes
        clusters = []
        used_indices = set()
        
        for i, tile_hash in enumerate(tile_hashes):
            if i in used_indices:
                continue
                
            # Start new cluster
            cluster = [tile_hash]
            used_indices.add(i)
            
            # Find similar tiles
            for j, other_hash in enumerate(tile_hashes):
                if j in used_indices:
                    continue
                    
                # Check if hashes are similar (≤ threshold bit difference)
                hash_diff = tile_hash['hash'] - other_hash['hash']
                if hash_diff <= self.hash_threshold:
                    cluster.append(other_hash)
                    used_indices.add(j)
                    
            clusters.append(cluster)
            
        print(f"🔗 Created {len(clusters)} clusters from {len(all_tiles)} tiles")
        return clusters
        
    def _filter_clusters(self, clusters):
        """Filter out clusters with too few occurrences (likely UI elements)."""
        filtered = []
        
        for cluster in clusters:
            if len(cluster) >= self.min_occurrences:
                filtered.append(cluster)
            else:
                print(f"🚮 Filtered out cluster with {len(cluster)} occurrence(s)")
                
        print(f"✅ Kept {len(filtered)} clusters after filtering")
        return filtered
        
    def _save_templates(self, clusters):
        """Save one exemplar template per cluster."""
        # Clear existing templates
        for existing in self.tiles_dir.glob("*.png"):
            existing.unlink()
            
        for i, cluster in enumerate(clusters, 1):
            # Choose the first tile as exemplar
            exemplar = cluster[0]['tile_data']['image']
            
            # Save as template
            template_path = self.tiles_dir / f"{i}.png"
            cv.imwrite(str(template_path), exemplar)
            
            print(f"💾 Saved template {i}.png ({len(cluster)} similar tiles found)")

def main():
    if len(sys.argv) != 2:
        print("Usage: python -m src.template_creator <game_name>")
        print("Example: python -m src.template_creator royal-match")
        return
        
    game_name = sys.argv[1]
    extractor = AutomatedTemplateExtractor(game_name)
    extractor.extract_templates()

if __name__ == "__main__":
    main() 