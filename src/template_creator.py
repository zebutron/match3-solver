"""
Automated Grid Template Extractor for Match-3 Games

Takes screenshots of puzzle boards, automatically detects the game grid,
slices into tiles, clusters similar tiles using perceptual hashing,
and outputs template images for the match3-bot to use.

Usage:
    python -m src.template_creator
    
Then enter the game name when prompted (e.g., "royal-match").

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
        self.hash_threshold = 20  # Max bit difference for clustering (very aggressive for complex games)
        self.min_occurrences = 2  # Exclude tiles that appear only once
        
    def extract_templates(self):
        """Main pipeline: process all input images and generate template tiles."""
        print(f"🎮 Processing templates for game: {self.game_name}")
        
        if not self.inputs_dir.exists():
            print(f"❌ Input directory not found: {self.inputs_dir}")
            print(f"Please create the directory and add PNG/JPG screenshots.")
            return
            
        # Clear previous outputs
        print("🧹 Cleaning up previous template images...")
        for existing in self.tiles_dir.glob("*.png"):
            existing.unlink()
        if (self.tiles_dir / "debug_samples").exists():
            import shutil
            shutil.rmtree(self.tiles_dir / "debug_samples")
            
        # Find all input images
        input_files = list(self.inputs_dir.glob("*.png")) + list(self.inputs_dir.glob("*.jpg"))
        if not input_files:
            print(f"❌ No PNG/JPG files found in {self.inputs_dir}")
            return
            
        print(f"📸 Found {len(input_files)} input images")
        
        # PASS 1: Detect all board regions and analyze dimensions
        print("🔍 Pass 1: Detecting board regions across all images...")
        board_data = []
        for img_file in input_files:
            print(f"  📐 Analyzing: {img_file.name}")
            img = cv.imread(str(img_file))
            if img is None:
                print(f"  ⚠️  Could not load image: {img_file}")
                continue
                
            board_region, board_info = self._detect_board_region_with_info(img)
            board_data.append({
                'file': img_file,
                'image': img,
                'board_region': board_region,
                'board_info': board_info
            })
            
        if not board_data:
            print("❌ No valid board regions detected")
            return
            
        # PASS 2: Determine optimal target dimensions
        target_size = self._determine_target_board_size(board_data)
        print(f"🎯 Target board size determined: {target_size}x{target_size}")
        
        # PASS 3: Normalize all boards and extract tiles using grid line detection
        print("⚖️  Pass 3: Normalizing boards and detecting grid lines for precise slicing...")
        all_tiles = []
        for data in board_data:
            print(f"  🔧 Processing: {data['file'].name}")
            normalized_board = self._normalize_board(data['board_region'], target_size)
            tiles = self._extract_tiles_using_grid_lines(normalized_board, data['file'].name)
            all_tiles.extend(tiles)
            
        print(f"🧩 Extracted {len(all_tiles)} grid-aligned tiles from all images")
        
        # PASS 4: Cluster similar tiles using perceptual hashing
        clusters = self._cluster_tiles(all_tiles)
        
        # Filter out tiles that appear only once (likely UI elements)
        filtered_clusters = self._filter_clusters(clusters)
        
        # Save exemplar templates
        self._save_templates(filtered_clusters)
        
        print(f"✅ Template extraction complete! Generated {len(filtered_clusters)} templates.")
        
    def _detect_board_region_with_info(self, img):
        """Detect the game board region and return both the region and detection info."""
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
                    board_info = {
                        'method': 'contour',
                        'dimensions': (w, h),
                        'position': (x, y),
                        'aspect_ratio': aspect_ratio
                    }
                    print(f"    📐 Contour detection: {w}x{h} at ({x},{y})")
                    return img[y:y+h, x:x+w], board_info
        
        # Fallback: center crop (assume board is in the center 60% of image)
        h, w = img.shape[:2]
        crop_size = min(h, w) * 0.6
        crop_size = int(crop_size)
        
        start_x = (w - crop_size) // 2
        start_y = (h - crop_size) // 2
        
        board_info = {
            'method': 'center_crop',
            'dimensions': (crop_size, crop_size),
            'position': (start_x, start_y),
            'aspect_ratio': 1.0
        }
        print(f"    📐 Center-crop fallback: {crop_size}x{crop_size}")
        return img[start_y:start_y+crop_size, start_x:start_x+crop_size], board_info
        
    def _determine_target_board_size(self, board_data):
        """Analyze all detected boards and determine optimal target size."""
        print("  🔍 Analyzing board dimensions:")
        
        sizes = []
        for data in board_data:
            w, h = data['board_info']['dimensions']
            method = data['board_info']['method']
            size = min(w, h)  # Use square based on smaller dimension
            sizes.append(size)
            print(f"    {data['file'].name}: {w}x{h} ({method}) → {size}px")
        
        # Use the median size as target (robust to outliers)
        import statistics
        target_size = int(statistics.median(sizes))
        
        # Round to nearest multiple of 8 for clean grid division
        target_size = ((target_size + 4) // 8) * 8
        
        print(f"  📊 Size analysis: min={min(sizes)}, median={statistics.median(sizes)}, max={max(sizes)}")
        print(f"  🎯 Selected target: {target_size}px (rounded to 8x multiple)")
        
        return target_size
        
    def _normalize_board(self, board_region, target_size):
        """Resize board region to target dimensions."""
        return cv.resize(board_region, (target_size, target_size), interpolation=cv.INTER_CUBIC)
        
    def _slice_into_uniform_tiles(self, board_img, source_name):
        """Slice normalized board image into equal square tiles."""
        h, w = board_img.shape[:2]
        
        # Use 8x8 grid (standard for most match-3 games)
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
                    'source_name': source_name,
                    'dimensions': (tile_width, tile_height)
                })
                
        return tiles
        
    def _extract_tiles_using_grid_lines(self, board_img, source_name):
        """Extract tiles by detecting actual grid lines in the board image."""
        h, w = board_img.shape[:2]
        
        # Convert to grayscale for line detection
        gray = cv.cvtColor(board_img, cv.COLOR_BGR2GRAY)
        
        # Enhance edges to make grid lines more visible
        blurred = cv.GaussianBlur(gray, (3, 3), 0)
        edges = cv.Canny(blurred, 50, 150)
        
        # Detect horizontal and vertical lines using HoughLinesP
        # Use more restrictive parameters to find major grid lines only
        min_line_length = min(h, w) // 6  # Lines must span at least 1/6 of the image
        lines = cv.HoughLinesP(edges, 1, np.pi/180, threshold=80, minLineLength=min_line_length, maxLineGap=15)
        
        horizontal_lines = []
        vertical_lines = []
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                
                # Calculate angle and line length
                angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                
                # Only consider long lines that span significant portion of the image
                min_span = min(h, w) * 0.5
                
                # Horizontal lines (angle close to 0 or 180)
                if abs(angle) < 5 or abs(angle - 180) < 5 or abs(angle + 180) < 5:
                    if length > min_span:
                        avg_y = (y1 + y2) // 2
                        horizontal_lines.append(avg_y)
                
                # Vertical lines (angle close to 90 or -90)
                elif abs(abs(angle) - 90) < 5:
                    if length > min_span:
                        avg_x = (x1 + x2) // 2
                        vertical_lines.append(avg_x)
        
        # Remove duplicates and sort
        horizontal_lines = sorted(list(set(horizontal_lines)))
        vertical_lines = sorted(list(set(vertical_lines)))
        
        # Filter to get only the main game grid (expect 6-10 cells)
        horizontal_lines = self._extract_main_grid_lines(horizontal_lines, h)
        vertical_lines = self._extract_main_grid_lines(vertical_lines, w)
        
        print(f"    📏 Detected {len(horizontal_lines)-1}x{len(vertical_lines)-1} grid from lines")
        
        # Debug: Save image with detected lines
        debug_dir = self.tiles_dir / "debug_samples"
        debug_dir.mkdir(exist_ok=True)
        debug_img = board_img.copy()
        for y in horizontal_lines:
            cv.line(debug_img, (0, y), (w, y), (0, 255, 0), 3)
        for x in vertical_lines:
            cv.line(debug_img, (x, 0), (x, h), (0, 255, 0), 3)
        debug_path = debug_dir / f"grid_detection_{source_name[:10]}.png"
        cv.imwrite(str(debug_path), debug_img)
        
        # If we don't have a reasonable grid, fall back to equal division
        rows = len(horizontal_lines) - 1
        cols = len(vertical_lines) - 1
        if rows < 5 or rows > 10 or cols < 5 or cols > 10:
            print(f"    ⚠️  Grid size {rows}x{cols} not reasonable, falling back to 8x8 division")
            return self._slice_into_uniform_tiles(board_img, source_name)
        
        # Extract tiles using detected grid lines
        tiles = []
        for i in range(len(horizontal_lines) - 1):
            for j in range(len(vertical_lines) - 1):
                y1, y2 = horizontal_lines[i], horizontal_lines[i + 1]
                x1, x2 = vertical_lines[j], vertical_lines[j + 1]
                
                # Add small padding to avoid grid lines in tiles
                padding = 3
                y1, x1 = max(0, y1 + padding), max(0, x1 + padding)
                y2, x2 = min(h, y2 - padding), min(w, x2 - padding)
                
                # Only include tiles that are reasonably sized
                tile_w, tile_h = x2 - x1, y2 - y1
                min_tile_size = min(h, w) // 12  # Tiles should be at least 1/12 of board dimension
                
                if tile_h >= min_tile_size and tile_w >= min_tile_size:
                    tile = board_img[y1:y2, x1:x2]
                    tiles.append({
                        'image': tile,
                        'position': (i, j),
                        'source_name': source_name,
                        'dimensions': (tile_w, tile_h),
                        'method': 'grid_lines'
                    })
        
        return tiles
        
    def _extract_main_grid_lines(self, lines, dimension):
        """Extract perfectly regular grid lines with strict equal spacing enforcement."""
        if len(lines) < 3:  # Need at least 3 lines to detect spacing
            # Fallback: create 8x8 grid
            step = dimension // 8
            return [i * step for i in range(9)]
        
        # Always include image borders
        if 0 not in lines:
            lines.insert(0, 0)
        if dimension not in lines:
            lines.append(dimension)
        
        lines = sorted(list(set(lines)))  # Remove duplicates and sort
        
        # Try different grid sizes (6-10 cells typical for match-3)
        best_grid = None
        best_score = float('inf')
        
        for grid_size in range(6, 11):
            perfect_spacing = dimension / grid_size
            perfect_grid = [round(i * perfect_spacing) for i in range(grid_size + 1)]
            
            # For each detected line, find how well it matches perfect grid positions
            tolerance = perfect_spacing * 0.1  # Very strict: 10% tolerance
            matched_lines = []
            
            for perfect_pos in perfect_grid:
                # Find the closest detected line to this perfect position
                closest_line = min(lines, key=lambda x: abs(x - perfect_pos))
                distance = abs(closest_line - perfect_pos)
                
                if distance <= tolerance:
                    matched_lines.append(closest_line)
                else:
                    # If we can't find a line close enough, use the perfect position
                    matched_lines.append(perfect_pos)
            
            # Calculate score: penalize deviations from perfect regularity
            spacings = [matched_lines[i+1] - matched_lines[i] for i in range(len(matched_lines)-1)]
            if len(spacings) > 0:
                avg_spacing = sum(spacings) / len(spacings)
                variance = sum((s - avg_spacing)**2 for s in spacings) / len(spacings)
                
                # Also penalize if we have too few actual detected lines matching
                actual_matches = sum(1 for perfect_pos in perfect_grid 
                                   if any(abs(line - perfect_pos) <= tolerance for line in lines))
                match_penalty = (grid_size + 1 - actual_matches) * 100
                
                score = variance + match_penalty
                
                if score < best_score:
                    best_score = score
                    best_grid = matched_lines
        
        # If no good regular grid found, fall back to equal division
        if best_grid is None or best_score > 50:  # Variance threshold
            print(f"    🔧 No regular grid detected (best score: {best_score:.1f}), using equal division")
            step = dimension // 8
            best_grid = [i * step for i in range(9)]
        else:
            # Calculate final regularity statistics
            spacings = [best_grid[i+1] - best_grid[i] for i in range(len(best_grid)-1)]
            avg_spacing = sum(spacings) / len(spacings)
            max_deviation = max(abs(s - avg_spacing) for s in spacings)
            print(f"    ✅ Regular grid found: {len(best_grid)-1} cells, avg spacing: {avg_spacing:.1f}px, max deviation: {max_deviation:.1f}px")
        
        return best_grid
        
    def _cluster_tiles(self, all_tiles):
        """Cluster tiles using perceptual hashing."""
        print(f"🔗 Clustering {len(all_tiles)} uniform tiles using perceptual hashing...")
        
        # Show tile dimensions (they should all be the same now)
        if all_tiles:
            sample_dims = all_tiles[0]['dimensions']
            print(f"  📏 All tiles normalized to: {sample_dims[0]}x{sample_dims[1]} pixels")
        
        # DEBUG: Save a few sample tiles to inspect
        debug_dir = self.tiles_dir / "debug_samples"
        debug_dir.mkdir(exist_ok=True)
        for i, tile_data in enumerate(all_tiles[:12]):  # Save first 12 tiles
            debug_path = debug_dir / f"sample_{i}_{tile_data['source_name'][:10]}_{tile_data['position']}.png"
            cv.imwrite(str(debug_path), tile_data['image'])
        print(f"  🐛 Saved {min(12, len(all_tiles))} sample tiles to {debug_dir} for inspection")
        
        # Compute hash for each tile
        tile_hashes = []
        for i, tile_data in enumerate(all_tiles):
            tile_img = tile_data['image']
            
            # Convert to PIL Image for imagehash
            pil_img = Image.fromarray(cv.cvtColor(tile_img, cv.COLOR_BGR2RGB))
            
            # Compute both perceptual and average hashes for better clustering
            phash = imagehash.phash(pil_img)
            ahash = imagehash.average_hash(pil_img)
            
            tile_hashes.append({
                'phash': phash,
                'ahash': ahash,
                'tile_data': tile_data,
                'index': i
            })
        
        # DEBUG: Show some hash values and differences
        print(f"  🐛 Sample hash values:")
        for i in range(min(5, len(tile_hashes))):
            hash_obj = tile_hashes[i]
            print(f"    Tile {i}: phash={hash_obj['phash']}, ahash={hash_obj['ahash']}")
        
        if len(tile_hashes) >= 2:
            diff = tile_hashes[0]['phash'] - tile_hashes[1]['phash']
            print(f"  🐛 Sample hash difference (tile 0 vs 1): {diff} bits")
        
        # Cluster tiles with similar hashes
        clusters = []
        used_indices = set()
        
        for i, tile_hash in enumerate(tile_hashes):
            if i in used_indices:
                continue
                
            # Start new cluster
            cluster = [tile_hash]
            used_indices.add(i)
            
            # DEBUG: Track comparisons for first cluster
            debug_this_cluster = (i == 0)
            comparisons_made = 0
            matches_found = 0
            
            # Find similar tiles
            for j, other_hash in enumerate(tile_hashes):
                if j in used_indices:
                    continue
                    
                # Check if hashes are similar (≤ threshold bit difference)
                hash_diff = tile_hash['phash'] - other_hash['phash']
                comparisons_made += 1
                
                if hash_diff <= self.hash_threshold:
                    cluster.append(other_hash)
                    used_indices.add(j)
                    matches_found += 1
                    
                    if debug_this_cluster and matches_found <= 3:
                        print(f"  🐛 Match found: tile {i} vs {j}, diff={hash_diff} bits")
                        
            if debug_this_cluster:
                print(f"  🐛 First cluster: made {comparisons_made} comparisons, found {matches_found} matches")
                    
            clusters.append(cluster)
            
        print(f"🔗 Created {len(clusters)} clusters from {len(all_tiles)} normalized tiles")
        
        # Show cluster size distribution
        cluster_sizes = [len(cluster) for cluster in clusters]
        cluster_sizes.sort(reverse=True)
        print(f"  📊 Cluster sizes: {cluster_sizes[:10]}{'...' if len(cluster_sizes) > 10 else ''}")
        
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
    print("🎮 Automated Template Extractor for Match-3 Games")
    print("=" * 50)
    
    # Interactive game name input
    game_name = input("Enter the target game name (e.g., 'royal-match', 'candy-crush'): ").strip()
    
    if not game_name:
        print("❌ Game name cannot be empty")
        return
        
    # Clean up game name (replace spaces with hyphens, lowercase)
    game_name = game_name.lower().replace(" ", "-")
    print(f"🎯 Processing templates for: {game_name}")
    
    extractor = AutomatedTemplateExtractor(game_name)
    extractor.extract_templates()

if __name__ == "__main__":
    main() 