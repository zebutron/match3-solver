# match3-solver

An iPhone-Mirroring Match-3 autopilot that automatically explores, plays, and captures video of match-3 puzzle games.

## Goals

Provide an autonomous pipeline to play and screen record match-3 games, which will...
• Starting from the game's home screen (or first tutorial), navigate game UI automatically, clicking through tutorials, and exploring new features and menus as they unlock throughout the first 30 levels, en route to the next puzzle level.
• Once the next puzzle is begun, hand off puzzle boards to the match-3 solver, play and complete the puzzle, and then resume non-solver control once the current puzzle is completed, in order to explore and record newly unlocked features, and then enter the next puzzle.
• Record videos of all of the above.

## Setup Workflow

### 1. Install Dependencies
```bash
pyenv install
pip install -r requirements.txt
```

### 2. Prepare Game Templates
For each match-3 game you want to automate, you need to extract tile templates:

**Step 1: Collect Screenshots**
- Take 2-3 clear screenshots of the game's puzzle board
- Save them as PNG/JPG files in: `data/templates/{game_name}/inputs/`
- Example: `data/templates/royal-match/inputs/screenshot1.png`

**Step 2: Extract Templates Automatically**
```bash
python -m src.template_creator royal-match
```

This automated tool will:
- 🔍 Detect the game board region in each screenshot
- ✂️ Slice the board into 8x8 grid tiles  
- 🔗 Cluster similar tiles using perceptual hashing
- 🚮 Filter out UI elements (tiles that appear only once)
- 💾 Save template exemplars to `data/templates/{game_name}/tiles/`

### 3. Run the Autopilot
```bash
python -m src.fsm_runner royal-match
```

## Architecture

- **`src/template_creator.py`** - Automated tile template extraction
- **`src/fsm_runner.py`** - Main state machine loop (coming soon)
- **`src/grab.py`** - Screen capture for iPhone mirroring
- **`src/navigator.py`** - UI automation and menu navigation  
- **`src/solver.py`** - Wrapper around match3-bot solver
- **`src/states.py`** - Game state detection
- **`src/recorder.py`** - Video recording pipeline
- **`vendor/match3-bot/`** - Core match-3 solving algorithms
- **`data/templates/{game}/`** - Game-specific tile templates

## Supported Games

Add any match-3 game by running the template extraction workflow:
- Royal Match
- Candy Crush Saga  
- Bejeweled
- _(Any match-3 game with clear tile graphics)_

## Dependencies

- **OpenCV** - Image processing and template matching
- **imagehash** - Perceptual hashing for tile clustering  
- **Pillow** - Image manipulation
- **pyautogui** - Mouse/keyboard automation
- **mss** - Fast screen capture
- **ffmpeg-python** - Video recording 