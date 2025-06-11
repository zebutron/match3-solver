# Match3 Bot

🎯 **Automated Match-3 gameplay bot with dynamic grid detection and intelligent move selection.**

## ⚠️ 🤖 **IMPORTANT - FOR AI ASSISTANTS**

**TO RUN THE BOT SCRIPT, ALWAYS USE:**
```bash
cd vendor/match3-bot && python3 Python_match_3_bot_test.py
```

**❌ DO NOT try:** `python3 Python_match_3_bot_test.py` from the project root  
**✅ ALWAYS use:** `cd vendor/match3-bot && python3 Python_match_3_bot_test.py`

*The script is located in the `vendor/match3-bot/` subdirectory, NOT in the project root.*

## 🚀 Core Workflow

### 1. Template Processing Pipeline

Process raw piece images into edge-detected templates for robust matching:

```bash
# Place piece images in: data/templates/royal-match/inputs/pieces/
# Run the edge detection processor:
python3 edge_detection_template_processor.py

# Output: Processed templates in data/templates/royal-match/extracted/pieces/
```

**Features:**
- ✅ **Hue-focused edge detection** for color-to-color boundary detection
- ✅ **Multi-scale template optimization** with automatic scaling
- ✅ **Color-named templates** (R, G, B, O) for clear debugging
- ✅ **Robust against lighting variations** and obstacles

### 2. Gameplay Bot

Intelligent bot that dynamically detects board layout and executes optimal moves:

```bash
cd vendor/match3-bot
python3 Python_match_3_bot_test.py
```

**Features:**
- 🎯 **Dynamic grid detection** - automatically detects board dimensions (8x10, 9x12, etc.)
- 🧮 **Anchor-based grid inference** using high-confidence piece positions
- 🎲 **Smart move selection** with randomization (70% best, 30% random from top 3)
- 🚫 **Failed move tracking** prevents infinite retry loops
- 📍 **Precise coordinate mapping** with 2-4 pixel accuracy
- 🎮 **Interactive setup** with click-based or manual coordinate entry

## 📁 Project Structure

```
match3-solver/
├── edge_detection_template_processor.py    # Template processing pipeline
├── vendor/match3-bot/
│   └── Python_match_3_bot_test.py         # Main gameplay bot
├── data/templates/royal-match/
│   ├── inputs/pieces/                     # Raw piece images (input)
│   ├── extracted/pieces/                  # Processed templates (output)
│   └── coordinates.txt                    # Saved screen coordinates
├── requirements.txt                       # Python dependencies
└── README.md                             # This file
```

## 🛠️ Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Process templates:**
   - Add piece images to `data/templates/royal-match/inputs/pieces/`
   - Run `python3 edge_detection_template_processor.py`

3. **Run the bot:**
   ```bash
   cd vendor/match3-bot
   python3 Python_match_3_bot_test.py
   ```

## 🎯 How It Works

### Template Processing
1. **Edge Detection:** Converts piece images to hue-focused edge templates
2. **Scale Optimization:** Finds optimal template scaling for current puzzle
3. **Threshold Tuning:** Optimizes matching thresholds per piece type

### Bot Operation
1. **Grid Detection:** Uses anchor pieces to infer complete grid structure
2. **Piece Recognition:** Matches templates against board with confidence scoring
3. **Move Analysis:** Finds all legal 3/4/5-matches with pattern recognition
4. **Smart Selection:** Prioritizes moves with randomization and failed-move avoidance
5. **Execution:** Precise screen coordinate mapping for reliable move execution

## 🎮 Supported Games

- ✅ **Royal Match** - Full support with dynamic grid detection
- 🔧 **Extensible** - Template system works with any match-3 game

## 🏆 Key Achievements

- **Dynamic grid detection** replacing hardcoded assumptions
- **Anchor-based coordinate inference** for precise positioning  
- **Smart move selection** with anti-loop protection
- **Robust template matching** against lighting/obstacle variations
- **Multi-board support** automatically adapts to different level types

---

*Built with computer vision, template matching, and intelligent game analysis.* 