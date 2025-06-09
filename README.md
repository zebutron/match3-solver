# match3-solver

An iPhone-Mirroring Match-3 autopilot that automatically explores, plays, and captures video of match-3 puzzle games.

## Goals

Provide an autonomous pipeline to play and screen record match-3 games, which will...
• Starting from the game's home screen (or first tutorial), navigate game UI automatically, clicking through tutorials, and exploring new features and menus as they unlock throughout the first 30 levels, en route to the next puzzle level.
• Once the next puzzle is begun, hand off puzzle boards to the match-3 solver, play and complete the puzzle, and then resume non-solver control once the current puzzle is completed, in order to explore and record newly unlocked features, and then enter the next puzzle.
• Record videos of all of the above.

## Quick Start

```bash
pyenv install
pip install -r requirements.txt
python -m src.fsm_runner
``` 