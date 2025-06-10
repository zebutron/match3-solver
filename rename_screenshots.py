#!/usr/bin/env python3
import os
from pathlib import Path

pieces_dir = Path('data/templates/royal-match/inputs/pieces')
files = list(pieces_dir.glob('Screenshot*.png'))
files.sort()

print(f'Found {len(files)} screenshot files')
for i, file in enumerate(files, 1):
    new_name = pieces_dir / f'{i}.png'
    print(f'Renaming {file.name} -> {new_name.name}')
    file.rename(new_name)

print('Done renaming files') 