import os
from music21 import corpus
from config import DATA_DIR

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

print(f"Extracting Bach chorales to {DATA_DIR}...")

# Get all Bach chorale paths from the corpus
bach_paths = corpus.getComposer('bach')

count = 0
for path in bach_paths:
    # We only want the 4-part chorales (usually .xml or .mxl in the corpus)
    # We parse them and write them out as .mid
    try:
        song = corpus.parse(path)
        filename = os.path.basename(path).replace('.mxl', '.mid').replace('.xml', '.mid')
        
        # Save to your folder
        song.write('midi', fp=os.path.join(DATA_DIR, filename))
        count += 1
        print(f"Saved {filename}")
    except:
        pass

print(f"Done! Extracted {count} chorales.")