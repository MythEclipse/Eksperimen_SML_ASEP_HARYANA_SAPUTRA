import json
import subprocess

# Get the original notebook content from git
original_json = subprocess.check_output(['git', 'show', 'HEAD:Eksperimen_SML_Full_ASEP_HARYANA_SAPUTRA.ipynb']).decode('utf-8')
nb = json.loads(original_json)

for i, cell in enumerate(nb['cells']):
    # check for markdown images
    if cell['cell_type'] == 'markdown':
        source = "".join(cell['source'])
        if '![' in source or '<img' in source:
            print(f"Markdown image found in cell {i}:\n{source[:200]}")
    
    # check for cell outputs that display images
    if cell['cell_type'] == 'code':
        for out in cell.get('outputs', []):
            if 'data' in out and 'image/png' in out['data']:
                source = "".join(cell['source'])
                print(f"Code cell {i} generated an image. Code snippet:\n{source[:200]}")
