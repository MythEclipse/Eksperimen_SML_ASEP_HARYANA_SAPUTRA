import json
import subprocess

original_json = subprocess.check_output(['git', 'show', 'HEAD:Eksperimen_SML_Full_ASEP_HARYANA_SAPUTRA.ipynb']).decode('utf-8')
nb_old = json.loads(original_json)

with open('Eksperimen_SML_Full_ASEP_HARYANA_SAPUTRA.ipynb', 'r', encoding='utf-8') as f:
    nb_new = json.load(f)

old_sources = set(["".join(c['source']).strip() for c in nb_old['cells'] if ''.join(c['source']).strip() != ''])
new_sources = set(["".join(c['source']).strip() for c in nb_new['cells'] if ''.join(c['source']).strip() != ''])

missing = old_sources - new_sources
print(f"Number of deleted cells: {len(missing)}")
for idx, m in enumerate(missing):
    print(f"--- Deleted Cell {idx+1} ---")
    print(m[:200])
    print("-------------------------")
