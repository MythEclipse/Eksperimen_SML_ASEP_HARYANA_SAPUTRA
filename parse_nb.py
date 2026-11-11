import json
with open("Eksperimen_SML_Full_ASEP_HARYANA_SAPUTRA.ipynb", encoding='utf-8') as f:
    nb = json.load(f)

for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "markdown":
        text = "".join(cell["source"]).strip().split('\n')[0]
        print(f"[{i}] MD: {text[:80]}")
    elif cell["cell_type"] == "code":
        text = "".join(cell["source"]).strip().split('\n')[0]
        print(f"[{i}] CODE: {text[:80]}")
