import json

with open('02__23201641_23201143_23201481_23301518 (1).ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

with open('extracted_notebook.txt', 'w', encoding='utf-8') as out:
    for i, cell in enumerate(nb.get('cells', [])):
        out.write(f'\n--- Cell {i} ({cell.get("cell_type")}) ---\n')
        out.write(''.join(cell.get('source', [])))
        out.write('\n')
