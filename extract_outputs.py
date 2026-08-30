import json
import re

with open('02__23201641_23201143_23201481_23301518 (1).ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

with open('extracted_outputs.txt', 'w', encoding='utf-8') as out:
    for cell in nb.get('cells', []):
        if cell.get('cell_type') == 'code':
            source = ''.join(cell.get('source', []))
            if 'Test Accuracy' in source or 'Test Macro F1' in source or 'BERT 2 Test Accuracy' in source:
                out.write('--- SOURCE ---\n' + source + '\n')
                out.write('--- OUTPUT ---\n')
                for output in cell.get('outputs', []):
                    if output.get('output_type') == 'stream':
                        out.write(''.join(output.get('text', [])))
                    elif output.get('output_type') == 'execute_result' or output.get('output_type') == 'display_data':
                        data = output.get('data', {})
                        if 'text/plain' in data:
                            out.write(''.join(data['text/plain']))
                out.write('\n\n')
