import re
import nbformat

def find_legacy_strings(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
    
    legacy_candidates = set()
    # Pattern for quoted CamelCase strings which are typical of the old dataset
    pattern = re.compile(r"['\"]([A-Z][a-zA-Z]+)['\"]")
    
    for cell in nb.cells:
        if cell.cell_type == 'code':
            matches = pattern.findall(cell.source)
            legacy_candidates.update(matches)
            
    # Filter out common false positives
    exclusions = {'Logistic', 'Regression', 'Random', 'Forest', 'XGBoost', 'LightGBM', 'CatBoost', 'Optuna', 'True', 'False', 'None'}
    filtered = [s for s in legacy_candidates if s not in exclusions]
    return sorted(filtered)

if __name__ == "__main__":
    print(find_legacy_strings('churn_prediction_world_class.ipynb'))
