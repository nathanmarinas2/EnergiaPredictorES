import json

file_path = r"c:/Users/usuario/Desktop/Universidad/proyectos/electricidad_españa/notebooks/EnergiaPredictorES_Colab.ipynb"

with open(file_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

modified = False
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        new_source = []
        changes_in_cell = False
        
        for line in source:
            if "from darts import TimeSeries" in line and "from darts.utils.missing_values" not in "".join(source):
                 new_source.append(line)
                 new_source.append("from darts.utils.missing_values import fill_missing_values\n")
                 changes_in_cell = True
            elif "series = series.interpolate()" in line:
                new_source.append("series = fill_missing_values(series, fill='auto')\n")
                changes_in_cell = True
            elif "future_covariates = future_covariates.interpolate()" in line:
                new_source.append("future_covariates = fill_missing_values(future_covariates, fill='auto')\n")
                changes_in_cell = True
            else:
                new_source.append(line)
        
        if changes_in_cell:
            cell['source'] = new_source
            modified = True

if modified:
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=2)
    print("Notebook updated successfully")
else:
    print("No changes needed")
