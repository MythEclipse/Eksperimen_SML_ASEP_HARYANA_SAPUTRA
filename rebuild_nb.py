import json
import os

notebook_path = '/home/asephs/Pijak/Eksperimen_SML_ASEP_HARYANA_SAPUTRA/Eksperimen_SML_Full_ASEP_HARYANA_SAPUTRA.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# The new cleanly segmented blocks
code_import = [
    "import os\n",
    "import warnings\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import matplotlib.ticker as mticker\n",
    "import seaborn as sns\n",
    "import time\n",
    "import mlflow\n",
    "import mlflow.sklearn\n",
    "import joblib\n",
    "import shutil\n",
    "from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV\n",
    "from sklearn.preprocessing import StandardScaler, OneHotEncoder\n",
    "from sklearn.pipeline import Pipeline\n",
    "from sklearn.compose import ColumnTransformer\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score\n",
    "from imblearn.over_sampling import SMOTE\n",
    "from collections import Counter\n",
    "\n",
    "warnings.filterwarnings('ignore')\n",
    "sns.set_theme(style='whitegrid', palette='muted')\n",
    "plt.rcParams['figure.dpi'] = 120\n",
    "print('Libraries loaded successfully.')\n"
]

code_feat_split = [
    "# 1. Feature Engineering (Derived Features)\n",
    "print(\"Adding derived features...\")\n",
    "df_feat = df_train.copy()\n",
    "SUIT_COLS = ['S1', 'S2', 'S3', 'S4', 'S5']\n",
    "RANK_COLS = ['C1', 'C2', 'C3', 'C4', 'C5']\n",
    "DERIVED_COLS = ['all_same_suit', 'n_unique_suits', 'n_unique_ranks', 'rank_range']\n",
    "\n",
    "df_feat['all_same_suit'] = (df_feat[SUIT_COLS].nunique(axis=1) == 1).astype(int)\n",
    "df_feat['n_unique_suits'] = df_feat[SUIT_COLS].nunique(axis=1)\n",
    "df_feat['n_unique_ranks'] = df_feat[RANK_COLS].nunique(axis=1)\n",
    "df_feat['rank_range'] = df_feat[RANK_COLS].max(axis=1) - df_feat[RANK_COLS].min(axis=1)\n",
    "\n",
    "X = df_feat.drop(columns=['CLASS'])\n",
    "y = df_feat['CLASS']\n",
    "\n",
    "# 2. Train-Test Split (80/20 Stratified)\n",
    "print(\"Splitting data (80/20)...\")\n",
    "X_train_raw, X_test_raw, y_train, y_test = train_test_split(\n",
    "    X, y, test_size=0.2, random_state=42, stratify=y\n",
    ")\n"
]

code_norm = [
    "print(\"Defining StandarScaler (Normalisasi)...\")\n",
    "rank_scale = StandardScaler()\n"
]

code_outlier = [
    "print(\"Dataset is completely constrained by poker rules (1-4 suit, 1-13 rank).\")\n",
    "print(\"Outliers are systematically impossible, proceeding.\")\n"
]

code_encode = [
    "print(\"Encoding and Scaling pipeline...\")\n",
    "preprocessor = ColumnTransformer(\n",
    "    transformers=[\n",
    "        ('suit_ohe', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), SUIT_COLS),\n",
    "        ('rank_scale', rank_scale, RANK_COLS + DERIVED_COLS)\n",
    "    ],\n",
    "    remainder='drop'\n",
    ")\n",
    "\n",
    "X_train_enc = preprocessor.fit_transform(X_train_raw)\n",
    "X_test_enc = preprocessor.transform(X_test_raw)\n",
    "\n",
    "ohe_names = preprocessor.named_transformers_['suit_ohe'].get_feature_names_out(SUIT_COLS).tolist()\n",
    "scale_names = RANK_COLS + DERIVED_COLS\n",
    "feature_names = ohe_names + scale_names\n"
]

code_binning = [
    "print(\"Binning (Pengelompokan Data) is skipped for Poker rules.\")\n"
]

code_smote = [
    "print(\"Applying SMOTE...\")\n",
    "smote = SMOTE(random_state=42, k_neighbors=3)\n",
    "X_train_balanced, y_train_balanced = smote.fit_resample(X_train_enc, y_train)\n",
    "\n",
    "# 5. Saving Preprocessed Data\n",
    "out_dir = './Membangun_model/pokerhand_preprocessing'\n",
    "os.makedirs(out_dir, exist_ok=True)\n",
    "\n",
    "pd.DataFrame(X_train_balanced, columns=feature_names).to_csv(os.path.join(out_dir, 'X_train.csv'), index=False)\n",
    "pd.DataFrame(X_test_enc, columns=feature_names).to_csv(os.path.join(out_dir, 'X_test.csv'), index=False)\n",
    "pd.DataFrame({'CLASS': y_train_balanced}).to_csv(os.path.join(out_dir, 'y_train.csv'), index=False)\n",
    "pd.DataFrame({'CLASS': y_test}).to_csv(os.path.join(out_dir, 'y_test.csv'), index=False)\n",
    "joblib.dump(preprocessor, os.path.join(out_dir, 'preprocessor.joblib'))\n",
    "print(\"Data and Preprocessor successfully saved to\", out_dir)\n",
    "\n",
    "before = Counter(y_train)\n",
    "after  = Counter(y_train_balanced)\n",
    "\n",
    "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n",
    "classes = sorted(before.keys())\n",
    "axes[0].bar(classes, [before[c] for c in classes], color=sns.color_palette('viridis', 10))\n",
    "axes[0].set_title('Before SMOTE')\n",
    "axes[0].set_yscale('log')\n",
    "\n",
    "axes[1].bar(classes, [after[c] for c in classes], color=sns.color_palette('plasma', 10))\n",
    "axes[1].set_title('After SMOTE (Balanced)')\n",
    "plt.show()\n"
]

code_model = [
    "import mlflow\n",
    "import mlflow.sklearn\n",
    "from sklearn.model_selection import StratifiedKFold, GridSearchCV\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "import time\n",
    "\n",
    "mlflow.sklearn.autolog(disable=True)\n",
    "mlflow.set_tracking_uri('./Membangun_model/mlruns')\n",
    "mlflow.set_experiment('poker-hand-tuning-local')\n",
    "\n",
    "param_grid = {\n",
    "    'n_estimators': [100, 200],\n",
    "    'max_depth': [10, 15],\n",
    "    'min_samples_split': [2, 5],\n",
    "}\n",
    "\n",
    "base_model = RandomForestClassifier(class_weight='balanced', n_jobs=-1, random_state=42)\n",
    "cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)\n",
    "\n",
    "t0 = time.time()\n",
    "grid_search = GridSearchCV(\n",
    "    estimator=base_model,\n",
    "    param_grid=param_grid,\n",
    "    cv=cv,\n",
    "    scoring='f1_weighted',\n",
    "    n_jobs=-1,\n",
    "    verbose=1,\n",
    "    refit=True,\n",
    "    return_train_score=True\n",
    ")\n",
    "print(\"Fitting GridSearchCV...\")\n",
    "grid_search.fit(X_train_balanced, y_train_balanced)\n",
    "\n",
    "best_params = grid_search.best_params_\n",
    "best_model = grid_search.best_estimator_\n",
    "print(f\"Best params : {best_params}\")\n"
]

code_eval = [
    "with mlflow.start_run(run_name='RandomForest_GridSearchCV') as run:\n",
    "    for k, v in best_params.items():\n",
    "        mlflow.log_param(f'best_{k}', v)\n",
    "        \n",
    "    y_pred = best_model.predict(X_test_enc)\n",
    "    acc = accuracy_score(y_test, y_pred)\n",
    "    f1_w = f1_score(y_test, y_pred, average='weighted')\n",
    "    \n",
    "    mlflow.log_metric('accuracy', acc)\n",
    "    mlflow.log_metric('f1_weighted', f1_w)\n",
    "    mlflow.sklearn.log_model(sk_model=best_model, artifact_path='best_model')\n",
    "    \n",
    "    report = classification_report(y_test, y_pred, digits=4)\n",
    "    print(report)\n",
    "    \n",
    "    fig, ax = plt.subplots(figsize=(10, 8))\n",
    "    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')\n",
    "    plt.title('Confusion Matrix - Best Model')\n",
    "    plt.show()\n",
    "    \n",
    "    local_export = './best_model_local'\n",
    "    if os.path.exists(local_export): shutil.rmtree(local_export)\n",
    "    mlflow.sklearn.save_model(sk_model=best_model, path=local_export)\n"
]

def new_code_cell(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source
    }

new_cells = []
for cell in nb['cells']:
    # Clean up any giant monolithic blocks we added previously
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if "4. Data Preprocessing & Saving" in source: continue
        if "Visualisasi sebelum vs sesudah SMOTE" in source: continue
        if "5. Hyperparameter Tuning & MLflow" in source: continue
        if "5.2 MLflow Logging Run" in source: continue
        if "from sklearn.ensemble import RandomForestClassifier" in source:
            if "rf_model.fit" in source: continue # Skip old basic RF
        if "# Prediksi pada data sample testing" in source: continue

        # Replace the imports cell entirely with our comprehensive one
        if "import pandas as pd" in source and "Library" not in source and "from sklearn.model_selection import train_test_split" in source:
            continue
            
    new_cells.append(cell)

    # Now, inject strategically after specific Markdown headers
    text = "".join(cell['source']).strip().split('\n')[0]
    if '# 2. Import Library' in text:
        new_cells.append(new_code_cell(code_import))
        
    if text.startswith('## 5.2 Menghapus Data Duplikat'):
        # Right after we do standard duplicate checking, add Feature Engineering & Splitting
        new_cells.append(new_code_cell(code_feat_split))
        
    if text.startswith('## 5.3 Normalisasi'):
        new_cells.append(new_code_cell(code_norm))
        
    if text.startswith('## 5.4 Deteksi'):
        new_cells.append(new_code_cell(code_outlier))
        
    if text.startswith('## 5.5 Encoding'):
        new_cells.append(new_code_cell(code_encode))
        
    if text.startswith('## 5.6 Binning'):
        new_cells.append(new_code_cell(code_binning))
        new_cells.append(new_code_cell(code_smote)) # SMOTE & Save run right after Binning
        
    if text.startswith('# 6. Modelling'):
        new_cells.append(new_code_cell(code_model))
        
    if text.startswith('# 7. Evaluasi'):
        new_cells.append(new_code_cell(code_eval))
        
nb['cells'] = new_cells

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
    f.write('\n')

print("Notebook architecture updated. Run jupyter_nbconvert now.")
