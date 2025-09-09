"""
Food Structure–Function ML Demo: Modeling Example

This script demonstrates how to train and evaluate multiple machine learning models
to predict functional properties of whey protein powders from microstructural descriptors.
"""

# In[1]: Imports
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import RidgeCV
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor

import shap

# ================================
# 路径设置：基于项目目录结构
# ================================
BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # DEMO2_Whey_Protein_Functionality
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# In[2]: Load Data
excel_path = os.path.join(DATA_DIR, "protein_data.xlsx")  # 注意：请把Excel文件放在 data/ 下
df = pd.read_excel(excel_path, sheet_name="Sheet1")
print(df.head())

# In[3]: Define Features and Targets
micro_features = [c for c in df.columns if c in [
    "D [4,3]","D [3,2]","zeta_potential","helix1","helix2",
    "antiparalllel1","antiparalllel2","antiparalllel3","paralllel",
    "turn","coil","SH_conc.","Protein_conc","Fat_conc","Ash_conc",
    "Polysaccharide","Reducing_sugar","beta/alpha"
]]

macro_targets = [c for c in df.columns if c in [
    "water_holding","oil_holding","EAI","ESI","FC","FS",
    "lumisider_stability","min_gel_con."
]]

X = df[micro_features].values
Y = df[macro_targets].values

print("X shape:", X.shape, "Y shape:", Y.shape)

# In[4]: Train/Test Split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# In[5]: Define Models
models = {
    'PLS': PLSRegression(n_components=min(3, X_train.shape[1])),
    'Ridge': RidgeCV(alphas=[0.1, 1, 10]),
    'SVR': MultiOutputRegressor(SVR(kernel='rbf', C=1.0, epsilon=0.1)),
    'RF': RandomForestRegressor(n_estimators=200, random_state=42),
}

# In[6]: Train and Evaluate
results = {}

for name, model in models.items():
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('model', model)
    ])
    pipe.fit(X_train, Y_train)
    Y_pred = pipe.predict(X_test)
    rmse = np.sqrt(mean_squared_error(Y_test, Y_pred))
    r2 = r2_score(Y_test, Y_pred, multioutput='uniform_average')
    results[name] = {"rmse": rmse, "r2": r2}
    print(f"{name}: RMSE={rmse:.3f}, R2={r2:.3f}")

# In[7]: Compare Results
res_df = pd.DataFrame(results).T
res_df.plot(kind='bar', figsize=(8,5))
plt.ylabel("Score")
plt.title("Model Comparison (lower RMSE, higher R2 is better)")
plt.savefig(os.path.join(OUTPUT_DIR, "model_comparison.png"), dpi=150)
plt.close()

# In[8]: SHAP Example (Random Forest)
for i, target in enumerate(macro_targets):
    rf_single = RandomForestRegressor(n_estimators=200, random_state=42)
    rf_single.fit(X_train, Y_train[:, i])
    explainer = shap.TreeExplainer(rf_single)
    shap_values = explainer.shap_values(X_test)
    shap.summary_plot(shap_values, X_test, feature_names=micro_features, show=False)
    plt.title(f"SHAP for target: {target}")
    plt.savefig(os.path.join(OUTPUT_DIR, f"shap_{target}.png"), dpi=150)
    plt.close()

# In[9]: Save Results
res_out_path = os.path.join(OUTPUT_DIR, "model_comparison.xlsx")
res_df.to_excel(res_out_path)
print(f"Results saved to {res_out_path}")
