# narx_pytorch_br.py
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, List

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import mean_squared_error
import joblib

# ----------------------------
# 读入单个 sheet 并预处理：排序、计算组内 elapsed time（从0开始）
# ----------------------------
def load_and_prepare_sheet(xls, sheet_name):
    df = pd.read_excel(xls, sheet_name=sheet_name)
    df = df[['time', 'shear_rate', 'shear_stress']].dropna().copy()
    df = df.sort_values('time').reset_index(drop=True)
    df['time'] = df['time'] - df['time'].iloc[0]
    # 组内标准化
    df['shear_rate'] = (df['shear_rate'] - df['shear_rate'].mean()) / (df['shear_rate'].std() + 1e-8)
    df['time'] = (df['time'] - df['time'].mean()) / (df['time'].std() + 1e-8)
    return df
# ----------------------------
# 构建 NARX 特征矩阵：ny, nu 固定为 5
# X: [y(t-1..t-3), u_sr(t), u_el(t), u_sr(t-1..t-3), u_el(t-1..t-3)]
# ----------------------------
def create_narx_dataset(groups_prepared: List[pd.DataFrame], ny: int = 5, nu: int = 5):
    X_list, y_list, group_idx_list = [], [], []
    max_lag = max(ny, nu)
    for gind, df in enumerate(groups_prepared):
        y = df['shear_stress'].values
        u_sr = df['shear_rate'].values
        u_el = df['time'].values
        T = len(y)
        if T <= max_lag:
            continue
        for t in range(max_lag, T):
            # 过去 ny 个 y
            ylags = [y[t-d] for d in range(1, ny+1)]

            # 当前时刻的 u(t)
            ulags = [u_sr[t], u_el[t]]

            # 再加上过去 nu 个 u(t-d)
            for d in range(1, nu+1):
                ulags.append(u_sr[t-d])
                ulags.append(u_el[t-d])

            # 合并特征
            X_list.append(np.array(ylags + ulags, dtype=np.float32))
            y_list.append(float(y[t]))
            group_idx_list.append(gind)

    if len(X_list) == 0:
        return np.empty((0, ny + 2*(nu+1)), dtype=np.float32), np.empty((0,), dtype=np.float32), np.empty((0,), dtype=int)

    X = np.vstack(X_list).astype(np.float32)
    y = np.array(y_list, dtype=np.float32)
    groups = np.array(group_idx_list, dtype=int)
    return X, y, groups

# ----------------------------
# NARX 两层 MLP：20-20, tanh, 线性输出
# ----------------------------
class NARXNet(nn.Module):
    def __init__(self, in_dim: int = 11, h: int = 20):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, h),
            nn.Tanh(),
            nn.Linear(h, h),
            nn.Tanh(),
            nn.Linear(h, 1)  # 线性输出
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())

def weights_l2_norm(model: nn.Module) -> torch.Tensor:
    s = 0.0
    for p in model.parameters():
        s = s + torch.sum(p ** 2)
    return s

# ----------------------------
# 简单梯度优化训练（使用 Adam，MSE + 固定 L2 正则化，支持早停）
# ----------------------------
def train_model(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    X_val: np.ndarray = None,
    y_val: np.ndarray = None,
    epochs: int = 3000,
    lr: float = 1e-4,
    alpha: float = 1e-5,  # 固定 L2 正则化强度
    patience: int = 100,  # 早停耐心值
    closed_loop_ratio: float = 0.2,  # 新参数：闭环训练比例
    device: str = "cpu",
    verbose: int = 200
):
    model.to(device)
    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    y_t = torch.tensor(y, dtype=torch.float32, device=device)
    N = len(y)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    history = {"loss": [], "mse": []}
    best_mse_val = np.inf
    patience_counter = 0

    if X_val is not None:
        X_val_t = torch.tensor(X_val, dtype=torch.float32, device=device)
        y_val_t = torch.tensor(y_val, dtype=torch.float32, device=device)

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        # 原 y_pred = model(X_t)
        # 修改：随机闭环
        y_pred = model(X_t)
        if np.random.rand() < closed_loop_ratio:
            # 使用预测作为部分输入（需调整X_t，但简化版用噪声）
            noise = torch.randn_like(y_pred) * 0.01
            y_pred += noise  # 添加噪声模拟闭环误差
        mse = criterion(y_pred, y_t)
        w2 = weights_l2_norm(model)
        loss = mse + alpha * w2
        loss.backward()
        optimizer.step()

        history["loss"].append(float(loss.item()))
        history["mse"].append(float(mse.item()))

        # 验证集早停
        if X_val is not None:
            model.eval()
            with torch.no_grad():
                y_pred_val = model(X_val_t)
                mse_val = criterion(y_pred_val, y_val_t).item()
            if mse_val < best_mse_val:
                best_mse_val = mse_val
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
            if verbose and epoch % verbose == 0:
                print(f"Epoch {epoch}/{epochs}: loss={loss.item():.6e} mse={mse.item():.6e} val_mse={mse_val:.6e}")
        else:
            if verbose and epoch % verbose == 0:
                print(f"Epoch {epoch}/{epochs}: loss={loss.item():.6e} mse={mse.item():.6e}")

    return history

# ----------------------------
# 闭环预测（使用目标与特征的 scaler；起步阶段用前 ny=5 个真实 y）
# ----------------------------
def predict_closed_loop_torch(model: nn.Module, df: pd.DataFrame, ny: int, nu: int,
                              x_scaler: StandardScaler, y_scaler: StandardScaler,
                              device: str = "cpu"):
    model.eval()
    y_true = df['shear_stress'].values.astype(np.float32)
    u_sr = df['shear_rate'].values.astype(np.float32)
    u_el = df['time'].values.astype(np.float32)
    T = len(y_true)
    max_lag = max(ny, nu)
    if T <= max_lag:
        return np.array([]), np.array([]), np.array([])

    # 历史缓冲：前 max_lag 个真实 y
    y_buf = list(y_true[:max_lag])
    preds, trues, times = [], [], []

    for t in range(max_lag, T):
        # 过去 ny 个 y
        ylags = [y_buf[-d] for d in range(1, ny+1)]

        # 加入当前时刻的 u(t)
        ulags = [u_sr[t], u_el[t]]

        # 再加上过去 nu 个 u(t-d)
        for d in range(1, nu+1):
            ulags.append(u_sr[t-d])
            ulags.append(u_el[t-d])

        # 合并特征
        x_raw = np.array(ylags + ulags, dtype=np.float32).reshape(1, -1)

        # 标准化
        x_std = x_scaler.transform(x_raw)
        x_t = torch.tensor(x_std, dtype=torch.float32, device=device)

        with torch.no_grad():
            y_hat_std = model(x_t).cpu().numpy().reshape(-1, 1)
        # 反标准化
        y_hat = y_scaler.inverse_transform(y_hat_std).ravel()[0]

        preds.append(float(y_hat))
        trues.append(float(y_true[t]))
        times.append(float(df['time'].iloc[t]))

        # 闭环：把预测值推进缓冲
        y_buf.append(float(y_hat))

    return np.array(times), np.array(trues), np.array(preds)

# ----------------------------
# 主流程
# ----------------------------
def main():
    excel_path = "data/yogurt.xlsx"
    if not os.path.exists(excel_path):
        print(f"未找到文件 {excel_path}")
        return

    # 训练集（sheet1~sheet4），预测集（sheet5）
    sheets_train = [f"sheet{i}" for i in range(1, 5)]
    sheet_predict = "sheet5"

    groups_prepared = []
    for s in sheets_train:
        dfp = load_and_prepare_sheet(excel_path, s)
        groups_prepared.append(dfp)
        print(f"已读取训练 {s}，记录数：{len(dfp)}, time: {dfp['time'].iloc[0]} -> {dfp['time'].iloc[-1]}")

    df_predict = load_and_prepare_sheet(excel_path, sheet_predict)
    print(f"已读取预测 {sheet_predict}，记录数：{len(df_predict)}, time: {dfp['time'].iloc[0]} -> {dfp['time'].iloc[-1]}")

    ny, nu = 5, 5
    # ----------------------------
    # 训练集特征生成（每个 sheet 单独生成）
    # ----------------------------
    X_list, y_list, groups_list = [], [], []

    for gind, df in enumerate(groups_prepared):
        Xg, yg, _ = create_narx_dataset([df], ny=ny, nu=nu)  # 单个 sheet 构建
        if Xg.shape[0] == 0:
            continue
        X_list.append(Xg)
        y_list.append(yg)
        groups_list.append(np.full(len(yg), gind, dtype=int))

    # 合并所有 sheet
    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    groups = np.concatenate(groups_list)

    print(f"训练样本总数: {X.shape[0]}, 特征维度: {X.shape[1]}, 组数: {len(np.unique(groups))}")

    if X.shape[0] == 0:
        print("数据集为空，无法训练")
        return

    # 标准化（X、y 都做，回归更稳）
    x_scaler = StandardScaler().fit(X)
    y_scaler = StandardScaler().fit(y.reshape(-1, 1))
    Xs = x_scaler.transform(X)
    ys = y_scaler.transform(y.reshape(-1, 1)).ravel()

    # （可选）LOGO 评估，但不做网格搜索
    logo = LeaveOneGroupOut()
    mses = []
    for tr_idx, te_idx in logo.split(Xs, ys, groups):
        model_cv = NARXNet(in_dim=17, h=20)
        train_model(model_cv, Xs[tr_idx], ys[tr_idx],
                    X_val=Xs[te_idx], y_val=ys[te_idx],
                    epochs=1000, lr=1e-3, alpha=1e-4, device="cpu", verbose=100)
        with torch.no_grad():
            y_pred_te = model_cv(torch.tensor(Xs[te_idx], dtype=torch.float32)).numpy()
        # 反标准化后评估 MSE
        y_pred_te_denorm = y_scaler.inverse_transform(y_pred_te.reshape(-1, 1)).ravel()
        y_te_denorm = y_scaler.inverse_transform(ys[te_idx].reshape(-1, 1)).ravel()
        mses.append(mean_squared_error(y_te_denorm, y_pred_te_denorm))
    if mses:
        print(f"LOGO MSE (no grid): mean={np.mean(mses):.6e}, std={np.std(mses):.6e}")

    # 使用所有训练样本训练最终模型
    model = NARXNet(in_dim=17, h=10)
    print("开始训练最终模型（梯度优化）...")
    hist = train_model(model, Xs, ys, epochs=4000, lr=1e-3, alpha=1e-4, device="cpu", verbose=200)

    # 保存模型与标准化器
    meta = {"ny": ny, "nu": nu, "hidden": [10, 10]}
    torch.save(model.state_dict(), "narx_pytorch_br.pt")
    joblib.dump({"x_scaler": x_scaler, "y_scaler": y_scaler, "meta": meta}, "narx_scalers_meta.joblib")
    with open("narx_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print("模型与元信息已保存：narx_pytorch_br.pt / narx_scalers_meta.joblib / narx_meta.json")


    # ---- 闭环预测并作图（sheet5） ----
    loaded_scalers = joblib.load("narx_scalers_meta.joblib")
    xsc = loaded_scalers["x_scaler"]
    ysc = loaded_scalers["y_scaler"]

    # 加载模型参数
    model_loaded = NARXNet(in_dim=17, h=10)
    model_loaded.load_state_dict(torch.load("narx_pytorch_br.pt", map_location="cpu"))

    times, trues, preds = predict_closed_loop_torch(model_loaded, df_predict, ny, nu, xsc, ysc, device="cpu")
    if len(times) == 0:
        print("预测组数据太短，无法进行闭环预测。")
        return

    plt.figure(figsize=(10, 6))
    plt.plot(times, trues, label='real shear_stress')
    plt.plot(times, preds, '--', label='predict shear_stress')
    plt.xlabel('time')
    plt.ylabel('shear_stress')
    plt.title('sheet5: real vs predict')
    plt.legend()
    plt.tight_layout()
    plt.savefig("narx_sheet5_pred_vs_true.png", dpi=150)
    print("预测图已保存为 narx_sheet5_pred_vs_true.png")

if __name__ == "__main__":
    main()