import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import joblib
import os
import sys
import math
from typing import List, Tuple
from sklearn.metrics import r2_score, mean_squared_error

# 导入模块
from NARX import NARXNet
from LSTM import LSTMModel, predict_with_model

# ================================
# 路径设置：基于项目目录结构
# ================================
BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # DEMO1_Yogurt_Viscosity_Prediction
MODEL_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 定义 create_narx_dataset
def create_narx_dataset(groups_prepared: List[pd.DataFrame], ny: int = 5, nu: int = 5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
            ylags = [y[t-d] for d in range(1, ny+1)]
            ulags = [u_sr[t], u_el[t]]
            for d in range(1, nu+1):
                ulags.append(u_sr[t-d])
                ulags.append(u_el[t-d])
            X_list.append(np.array(ylags + ulags, dtype=np.float32))
            y_list.append(float(y[t]))
            group_idx_list.append(gind)

    if len(X_list) == 0:
        return np.empty((0, ny + 2*(nu+1)), dtype=np.float32), np.empty((0,), dtype=np.float32), np.empty((0,), dtype=int)

    X = np.vstack(X_list).astype(np.float32)
    y = np.array(y_list, dtype=np.float32)
    groups = np.array(group_idx_list, dtype=int)
    return X, y, groups

# 定义 load_and_prepare_sheet
def load_and_prepare_sheet(xls: str, sheet_name: str, ny: int = 5, nu: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_excel(xls, sheet_name=sheet_name)
    df = df[['time', 'shear_rate', 'shear_stress']].dropna().copy()
    df = df.sort_values('time').reset_index(drop=True)
    if len(df) <= max(ny, nu):
        raise ValueError(f"数据量不足：样本数 {len(df)} 小于 max(ny={ny}, nu={nu})")
    df['time'] = df['time'] - df['time'].iloc[0]
    df['shear_rate'] = (df['shear_rate'] - df['shear_rate'].mean()) / (df['shear_rate'].std() + 1e-8)
    df['time'] = (df['time'] - df['time'].mean()) / (df['time'].std() + 1e-8)
    X, y, _ = create_narx_dataset([df], ny=ny, nu=nu)
    return X, y

# 融合预测函数（完全闭环）
def hybrid_closed_loop_predict_fixed(
    narx_model, lstm_preds,
    X_narx, df_predict_narx, df_predict_lstm,
    ny, nu, x_scaler, y_scaler, device="cpu", alpha_init=1.0, alpha_min=0.01, decay_rate=0.03, seq_len=5,
    debug_out_path=os.path.join(OUTPUT_DIR, "hybrid_deltas_fixed.xlsx")

):
    import numpy as np
    import torch
    import pandas as pd
    import os
    import math

    # 真实值（raw）
    y_true = df_predict_lstm["shear_stress"].values.astype(np.float32)
    T = len(y_true)
    max_lag = max(ny, nu)
    if T <= max_lag:
        raise ValueError(f"样本点不足，至少需要 {max_lag + 1} 个点，当前 {T} 个点")

    # scaled 版本（用于 delta 计算）
    y_scaled = y_scaler.transform(y_true.reshape(-1, 1)).flatten()

    # 初始化历史 for hybrid（统一到max_lag）
    y_hist_raw = y_true[:max_lag].astype(np.float32).copy()
    y_hybrid_scaled = list(y_scaled[:max_lag].tolist())

    # 初始化历史 for NARX closed-loop only
    y_hist_raw_narx = y_true[:max_lag].astype(np.float32).copy()
    y_narx_scaled_hist = list(y_scaled[:max_lag].tolist())

    logs_hybrid = []
    logs_narx = []

    # 假设 NARX 输出是 scaled
    print("[Info] 假设 NARX 输出为 scaled，并使用 y_scaler.inverse_transform 转为 raw")

    # --- 闭环预测 ---
    for t in range(max_lag, T):
        # 通用 ulags（近到远：当前 + 滞后）
        ulags = [df_predict_narx['shear_rate'].values[t],
                 df_predict_narx['time'].values[t]]
        for d in range(1, nu + 1):
            if t - d >= 0:
                ulags.append(df_predict_narx['shear_rate'].values[t - d])
                ulags.append(df_predict_narx['time'].values[t - d])
            else:
                ulags.append(0.0)
                ulags.append(0.0)

        # --- NARX only closed-loop ---
        ylags_narx = list(reversed(y_hist_raw_narx[-ny:]))  # 近到远 [y[t-1], ..., y[t-ny]]
        inp_narx = np.array(ylags_narx + ulags, dtype=np.float32)
        inp_scaled_narx = x_scaler.transform(inp_narx.reshape(1, -1)).flatten()
        inp_tensor_narx = torch.tensor(inp_scaled_narx, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            y_narx_only_scaled = float(narx_model(inp_tensor_narx).cpu().numpy().flatten()[0])
        y_narx_only_raw = float(y_scaler.inverse_transform([[y_narx_only_scaled]])[0, 0])
        delta_n_only_scaled = y_narx_only_scaled - y_narx_scaled_hist[-1]
        y_n_only_scaled = y_narx_scaled_hist[-1] + delta_n_only_scaled  # 等同 y_narx_only_scaled
        y_n_only_raw = y_narx_only_raw  # same
        # 更新历史 for NARX only
        y_narx_scaled_hist.append(y_n_only_scaled)
        y_hist_raw_narx = np.roll(y_hist_raw_narx, -1)
        y_hist_raw_narx[-1] = y_n_only_raw
        # 记录日志 for NARX only
        logs_narx.append({
            "t": t,
            "y_true_raw": float(y_true[t]),
            "y_narx_raw": y_n_only_raw,
            "delta_n_scaled": delta_n_only_scaled,
        })

        # --- Hybrid ---
        ylags_hybrid = list(reversed(y_hist_raw[-ny:]))  # 近到远 [y[t-1], ..., y[t-ny]]
        inp_hybrid = np.array(ylags_hybrid + ulags, dtype=np.float32)
        inp_scaled_hybrid = x_scaler.transform(inp_hybrid.reshape(1, -1)).flatten()
        inp_tensor_hybrid = torch.tensor(inp_scaled_hybrid, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            y_narx_scaled = float(narx_model(inp_tensor_hybrid).cpu().numpy().flatten()[0])
        y_narx_raw = float(y_scaler.inverse_transform([[y_narx_scaled]])[0, 0])

        # 动态计算 alpha（指数衰减）
        alpha = max(alpha_min, alpha_init * math.exp(-decay_rate * (t - max_lag)))

        # delta 融合
        y_prev_scaled = y_hybrid_scaled[-1]
        lstm_index = t - seq_len
        if lstm_index >= 1 and lstm_index < len(lstm_preds):
            y_lstm_t_scaled = float(y_scaler.transform([[float(lstm_preds[lstm_index])]])[0, 0])
            y_lstm_prev_scaled = float(y_scaler.transform([[float(lstm_preds[lstm_index - 1])]])[0, 0])
            delta_l_scaled = y_lstm_t_scaled - y_lstm_prev_scaled
        else:
            delta_l_scaled = 0.0

        delta_n_scaled = y_narx_scaled - y_prev_scaled
        delta_h_scaled = alpha * delta_n_scaled + (1 - alpha) * delta_l_scaled
        y_h_scaled = y_prev_scaled + delta_h_scaled
        y_h_raw = float(y_scaler.inverse_transform([[y_h_scaled]])[0, 0])

        # 更新历史 for hybrid
        y_hybrid_scaled.append(y_h_scaled)
        y_hist_raw = np.roll(y_hist_raw, -1)
        y_hist_raw[-1] = y_h_raw

        # 记录日志 for hybrid，添加 alpha
        logs_hybrid.append({
            "t": t,
            "y_true_raw": float(y_true[t]),
            "y_narx_raw": y_narx_raw,
            "y_h_raw": y_h_raw,
            "delta_n_scaled": delta_n_scaled,
            "delta_l_scaled": delta_l_scaled,
            "delta_h_scaled": delta_h_scaled,
            "alpha": alpha  # 新增：记录动态 alpha
        })

    # 输出 hybrid
    y_hybrid = np.array([y_true[i] for i in range(max_lag)] + [log["y_h_raw"] for log in logs_hybrid])
    # 输出 NARX closed-loop
    y_narx_closed = np.array([y_true[i] for i in range(max_lag)] + [log["y_narx_raw"] for log in logs_narx])

    # 写日志到同一Excel的不同sheet
    os.makedirs(os.path.dirname(debug_out_path), exist_ok=True)
    with pd.ExcelWriter(debug_out_path) as writer:
        pd.DataFrame(logs_hybrid).to_excel(writer, sheet_name="Hybrid", index=False)
        pd.DataFrame(logs_narx).to_excel(writer, sheet_name="NARX_Closed", index=False)
    print(f"[Info] 日志写入 {debug_out_path}")

    return np.arange(T), y_true, y_hybrid, y_narx_closed, logs_hybrid, logs_narx

# 主函数
def main():
    excel_path = os.path.join(DATA_DIR, "yogurt.xlsx")
    sheet_predict = "Sheet5"  # 注意大小写和 Excel 文件一致
    df = pd.read_excel(excel_path, sheet_name=sheet_predict)

    # 1. NARX 数据加载（仅用于初始化输入维度）
    try:
        X_narx, _ = load_and_prepare_sheet(excel_path, sheet_predict, ny=5, nu=5)
        print("X_narx 形状：", X_narx.shape)
    except Exception as e:
        print(f"数据加载错误：{e}")
        raise

    # 2. 加载原始数据
    df_raw = pd.read_excel(excel_path, sheet_name=sheet_predict)
    df_raw["time"] = df_raw["time"] - df_raw["time"].min()

    # --- LSTM 输入数据 ---
    df_predict_lstm = pd.DataFrame({
        "time": df_raw["time"].values,
        "shear_rate": df_raw["shear_rate"].values,
        "shear_stress": df_raw["shear_stress"].values
    })

    # --- NARX 输入数据（per-sheet 标准化，保持与训练一致）---
    df_predict_narx = df_raw.copy()
    df_predict_narx['shear_rate'] = (
        df_predict_narx['shear_rate'] - df_predict_narx['shear_rate'].mean()
    ) / (df_predict_narx['shear_rate'].std() + 1e-8)
    df_predict_narx['time'] = (
        df_predict_narx['time'] - df_predict_narx['time'].mean()
    ) / (df_predict_narx['time'].std() + 1e-8)

    # 3. 加载 NARX 模型及 scaler
    narx_meta = joblib.load(os.path.join(MODEL_DIR, "narx_scalers_meta.joblib"))
    x_scaler = narx_meta["x_scaler"]
    y_scaler = narx_meta["y_scaler"]
    meta = narx_meta["meta"]
    ny, nu = meta["ny"], meta["nu"]

    narx_model = NARXNet(in_dim=17, h=10)
    narx_model.load_state_dict(
        torch.load(os.path.join(MODEL_DIR, "narx_pytorch_br.pt"), map_location="cpu")
    )
    narx_model.eval()

    # 4. 加载 LSTM 模型及 scaler
    lstm_meta = joblib.load(os.path.join(MODEL_DIR, "lstm_meta_1.joblib"))
    seq_len = lstm_meta["seq_len"]
    scaler_X = lstm_meta["scaler_X"]
    scaler_y = lstm_meta["scaler_y"]

    lstm_model = LSTMModel(input_size=2, hidden_size=64, num_layers=2)
    lstm_model.load_state_dict(
        torch.load(os.path.join(MODEL_DIR, "lstm_model_1.pth"), map_location="cpu")
    )
    lstm_model.eval()

    # LSTM 输入准备
    X_new_scaled = scaler_X.transform(
        df_predict_lstm[["shear_rate", "time"]].values.astype(np.float32)
    )
    lstm_preds = predict_with_model(
        lstm_model, X_new_scaled, seq_len, scaler_y, device="cpu"
    )

    # ⚠️ 把 LSTM 预测转换到标准化空间（可用于 debug 对比）
    lstm_preds_scaled = y_scaler.transform(lstm_preds.reshape(-1, 1)).flatten()

    # 5. 混合预测 + 输出 delta_n / delta_l / delta_h
    debug_out_path = os.path.join(OUTPUT_DIR, "hybrid_deltas_fixed.xlsx")
    with pd.ExcelWriter(debug_out_path) as writer:
        df.to_excel(writer, index=False, sheet_name="Predictions")
    times, trues, hybrid_preds, narx_closed_preds, logs_hybrid, logs_narx = hybrid_closed_loop_predict_fixed(
        narx_model,
        lstm_preds,
        X_narx,
        df_predict_narx,
        df_predict_lstm,
        ny,
        nu,
        x_scaler,
        y_scaler,
        device="cpu",
        seq_len=seq_len,
        debug_out_path=debug_out_path
    )

    # 填充 LSTM 预测到完整序列（lstm_preds[0] 对应 y[seq_len]，前 seq_len 个为 NaN）
    T = len(trues)
    max_lag = max(ny, nu)
    lstm_full = np.full(T, np.nan)
    lstm_full[seq_len:] = lstm_preds

    # 6. 绘图：添加 LSTM 曲线
    plt.figure(figsize=(12, 7))
    plt.plot(times, trues, label="True")
    plt.plot(times, hybrid_preds, label="Hybrid Closed-loop")
    plt.plot(times, narx_closed_preds, label="NARX Closed-loop")
    plt.plot(times, lstm_full, label="LSTM", linestyle='--')
    plt.legend()
    plt.xlabel("Time step")
    plt.ylabel("Shear Stress")
    plt.title("Hybrid NARX-LSTM Closed-loop Prediction")
    plt.show()

    # 7. 记录所有数据到 Excel 的新 sheet "All_Models"
    # 创建完整 DataFrame，从 t=0 到 T-1
    all_logs = []
    for t in range(T):
        row = {
            "t": t,
            "y_true_raw": trues[t],
            "y_narx_raw": np.nan,
            "y_lstm_raw": lstm_full[t],
            "y_h_raw": np.nan,
            "delta_n_scaled": np.nan,
            "delta_l_scaled": np.nan,
            "delta_h_scaled": np.nan,
            "alpha": np.nan
        }
        if t < max_lag:
            # 早期使用 true 值填充预测
            row["y_narx_raw"] = trues[t]
            row["y_h_raw"] = trues[t]
        else:
            # 从 max_lag 开始，使用 logs
            narx_log = next((log for log in logs_narx if log["t"] == t), None)
            hybrid_log = next((log for log in logs_hybrid if log["t"] == t), None)
            if narx_log:
                row["y_narx_raw"] = narx_log["y_narx_raw"]
                row["delta_n_scaled"] = narx_log["delta_n_scaled"]
            if hybrid_log:
                row["y_h_raw"] = hybrid_log["y_h_raw"]
                row["delta_n_scaled"] = hybrid_log["delta_n_scaled"]  # Hybrid 也有 delta_n
                row["delta_l_scaled"] = hybrid_log["delta_l_scaled"]
                row["delta_h_scaled"] = hybrid_log["delta_h_scaled"]
                row["alpha"] = hybrid_log["alpha"]
        all_logs.append(row)

    all_logs_df = pd.DataFrame(all_logs)
    with pd.ExcelWriter(debug_out_path, mode='a', if_sheet_exists='replace') as writer:
        all_logs_df.to_excel(writer, sheet_name="All_Models", index=False)
    print(f"[Info] 所有数据记录写入 {debug_out_path} 的 All_Models sheet")

    # 8. 计算 R2 和 RMSE（从 max(max_lag, seq_len) 开始，确保所有模型都有预测值）
    eval_start = max(max_lag, seq_len)
    y_true_eval = trues[eval_start:]
    y_narx_eval = narx_closed_preds[eval_start:]
    y_hybrid_eval = hybrid_preds[eval_start:]
    y_lstm_eval = lstm_full[eval_start:]

    # 移除 NaN（虽 eval_start 已避免，但以防）
    valid_mask = ~np.isnan(y_lstm_eval)
    y_true_eval = y_true_eval[valid_mask]
    y_narx_eval = y_narx_eval[valid_mask]
    y_hybrid_eval = y_hybrid_eval[valid_mask]
    y_lstm_eval = y_lstm_eval[valid_mask]

    if len(y_true_eval) > 1:  # 确保有足够数据计算
        r2_narx = r2_score(y_true_eval, y_narx_eval)
        rmse_narx = np.sqrt(mean_squared_error(y_true_eval, y_narx_eval))

        r2_lstm = r2_score(y_true_eval, y_lstm_eval)
        rmse_lstm = np.sqrt(mean_squared_error(y_true_eval, y_lstm_eval))

        r2_hybrid = r2_score(y_true_eval, y_hybrid_eval)
        rmse_hybrid = np.sqrt(mean_squared_error(y_true_eval, y_hybrid_eval))

        print(f"R2 (NARX): {r2_narx:.4f}, RMSE (NARX): {rmse_narx:.4f}")
        print(f"R2 (LSTM): {r2_lstm:.4f}, RMSE (LSTM): {rmse_lstm:.4f}")
        print(f"R2 (Hybrid): {r2_hybrid:.4f}, RMSE (Hybrid): {rmse_hybrid:.4f}")

        # 9. 绘制柱状图比较
        models = ['NARX', 'LSTM', 'Hybrid']
        r2_values = [r2_narx, r2_lstm, r2_hybrid]
        rmse_values = [rmse_narx, rmse_lstm, rmse_hybrid]

        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        axs[0].bar(models, r2_values, color=['blue', 'green', 'orange'])
        axs[0].set_title('R2 Score Comparison')
        axs[0].set_ylabel('R2 Score')
        axs[0].set_ylim(min(r2_values) - 0.1, max(r2_values) + 0.1)

        axs[1].bar(models, rmse_values, color=['blue', 'green', 'orange'])
        axs[1].set_title('RMSE Comparison')
        axs[1].set_ylabel('RMSE')
        axs[1].set_ylim(0, max(rmse_values) + 0.1)

        plt.tight_layout()
        plt.show()
    else:
        print("数据不足，无法计算 R2 和 RMSE")

if __name__ == "__main__":
    main()