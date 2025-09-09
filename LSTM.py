import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import joblib
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# =============================
# 1. Data Loading + Time Normalization
# =============================
def load_data_from_excel(file, sheet_name):
    try:
        if not os.path.exists(file):
            raise FileNotFoundError(f"Excel 文件 {file} 不存在。")
        df = pd.read_excel(file, sheet_name=sheet_name)
        required_columns = ["time", "shear_rate", "shear_stress"]
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"表格 {sheet_name} 缺少必需列：{required_columns}")

        if df.empty:
            raise ValueError(f"表格 {sheet_name} 为空。")

        # 时间归一化
        df["time"] = df["time"] - df["time"].min()

        # 输入特征：shear_rate + time
        X = df[["shear_rate", "time"]].values.astype(np.float32)
        # 确保 X 是二维的
        if X.ndim == 1:
            X = X.reshape(1, -1)
        # 目标：shear_stress
        y = df["shear_stress"].values.astype(np.float32)

        print(f"加载 {sheet_name}：X 形状={X.shape}, y 形状={y.shape}")
        return X, y
    except Exception as e:
        print(f"加载 {sheet_name} 时出错：{e}")
        raise

# =============================
# 2. PyTorch Dataset
# =============================
class SequenceDataset(Dataset):
    def __init__(self, X, y, seq_len):
        self.X = X  # Shape: (n_samples, seq_len, 2)
        self.y = y  # Shape: (n_samples,)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.X[idx], dtype=torch.float32),  # (seq_len, 2)
            torch.tensor(self.y[idx], dtype=torch.float32)   # scalar
        )

# =============================
# 3. LSTM Model
# =============================
class LSTMModel(nn.Module):
    def __init__(self, input_size=2, hidden_size=64, num_layers=2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)  # (batch, seq_len, hidden)
        out = out[:, -1, :]    # Take last time step
        out = self.fc(out)     # (batch, 1)
        return out.squeeze(-1) # (batch,)

# =============================
# 4. Training Function
# =============================
def train_model(model, train_loader, val_loader, lr=1e-3, epochs=50, device="cpu"):
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                val_loss += criterion(pred, yb).item()
        val_loss /= len(val_loader)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")

    return model

# =============================
# 5. Prediction Function
# =============================
def predict_with_model(model, X, seq_len, scaler_y, device="cpu"):
    model.eval()
    model = model.to(device)
    preds = []
    with torch.no_grad():
        for i in range(len(X) - seq_len):
            seq = torch.tensor(X[i:i+seq_len], dtype=torch.float32).view(1, seq_len, -1).to(device)
            yhat = model(seq).cpu().numpy().item()
            preds.append(yhat)
    # Inverse transform predictions
    preds = scaler_y.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
    return preds

# =============================
# 6. Plotting Function
# =============================
def plot_results(X_new, y_true, preds, seq_len, x_axis="time", save_path="output/lstm_pred_vs_true.png"):
    plt.figure(figsize=(8, 5))
    if x_axis == "time":
        x_vals = X_new[seq_len:, 1]  # time
        plt.xlabel("Time (s)")
    elif x_axis == "shear_rate":
        x_vals = X_new[seq_len:, 0]  # shear_rate
        plt.xlabel("Shear Rate (1/s)")
    else:
        x_vals = np.arange(len(y_true))  # sample index
        plt.xlabel("Sample")

    plt.plot(x_vals, y_true, label="True")
    plt.plot(x_vals, preds, label="Predicted")
    plt.legend()
    plt.ylabel("Shear Stress")
    plt.title("LSTM Predicted vs True Shear Stress")
    plt.grid(True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Prediction plot saved as {save_path}")

# =============================
# 7. Main Function
# =============================
def main():
    # 配置
    config = {
        "file": "data/yogurt.xlsx",
        "sheets": ["sheet1", "sheet2", "sheet3", "sheet4"],
        "test_sheet": "sheet5",
        "seq_len": 5,  # 可根据数据调整
        "batch_size": 32,
        "hidden_size": 64,
        "num_layers": 2,
        "lr": 1e-3,
        "epochs": 50,
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }

    # 加载和预处理训练数据
    groups = []
    all_X, all_y = [], []
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    for sheet in config["sheets"]:
        try:
            X, y = load_data_from_excel(config["file"], sheet)
            if len(X) < config["seq_len"]:
                print(f"警告：表格 {sheet} 数据量 ({len(X)}) 小于 seq_len ({config['seq_len']})，跳过此表格。")
                continue
            all_X.append(X)
            all_y.append(y.reshape(-1, 1))
            groups.append((X, y))
        except Exception as e:
            print(f"跳过表格 {sheet}：{e}")
            continue

    if not groups:
        raise ValueError("没有有效的表格数据可用于训练。请检查 Excel 文件或表格内容。")

    # 拟合 scaler
    all_X = np.vstack(all_X)
    all_y = np.vstack(all_y)
    scaler_X.fit(all_X)
    scaler_y.fit(all_y)

    # 构建序列
    X_all, y_all = [], []
    for X, y in groups:
        # 确保 X 是二维的
        if X.ndim == 1:
            X = X.reshape(1, -1)
        X_scaled = scaler_X.transform(X)
        y_scaled = scaler_y.transform(y.reshape(-1, 1)).flatten()
        if len(X_scaled) < config["seq_len"]:
            print(f"跳过数据组，长度 {len(X_scaled)} 小于 seq_len {config['seq_len']}。")
            continue
        for i in range(len(X_scaled) - config["seq_len"] + 1):
            X_all.append(X_scaled[i:i + config["seq_len"]])
            y_all.append(y_scaled[i + config["seq_len"] - 1])
    X_all = np.array(X_all)
    y_all = np.array(y_all)

    if len(X_all) == 0:
        raise ValueError(f"无法生成有效序列。请检查数据长度或减小 seq_len（当前为 {config['seq_len']}）。")

    print(f"生成序列：X_all 形状={X_all.shape}, y_all 形状={y_all.shape}")

    # 分割训练和验证集
    X_train, X_val, y_train, y_val = train_test_split(X_all, y_all, test_size=0.2, random_state=42)

    # 创建数据集和加载器
    train_dataset = SequenceDataset(X_train, y_train, config["seq_len"])
    val_dataset = SequenceDataset(X_val, y_val, config["seq_len"])
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)

    # 训练模型
    model = LSTMModel(
        input_size=2,
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"]
    )
    model = train_model(
        model,
        train_loader,
        val_loader,
        lr=config["lr"],
        epochs=config["epochs"],
        device=config["device"]
    )

    # 保存模型和 scaler
    os.makedirs("LSTM", exist_ok=True)
    torch.save(model.state_dict(), "LSTM/lstm_model_1.pth")
    joblib.dump({"seq_len": config["seq_len"], "scaler_X": scaler_X, "scaler_y": scaler_y}, "LSTM/lstm_meta_1.joblib")
    print("模型和元数据已保存到 output/ 文件夹。")

    # 测试集预测
    X_new, y_new = load_data_from_excel(config["file"], config["test_sheet"])
    if len(X_new) < config["seq_len"]:
        raise ValueError(f"测试表格 {config['test_sheet']} 数据量 ({len(X_new)}) 小于 seq_len ({config['seq_len']})")
    X_new_scaled = scaler_X.transform(X_new)
    preds = predict_with_model(model, X_new_scaled, config["seq_len"], scaler_y, device=config["device"])
    y_true = y_new[config["seq_len"]:]

    # 绘制结果
    plot_results(X_new, y_true, preds, config["seq_len"], x_axis="shear_rate",
                 save_path="LSTM/lstm_pred_vs_true_shear_1.png")
    plot_results(X_new, y_true, preds, config["seq_len"], x_axis="time",
                 save_path="LSTM/lstm_pred_vs_true_time_1.png")

# 其余函数（SequenceDataset, LSTMModel, train_model, predict_with_model, plot_results）保持不变

if __name__ == "__main__":
    main()