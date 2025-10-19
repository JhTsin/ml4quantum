#!/usr/bin/env python3
"""
Lasso regression for 2D TFIM quantum system
Aligned with: 'Rethink the Role of Deep Learning towards Large-scale Quantum Systems' (ICML 2025)

Key settings (based on the paper):
- Random Fourier Features (RFF) used for feature mapping (Appendix B.2)
- RFF formula: φ(x) = [x, sqrt(2/d) * cos(Wx+b)]
- W ~ N(0, I) scaled by γ (Hamiltonian geometry informed)
- Fixed λ = 10^3 (no hyperparameter tuning)
- 2D TFIM lattice, typical Lx×Ly ∈ {5×5, 8×8}
- Feature dimension after RFF: d + R (original + R cosine features)
"""

import os
import argparse
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
import ast


# ======================================================
# 参数解析
# ======================================================
def parse_args():
    parser = argparse.ArgumentParser(description='Train Lasso model (ICML 2025 settings)')
    parser.add_argument('--train-file', type=str, required=True, help='Path to training CSV file')
    parser.add_argument('--test-file', type=str, required=True, help='Path to test CSV file')
    parser.add_argument('--task', type=str, default='correlation',
                        choices=['correlation', 'entropy'],
                        help='Prediction task: correlation or entropy')

    # === 修改1: 固定 λ=10^3（论文设定），无调参 ===
    parser.add_argument('--lasso-alpha', type=float, default=1e3,
                        help='Regularization λ for Lasso (fixed = 10³)')
    parser.add_argument('--lasso-tol', type=float, default=1e-3)
    parser.add_argument('--lasso-maxiter', type=int, default=10000)
    parser.add_argument('--model-seed', type=int, default=999)

    parser.add_argument('--save-results', action='store_true')
    parser.add_argument('--output-dir', type=str, default='./results')

    # === 修改3: 保留随机傅立叶特征 (RFF) 映射 ===
    parser.add_argument('--num-rff', type=int, default=20,
                        help='Number of random Fourier features R (≈ 10–40 in paper)')
    parser.add_argument('--rff-gamma', type=float, default=0.6,
                        help='Scaling γ for RFF (≈ 0.5–0.7 in paper)')
    return parser.parse_args()


# ======================================================
# 数据加载
# ======================================================
def load_data(file_path, task='correlation'):
    """
    Load dataset (TFIM 2D shadow measurements + labels)
    输入 x : shadow measurement flatten vectors
    标签 y : correlation 或 entropy
    """
    print(f"Loading data from {file_path} ...")
    df = pd.read_csv(file_path)

    # measurement_samples 列存储 [List[List[float]]]
    X_raw = df['measurement_samples'].apply(ast.literal_eval).tolist()
    X = np.array(X_raw)  # (N, M, Nq)
    N, num_meas, num_qubits = X.shape
    X_flat = X.reshape(N, -1) # (N, M*Nq)
    print(f" Loaded {N} samples, feature dim (before RFF): {X_flat.shape[1]}")

    # 标签列
    def parse_field(s):
        if isinstance(s, str) and s.startswith('Any['):
            s = s[4:-1]
        return ast.literal_eval(s) if isinstance(s, str) else s

    if task == 'correlation':
        y_approx = df['approx_correlation'].apply(parse_field).tolist()
        y_exact  = df['exact_correlation'].apply(parse_field).tolist()
    else:
        y_approx = df['approx_entropy'].apply(parse_field).tolist()
        y_exact  = df['exact_entropy'].apply(parse_field).tolist()

    return np.array(X_flat), np.array(y_approx), np.array(y_exact)


# ======================================================
# 随机傅立叶特征映射
# ======================================================
def generate_rff_params(D, R, gamma=0.6, seed=None):
    """
    Generate RFF random parameters W and b (to be reused for train/test)
    
    Args:
        D: Original feature dimension
        R: Number of RFF features
        gamma: Scaling parameter
        seed: Random seed
    
    Returns:
        W, b: Random projection matrix and phase shifts
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Generate random projection matrix W ~ N(0, I), then scale by γ
    W = np.random.randn(R, D) * gamma
    
    # Generate random phase shifts b ~ Uniform[0, 2π]
    b = np.random.uniform(0, 2*np.pi, R)
    
    return W, b


def apply_rff_transform(X, W, b):
    """
    Apply RFF transformation using given W and b matrices
    
    Formula (from paper): φ(x) = [x, sqrt(2/d) * cos(Wx + b)]
    where:
    - x is the original n×d feature matrix
    - W and b are pre-generated random parameters
    - d is the original feature dimension
    
    Returns: [x, sqrt(2/d) * cos(Wx + b)] with dimension D + R
    
    Note: This differs from standard RFF. The paper concatenates original 
    features with cosine features (no sine), which is problem-informed design
    for quantum systems according to Hamiltonian geometry.
    """
    N, D = X.shape
    
    # Compute projections: Wx + b
    proj = np.dot(X, W.T) + b  # Shape: (N, R)
    
    # Create RFF features: sqrt(2/d) * cos(Wx + b)
    # Scaling factor sqrt(2/d) as specified in paper
    scale = np.sqrt(2.0 / D)
    cos_features = scale * np.cos(proj)  # Shape: (N, R)
    
    # Concatenate original features with RFF: [x, sqrt(2/d) * cos(Wx + b)]
    X_rff = np.concatenate([X, cos_features], axis=1)  # Shape: (N, D + R)
    
    return X_rff


# ======================================================
# 训练与评估
# ======================================================
def train_and_evaluate(X_train, y_train_approx, X_test, y_test_exact, args):
    np.random.seed(args.model_seed)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # === 修改4: 添加 RFF 映射 (论文附录B.2说明) ===
    # CRITICAL: Generate W and b ONCE, then apply to both train and test
    # This ensures both datasets are in the SAME feature space
    D = X_train.shape[1]  # Original feature dimension
    W, b = generate_rff_params(D, R=args.num_rff, gamma=args.rff_gamma, 
                               seed=args.model_seed)
    
    # Apply RFF transformation using the SAME W and b for both sets
    X_train = apply_rff_transform(X_train, W, b)
    X_test  = apply_rff_transform(X_test, W, b)
    
    print(f" After RFF mapping: feature dim = {X_train.shape[1]} (d+R where d={D}, R={args.num_rff})")

    # 多输出处理
    if y_train_approx.ndim == 1:
        y_train_approx = y_train_approx.reshape(-1, 1)
        y_test_exact  = y_test_exact.reshape(-1, 1)
    outputs = y_train_approx.shape[1]

    # === 修改5: 固定 λ=1e3 ，无调参 ===
    model = Lasso(alpha=args.lasso_alpha,
                max_iter=args.lasso_maxiter,
                tol=args.lasso_tol,
                random_state=args.model_seed)

    # Train model directly (no cross-validation)
    model.fit(X_train, y_train_approx)
    
    # Evaluate on test set
    y_pred = model.predict(X_test)
    test_rmse = np.sqrt(np.mean((y_pred - y_test_exact)**2))

    print(f" λ = {args.lasso_alpha}, R = {args.num_rff}, γ = {args.rff_gamma}")
    print(f" Test RMSE = {test_rmse:.6f}")
    return test_rmse


# ======================================================
# 主函数
# ======================================================
def main():
    args = parse_args()
    print("="*60)
    print(" 2D TFIM Quantum System Prediction via Lasso Regression (ICML 2025)")
    print("="*60)
    print(f" Task: {args.task}")
    print(f" λ (Lasso regularization): {args.lasso_alpha}")
    print(f" RFF params: R = {args.num_rff}, γ = {args.rff_gamma}")
    print("="*60)

    # 加载数据
    X_train, y_train_approx, _ = load_data(args.train_file, task=args.task)
    X_test, _, y_test_exact = load_data(args.test_file, task=args.task)
    print(f" Train samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")

    # 训练与评估
    test_rmse = train_and_evaluate(X_train, y_train_approx,
                                   X_test, y_test_exact, args)

    # 结果输出
    print("="*60)
    print(f" Final Results — Task: {args.task}")
    print(f" Test RMSE: {test_rmse:.6f}")
    print("="*60)

    # 可选保存
    if args.save_results:
        os.makedirs(args.output_dir, exist_ok=True)
        
        # 提取训练和测试文件名信息
        train_basename = os.path.basename(args.train_file).replace('.csv', '')
        test_basename = os.path.basename(args.test_file).replace('.csv', '')
        
        # 创建更详细的文件名
        fname = os.path.join(args.output_dir,
                             f"lasso_tfim2d_{args.task}_{train_basename}_test_{test_basename}_R{args.num_rff}_gamma{args.rff_gamma}.txt")
        
        with open(fname, "w") as f:
            f.write(f"Lasso TFIM 2D Results (ICML 2025 Paper Settings)\n")
            f.write("="*60 + "\n")
            f.write(f"Task: {args.task}\n")
            f.write(f"Training file: {args.train_file}\n")
            f.write(f"Test file: {args.test_file}\n")
            f.write(f"Training samples: {X_train.shape[0]}\n")
            f.write(f"Test samples: {X_test.shape[0]}\n")
            f.write(f"Feature dimension (after RFF): {X_train.shape[1]}\n")
            f.write(f"Output dimension: {y_train_approx.shape[1] if len(y_train_approx.shape) > 1 else 1}\n")
            f.write("\n")
            f.write("Hyperparameters:\n")
            f.write(f"  λ (Lasso regularization): {args.lasso_alpha}\n")
            f.write(f"  RFF R: {args.num_rff}\n")
            f.write(f"  RFF γ: {args.rff_gamma}\n")
            f.write(f"  Random seed: {args.model_seed}\n")
            f.write("\n")
            f.write("Results:\n")
            f.write(f"  Test RMSE: {test_rmse:.6f}\n")
            f.write("="*60 + "\n")
        print(f" Results saved → {fname}")


if __name__ == '__main__':
    main()