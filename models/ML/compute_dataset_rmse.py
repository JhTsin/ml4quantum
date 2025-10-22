#!/usr/bin/env python3
"""
计算数据集中的RMSE统计
对于给定的TFIM数据集，计算以下6种RMSE：
1. exact_correlation vs 0 (baseline)
2. approx_correlation vs 0 (baseline)
3. approx_correlation vs exact_correlation (approximation error)
4. exact_entropy vs 0 (baseline)
5. approx_entropy vs 0 (baseline)
6. approx_entropy vs exact_entropy (approximation error)
"""

import argparse
import numpy as np
import pandas as pd
import ast
import os


def parse_field(s):
    """解析数据集中的字段（可能是字符串或已解析的值）"""
    if isinstance(s, str):
        # 处理 'Any[...]' 格式
        if s.startswith('Any['):
            s = s[4:-1]
        return ast.literal_eval(s)
    return s


def compute_rmse(y_pred, y_true):
    """计算RMSE"""
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    
    # 确保形状一致
    if y_pred.ndim == 1 and y_true.ndim == 1:
        mse = np.mean((y_pred - y_true) ** 2)
    else:
        # 对于多维输出，计算整体RMSE
        mse = np.mean((y_pred - y_true) ** 2)
    
    return np.sqrt(mse)


def compute_uniqueness_ratio(values, atol=1e-4):
    """计算数组中不重复数值占比：unique_count / total_count（按元素、全量展开）。
    
    使用绝对容差进行浮点比较以避免由于数值误差导致的误判。
    优化版本：使用numpy的unique函数提高效率。
    """
    arr = np.asarray(values).ravel()
    total = arr.size
    if total == 0:
        return 0.0
    
    # 使用numpy的unique函数，但需要处理容差
    # 先尝试直接使用unique，如果结果合理就使用
    unique_vals = np.unique(arr)
    
    # 如果unique值太多，说明可能有数值误差，需要容差处理
    if len(unique_vals) > total * 0.1:  # 如果unique值超过总数的10%，可能有问题
        # 使用更高效的方法：先排序，然后使用容差去重
        sorted_arr = np.sort(arr)
        unique_count = 1
        for i in range(1, len(sorted_arr)):
            if not np.isclose(sorted_arr[i], sorted_arr[i-1], atol=atol, rtol=0.0):
                unique_count += 1
    else:
        unique_count = len(unique_vals)
    
    return unique_count / total


def load_and_compute_rmse(file_path):
    """
    加载数据集并计算所有RMSE指标
    
    Returns:
        dict: 包含所有RMSE结果的字典
    """
    print(f"Loading dataset: {file_path}")
    
    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        return None
    
    # 读取CSV文件
    df = pd.read_csv(file_path)
    n_samples = len(df)
    print(f"Number of samples: {n_samples}")
    
    # 解析各个字段
    try:
        exact_corr = df['exact_correlation'].apply(parse_field).tolist()
        approx_corr = df['approx_correlation'].apply(parse_field).tolist()
        exact_entropy = df['exact_entropy'].apply(parse_field).tolist()
        approx_entropy = df['approx_entropy'].apply(parse_field).tolist()
    except Exception as e:
        print(f"Error parsing data: {e}")
        return None
    
    # 转换为numpy数组
    exact_corr = np.array(exact_corr)
    approx_corr = np.array(approx_corr)
    exact_entropy = np.array(exact_entropy)
    approx_entropy = np.array(approx_entropy)
    
    print(f"Data shapes:")
    print(f"  Correlation shape: {exact_corr.shape}")
    print(f"  Entropy shape: {exact_entropy.shape}")
    
    # 创建全零数组作为baseline
    zeros_corr = np.zeros_like(exact_corr)
    zeros_entropy = np.zeros_like(exact_entropy)
    
    # 计算6种RMSE
    results = {}
    
    # === Correlation相关的RMSE ===
    print("\n" + "="*60)
    print("CORRELATION RMSE Analysis")
    print("="*60)
    
    # 1. exact_correlation vs 0
    rmse_exact_corr_vs_zero = compute_rmse(exact_corr, zeros_corr)
    results['exact_correlation_vs_zero'] = rmse_exact_corr_vs_zero
    print(f"1. RMSE(exact_correlation vs 0):        {rmse_exact_corr_vs_zero:.6f}")
    
    # 2. approx_correlation vs 0
    rmse_approx_corr_vs_zero = compute_rmse(approx_corr, zeros_corr)
    results['approx_correlation_vs_zero'] = rmse_approx_corr_vs_zero
    print(f"2. RMSE(approx_correlation vs 0):       {rmse_approx_corr_vs_zero:.6f}")
    
    # 3. approx_correlation vs exact_correlation
    rmse_approx_vs_exact_corr = compute_rmse(approx_corr, exact_corr)
    results['approx_vs_exact_correlation'] = rmse_approx_vs_exact_corr
    print(f"3. RMSE(approx_correlation vs exact):   {rmse_approx_vs_exact_corr:.6f}")
    
    # 计算相对误差（相对于exact vs 0的baseline）
    relative_error_corr = (rmse_approx_vs_exact_corr / rmse_exact_corr_vs_zero) * 100
    print(f"   → Relative error: {relative_error_corr:.2f}%")
    
    # === Entropy相关的RMSE ===
    print("\n" + "="*60)
    print("ENTROPY RMSE Analysis")
    print("="*60)
    
    # 4. exact_entropy vs 0
    rmse_exact_entropy_vs_zero = compute_rmse(exact_entropy, zeros_entropy)
    results['exact_entropy_vs_zero'] = rmse_exact_entropy_vs_zero
    print(f"4. RMSE(exact_entropy vs 0):            {rmse_exact_entropy_vs_zero:.6f}")
    
    # 5. approx_entropy vs 0
    rmse_approx_entropy_vs_zero = compute_rmse(approx_entropy, zeros_entropy)
    results['approx_entropy_vs_zero'] = rmse_approx_entropy_vs_zero
    print(f"5. RMSE(approx_entropy vs 0):           {rmse_approx_entropy_vs_zero:.6f}")
    
    # 6. approx_entropy vs exact_entropy
    rmse_approx_vs_exact_entropy = compute_rmse(approx_entropy, exact_entropy)
    results['approx_vs_exact_entropy'] = rmse_approx_vs_exact_entropy
    print(f"6. RMSE(approx_entropy vs exact):       {rmse_approx_vs_exact_entropy:.6f}")
    
    # 计算相对误差（相对于exact vs 0的baseline）
    relative_error_entropy = (rmse_approx_vs_exact_entropy / rmse_exact_entropy_vs_zero) * 100
    print(f"   → Relative error: {relative_error_entropy:.2f}%")
    
    # 额外统计信息
    print("\n" + "="*60)
    print("ADDITIONAL STATISTICS")
    print("="*60)
    
    # 计算均值和方差
    exact_corr_mean = np.mean(exact_corr)
    exact_corr_var = np.var(exact_corr)
    approx_corr_mean = np.mean(approx_corr)
    approx_corr_var = np.var(approx_corr)
    exact_entropy_mean = np.mean(exact_entropy)
    exact_entropy_var = np.var(exact_entropy)
    approx_entropy_mean = np.mean(approx_entropy)
    approx_entropy_var = np.var(approx_entropy)
    
    # 保存到results
    results['exact_correlation_mean'] = exact_corr_mean
    results['exact_correlation_variance'] = exact_corr_var
    results['approx_correlation_mean'] = approx_corr_mean
    results['approx_correlation_variance'] = approx_corr_var
    results['exact_entropy_mean'] = exact_entropy_mean
    results['exact_entropy_variance'] = exact_entropy_var
    results['approx_entropy_mean'] = approx_entropy_mean
    results['approx_entropy_variance'] = approx_entropy_var
    
    print(f"Correlation - Mean(exact):     {exact_corr_mean:.6f}")
    print(f"Correlation - Variance(exact): {exact_corr_var:.6f}")
    print(f"Correlation - Mean(approx):    {approx_corr_mean:.6f}")
    print(f"Correlation - Variance(approx): {approx_corr_var:.6f}")
    print()
    print(f"Entropy - Mean(exact):         {exact_entropy_mean:.6f}")
    print(f"Entropy - Variance(exact):     {exact_entropy_var:.6f}")
    print(f"Entropy - Mean(approx):        {approx_entropy_mean:.6f}")
    print(f"Entropy - Variance(approx):    {approx_entropy_var:.6f}")
    
    # 计算数值多样性（不重复数值占比）
    print("\n" + "="*60)
    print("UNIQUENESS RATIO (数值多样性)")
    print("="*60)
    uniq_ratio_exact_corr = compute_uniqueness_ratio(exact_corr)
    uniq_ratio_approx_corr = compute_uniqueness_ratio(approx_corr)
    uniq_ratio_exact_entropy = compute_uniqueness_ratio(exact_entropy)
    uniq_ratio_approx_entropy = compute_uniqueness_ratio(approx_entropy)
    
    results['uniqueness_ratio_exact_correlation'] = uniq_ratio_exact_corr
    results['uniqueness_ratio_approx_correlation'] = uniq_ratio_approx_corr
    results['uniqueness_ratio_exact_entropy'] = uniq_ratio_exact_entropy
    results['uniqueness_ratio_approx_entropy'] = uniq_ratio_approx_entropy
    
    print(f"1. Uniqueness ratio (exact correlation):   {uniq_ratio_exact_corr:.6f}")
    print(f"2. Uniqueness ratio (approx correlation):  {uniq_ratio_approx_corr:.6f}")
    print(f"3. Uniqueness ratio (exact entropy):       {uniq_ratio_exact_entropy:.6f}")
    print(f"4. Uniqueness ratio (approx entropy):      {uniq_ratio_approx_entropy:.6f}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Compute RMSE for TFIM-like dataset. Only --file is supported.'
    )
    parser.add_argument('--file', type=str, required=True, help='Path to the dataset CSV file')
    args = parser.parse_args()

    file_path = args.file
    if not os.path.isfile(file_path):
        print(f'File not found: {file_path}')
        return

    df = pd.read_csv(file_path)

    # 检查所需列
    required_cols = [
        'exact_correlation', 'approx_correlation', 'exact_entropy', 'approx_entropy'
    ]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        print(f'Error: CSV文件缺少这些字段: {missing}')
        return

    try:
        exact_corr = df['exact_correlation'].apply(parse_field).tolist()
        approx_corr = df['approx_correlation'].apply(parse_field).tolist()
        exact_entropy = df['exact_entropy'].apply(parse_field).tolist()
        approx_entropy = df['approx_entropy'].apply(parse_field).tolist()
    except Exception as e:
        print(f'Error parsing columns: {e}')
        return

    try:
        exact_corr = np.array(exact_corr)
        approx_corr = np.array(approx_corr)
        exact_entropy = np.array(exact_entropy)
        approx_entropy = np.array(approx_entropy)
        zeros_corr = np.zeros_like(exact_corr)
        zeros_entropy = np.zeros_like(exact_entropy)
    except Exception as e:
        print(f'Error converting values to numpy arrays: {e}')
        return

    print("=================== RMSE RESULTS ===================")
    # 1. exact_correlation vs 0
    try:
        rmse1 = compute_rmse(exact_corr, zeros_corr)
        print(f"1. RMSE(exact_correlation vs 0):        {rmse1:.6f}")
    except Exception as e:
        print(f"1. exact_correlation vs 0: 计算失败 ({e})")
    # 2. approx_correlation vs 0
    try:
        rmse2 = compute_rmse(approx_corr, zeros_corr)
        print(f"2. RMSE(approx_correlation vs 0):       {rmse2:.6f}")
    except Exception as e:
        print(f"2. approx_correlation vs 0: 计算失败 ({e})")
    # 3. approx_correlation vs exact_correlation
    try:
        rmse3 = compute_rmse(approx_corr, exact_corr)
        print(f"3. RMSE(approx_correlation vs exact):   {rmse3:.6f}")
    except Exception as e:
        print(f"3. approx_correlation vs exact_correlation: 计算失败 ({e})")
    # 4. exact_entropy vs 0
    try:
        rmse4 = compute_rmse(exact_entropy, zeros_entropy)
        print(f"4. RMSE(exact_entropy vs 0):            {rmse4:.6f}")
    except Exception as e:
        print(f"4. exact_entropy vs 0: 计算失败 ({e})")
    # 5. approx_entropy vs 0
    try:
        rmse5 = compute_rmse(approx_entropy, zeros_entropy)
        print(f"5. RMSE(approx_entropy vs 0):           {rmse5:.6f}")
    except Exception as e:
        print(f"5. approx_entropy vs 0: 计算失败 ({e})")
    # 6. approx_entropy vs exact_entropy
    try:
        rmse6 = compute_rmse(approx_entropy, exact_entropy)
        print(f"6. RMSE(approx_entropy vs exact):       {rmse6:.6f}")
    except Exception as e:
        print(f"6. approx_entropy vs exact_entropy: 计算失败 ({e})")

    print("===================================================")
    
    # 计算均值和方差
    print("\n=================== MEAN AND VARIANCE ===================")
    try:
        exact_corr_mean = np.mean(exact_corr)
        exact_corr_var = np.var(exact_corr)
        approx_corr_mean = np.mean(approx_corr)
        approx_corr_var = np.var(approx_corr)
        exact_entropy_mean = np.mean(exact_entropy)
        exact_entropy_var = np.var(exact_entropy)
        approx_entropy_mean = np.mean(approx_entropy)
        approx_entropy_var = np.var(approx_entropy)
        
        print(f"Correlation - Mean(exact):     {exact_corr_mean:.6f}")
        print(f"Correlation - Variance(exact): {exact_corr_var:.6f}")
        print(f"Correlation - Mean(approx):    {approx_corr_mean:.6f}")
        print(f"Correlation - Variance(approx): {approx_corr_var:.6f}")
        print()
        print(f"Entropy - Mean(exact):         {exact_entropy_mean:.6f}")
        print(f"Entropy - Variance(exact):     {exact_entropy_var:.6f}")
        print(f"Entropy - Mean(approx):        {approx_entropy_mean:.6f}")
        print(f"Entropy - Variance(approx):    {approx_entropy_var:.6f}")
        print("=====================================================")
    except Exception as e:
        print(f"Mean and variance computation failed: {e}")
    
    # 计算数值多样性（不重复数值占比）
    print("\n=================== UNIQUENESS RATIO (数值多样性) ===================")
    try:
        uniq_ratio_exact_corr = compute_uniqueness_ratio(exact_corr)
        uniq_ratio_approx_corr = compute_uniqueness_ratio(approx_corr)
        uniq_ratio_exact_entropy = compute_uniqueness_ratio(exact_entropy)
        uniq_ratio_approx_entropy = compute_uniqueness_ratio(approx_entropy)
        
        print(f"1. Uniqueness ratio (exact correlation):   {uniq_ratio_exact_corr:.6f}")
        print(f"2. Uniqueness ratio (approx correlation):  {uniq_ratio_approx_corr:.6f}")
        print(f"3. Uniqueness ratio (exact entropy):       {uniq_ratio_exact_entropy:.6f}")
        print(f"4. Uniqueness ratio (approx entropy):      {uniq_ratio_approx_entropy:.6f}")
        print("================================================================")
    except Exception as e:
        print(f"Uniqueness ratio computation failed: {e}")

if __name__ == '__main__':
    main()

