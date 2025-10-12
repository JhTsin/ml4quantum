#!/usr/bin/env python3
"""
通用量子多体数据集分析脚本
========================

支持所有量子多体模型的数据集分析，包括：
- 2D横场Ising模型 (TFIM-2D)
- Heisenberg模型 (1D/2D)
- XXZ模型
- Cluster Ising模型
- 其他量子多体模型

使用方法：
    python analyze_tfim_2d.py <csv_file>

示例：
    python analyze_tfim_2d.py dataset_results/tfim_2d/n100|X(J,h,meas1000)_y(energy,entropy,corrs)_Nx3Ny3.csv
    python analyze_tfim_2d.py dataset_results/heisenberg_1d/n100|X(coupling,meas1000)_y(energy,entropy,corrs)_q8.csv
    python analyze_tfim_2d.py dataset_results/xxz_1d/n100|X(delta,meas1000)_y(energy,entropy,corrs)_q8.csv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast
import sys
from pathlib import Path

def parse_array_column(df, column_name):
    """解析存储为字符串的数组列"""
    def parse_value(val):
        if not isinstance(val, str):
            return val
        # 处理Julia格式的数组 (例如 "Any[1.0, 2.0, 3.0]")
        if val.startswith('Any[') and val.endswith(']'):
            val = val[3:]  # 移除 'Any' 前缀
        # 处理普通数组格式
        try:
            return ast.literal_eval(val)
        except (ValueError, SyntaxError) as e:
            print(f"Warning: Failed to parse value in column {column_name}: {val[:50]}...")
            return []
    
    return [parse_value(val) for val in df[column_name]]

def detect_model_type(df):
    """自动检测模型类型"""
    columns = df.columns.tolist()
    
    # 检测2D模型
    if 'Nx' in columns and 'Ny' in columns:
        Nx = df['Nx'].iloc[0]
        Ny = df['Ny'].iloc[0]
        return '2d', Nx, Ny
    
    # 检测1D模型
    elif 'qubits' in columns or 'N' in columns:
        if 'qubits' in columns:
            N = df['qubits'].iloc[0]
        else:
            N = df['N'].iloc[0]
        return '1d', N, 1
    
    # 默认情况
    else:
        return 'unknown', 1, 1

def plot_phase_diagram(df, save_path=None):
    """绘制能量-参数相图"""
    # 自动检测参数类型
    if 'transverse_field' in df.columns:
        x_param = df['transverse_field'].values
        x_label = 'Transverse Field h'
    elif 'coupling_strength' in df.columns:
        # 对于耦合强度，使用平均值
        coupling_data = parse_array_column(df, 'coupling_strength')
        x_param = [np.mean(c) for c in coupling_data]
        x_label = 'Average Coupling Strength'
    elif 'delta' in df.columns:
        # 对于XXZ模型
        delta_data = parse_array_column(df, 'delta')
        x_param = [np.mean(d) for d in delta_data]
        x_label = 'Average Delta'
    else:
        # 使用索引作为x轴
        x_param = np.arange(len(df))
        x_label = 'Sample Index'
    
    energies = df['ground_state_energy'].values
    
    model_type, Nx, Ny = detect_model_type(df)
    N = Nx * Ny
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 能量 vs 参数
    ax1.scatter(x_param, energies, alpha=0.6, s=30)
    ax1.set_xlabel(x_label, fontsize=12)
    ax1.set_ylabel('Ground State Energy', fontsize=12)
    ax1.set_title(f'Energy vs Parameter ({Nx}×{Ny} system)', fontsize=14)
    ax1.grid(True, alpha=0.3)
    
    # 单位格点能量 vs 参数
    energies_per_site = energies / N
    ax2.scatter(x_param, energies_per_site, alpha=0.6, s=30, color='orange')
    ax2.set_xlabel(x_label, fontsize=12)
    ax2.set_ylabel('Energy per Site', fontsize=12)
    ax2.set_title('Normalized Energy', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  ✓ Phase diagram saved to {save_path}")
    else:
        plt.show()
    
    return fig

def estimate_critical_field(df, n_bins=20):
    """估计相变点"""
    # 自动检测参数类型
    if 'transverse_field' in df.columns:
        x_param = df['transverse_field'].values
        param_name = 'transverse field'
    elif 'coupling_strength' in df.columns:
        coupling_data = parse_array_column(df, 'coupling_strength')
        x_param = [np.mean(c) for c in coupling_data]
        param_name = 'coupling strength'
    elif 'delta' in df.columns:
        delta_data = parse_array_column(df, 'delta')
        x_param = [np.mean(d) for d in delta_data]
        param_name = 'delta'
    else:
        return None
    
    energies = df['ground_state_energy'].values
    
    # 按参数分箱
    param_bins = np.linspace(x_param.min(), x_param.max(), n_bins)
    bin_centers = (param_bins[:-1] + param_bins[1:]) / 2
    
    # 计算每个箱的平均能量
    binned_energies = []
    for i in range(len(param_bins) - 1):
        mask = (x_param >= param_bins[i]) & (x_param < param_bins[i+1])
        if mask.any():
            binned_energies.append(energies[mask].mean())
        else:
            binned_energies.append(np.nan)
    
    binned_energies = np.array(binned_energies)
    
    # 计算数值导数（磁化率）
    valid_mask = ~np.isnan(binned_energies)
    if valid_mask.sum() < 3:
        print(f"  ⚠ Not enough data points to estimate critical {param_name}")
        return None
    
    dE_dparam = np.gradient(binned_energies[valid_mask], bin_centers[valid_mask])
    
    # 找到导数的极值
    critical_idx = np.argmax(np.abs(dE_dparam))
    param_critical = bin_centers[valid_mask][critical_idx]
    
    return param_critical, bin_centers[valid_mask], dE_dparam

def plot_susceptibility(df, save_path=None):
    """绘制磁化率（能量导数）"""
    result = estimate_critical_field(df)
    
    if result is None:
        return None
    
    param_critical, param_vals, susceptibility = result
    
    # 自动检测参数名称
    if 'transverse_field' in df.columns:
        param_name = 'Transverse Field h'
        title = 'Magnetic Susceptibility'
    elif 'coupling_strength' in df.columns:
        param_name = 'Coupling Strength J'
        title = 'Energy Susceptibility'
    elif 'delta' in df.columns:
        param_name = 'Delta'
        title = 'Energy Susceptibility'
    else:
        param_name = 'Parameter'
        title = 'Energy Susceptibility'
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(param_vals, np.abs(susceptibility), 'o-', linewidth=2, markersize=6)
    ax.axvline(param_critical, color='red', linestyle='--', linewidth=2, 
               label=f'Critical point: {param_name.split()[-1]} ≈ {param_critical:.2f}')
    ax.set_xlabel(param_name, fontsize=12)
    ax.set_ylabel('|dE/dparam| (Susceptibility)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  ✓ Susceptibility plot saved to {save_path}")
    else:
        plt.show()
    
    return fig, param_critical

def analyze_entanglement(df, save_path=None):
    """分析纠缠熵"""
    exact_entropies = parse_array_column(df, 'exact_entropy')
    
    # 自动检测参数类型
    if 'transverse_field' in df.columns:
        x_param = df['transverse_field'].values
        x_label = 'Transverse Field h'
    elif 'coupling_strength' in df.columns:
        coupling_data = parse_array_column(df, 'coupling_strength')
        x_param = [np.mean(c) for c in coupling_data]
        x_label = 'Average Coupling Strength'
    elif 'delta' in df.columns:
        delta_data = parse_array_column(df, 'delta')
        x_param = [np.mean(d) for d in delta_data]
        x_label = 'Average Delta'
    else:
        x_param = np.arange(len(df))
        x_label = 'Sample Index'
    
    # 计算平均熵
    avg_entropies = [np.mean(e) for e in exact_entropies]
    max_entropies = [np.max(e) for e in exact_entropies]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 平均熵 vs 参数
    ax1.scatter(x_param, avg_entropies, alpha=0.6, s=30, color='green')
    ax1.set_xlabel(x_label, fontsize=12)
    ax1.set_ylabel('Average Entanglement Entropy', fontsize=12)
    ax1.set_title('Mean Entanglement vs Parameter', fontsize=14)
    ax1.grid(True, alpha=0.3)
    
    # 最大熵 vs 参数
    ax2.scatter(x_param, max_entropies, alpha=0.6, s=30, color='purple')
    ax2.set_xlabel(x_label, fontsize=12)
    ax2.set_ylabel('Maximum Entanglement Entropy', fontsize=12)
    ax2.set_title('Max Entanglement vs Parameter', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  ✓ Entanglement plot saved to {save_path}")
    else:
        plt.show()
    
    return fig

def compare_exact_approx(df, save_path=None):
    """比较精确值和近似值的相关性"""
    exact_corrs = parse_array_column(df, 'exact_correlation')
    approx_corrs = parse_array_column(df, 'approx_correlation')
    
    exact_entropies = parse_array_column(df, 'exact_entropy')
    approx_entropies = parse_array_column(df, 'approx_entropy')
    
    # 展平为标量
    exact_corr_flat = np.concatenate([np.array(c).flatten() for c in exact_corrs])
    approx_corr_flat = np.concatenate([np.array(c).flatten() for c in approx_corrs])
    
    exact_entropy_flat = np.concatenate([np.array(e).flatten() for e in exact_entropies])
    approx_entropy_flat = np.concatenate([np.array(e).flatten() for e in approx_entropies])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 关联函数对比
    ax1.scatter(exact_corr_flat, approx_corr_flat, alpha=0.3, s=10)
    ax1.plot([-1, 1], [-1, 1], 'r--', linewidth=2, label='Perfect agreement')
    ax1.set_xlabel('Exact Correlation', fontsize=12)
    ax1.set_ylabel('Approximate Correlation', fontsize=12)
    ax1.set_title('Classical Shadow Correlation Accuracy', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    
    # 计算相关系数
    corr_r = np.corrcoef(exact_corr_flat, approx_corr_flat)[0, 1]
    ax1.text(0.05, 0.95, f'R = {corr_r:.4f}', 
             transform=ax1.transAxes, fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 熵对比
    ax2.scatter(exact_entropy_flat, approx_entropy_flat, alpha=0.3, s=10, color='green')
    max_entropy = max(exact_entropy_flat.max(), approx_entropy_flat.max())
    ax2.plot([0, max_entropy], [0, max_entropy], 'r--', linewidth=2, label='Perfect agreement')
    ax2.set_xlabel('Exact Entropy', fontsize=12)
    ax2.set_ylabel('Approximate Entropy', fontsize=12)
    ax2.set_title('Classical Shadow Entropy Accuracy', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')
    
    # 计算相关系数
    entropy_r = np.corrcoef(exact_entropy_flat, approx_entropy_flat)[0, 1]
    ax2.text(0.05, 0.95, f'R = {entropy_r:.4f}', 
             transform=ax2.transAxes, fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  ✓ Comparison plot saved to {save_path}")
    else:
        plt.show()
    
    return fig, corr_r, entropy_r

def print_statistics(df):
    """打印数据集统计信息"""
    print("\n" + "="*60)
    print("Dataset Statistics")
    print("="*60)
    
    # 自动检测系统信息
    model_type, Nx, Ny = detect_model_type(df)
    N = Nx * Ny
    
    print(f"\n📊 Basic Information:")
    if model_type == '2d':
        print(f"  Lattice size: {Nx} × {Ny}")
    elif model_type == '1d':
        print(f"  System size: {Nx}")
    else:
        print(f"  System size: {Nx}")
    print(f"  Total qubits: {N}")
    print(f"  Number of samples: {len(df)}")
    
    try:
        shots = len(parse_array_column(df, 'measurement_samples')[0])
        print(f"  Measurement shots: {shots}")
    except:
        print(f"  Measurement shots: Unknown")
    
    print(f"\n⚡ Energy Statistics:")
    energies = df['ground_state_energy'].values
    print(f"  Mean: {energies.mean():.4f}")
    print(f"  Std:  {energies.std():.4f}")
    print(f"  Min:  {energies.min():.4f}")
    print(f"  Max:  {energies.max():.4f}")
    
    # 自动检测参数统计
    if 'transverse_field' in df.columns:
        param_values = df['transverse_field'].values
        param_name = "Field"
    elif 'coupling_strength' in df.columns:
        coupling_data = parse_array_column(df, 'coupling_strength')
        param_values = [np.mean(c) for c in coupling_data]
        param_name = "Coupling"
    elif 'delta' in df.columns:
        delta_data = parse_array_column(df, 'delta')
        param_values = [np.mean(d) for d in delta_data]
        param_name = "Delta"
    else:
        param_values = np.arange(len(df))
        param_name = "Index"
    
    print(f"\n🧲 {param_name} Statistics:")
    print(f"  Mean: {np.mean(param_values):.4f}")
    print(f"  Std:  {np.std(param_values):.4f}")
    print(f"  Min:  {np.min(param_values):.4f}")
    print(f"  Max:  {np.max(param_values):.4f}")
    
    print(f"\n🔗 Entanglement Statistics:")
    exact_entropies = parse_array_column(df, 'exact_entropy')
    avg_entropies = [np.mean(e) for e in exact_entropies]
    print(f"  Mean entropy: {np.mean(avg_entropies):.4f}")
    print(f"  Std entropy:  {np.std(avg_entropies):.4f}")
    
    print("\n" + "="*60 + "\n")

def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_tfim_2d.py <csv_file>")
        print("\nExample:")
        print("  python analyze_tfim_2d.py dataset_results/tfim_2d/n100|X(J,h,meas1000)_y(energy,entropy,corrs)_Nx3Ny3.csv")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    
    if not Path(csv_file).exists():
        print(f"Error: File not found: {csv_file}")
        sys.exit(1)
    
    print("\n" + "="*60)
    print("2D TFIM Dataset Analysis")
    print("="*60)
    print(f"\nLoading: {csv_file}")
    
    # 读取数据
    df = pd.read_csv(csv_file)
    print(f"✓ Loaded {len(df)} samples")
    
    # 打印统计信息
    print_statistics(df)
    
    # 创建输出目录
    output_dir = Path("analysis_results")
    output_dir.mkdir(exist_ok=True)
    
    # 生成分析图表
    print("Generating analysis plots...")
    print("-"*60)
    
    base_name = Path(csv_file).stem
    
    # 1. 相图
    print("\n1. Phase diagram...")
    plot_phase_diagram(df, output_dir / f"{base_name}_phase_diagram.png")
    
    # 2. 磁化率
    print("\n2. Susceptibility...")
    result = plot_susceptibility(df, output_dir / f"{base_name}_susceptibility.png")
    if result:
        _, h_critical = result
        print(f"  → Estimated critical field: h_c ≈ {h_critical:.2f}")
    
    # 3. 纠缠熵
    print("\n3. Entanglement entropy...")
    analyze_entanglement(df, output_dir / f"{base_name}_entanglement.png")
    
    # 4. 精确vs近似
    print("\n4. Classical shadow accuracy...")
    _, corr_r, entropy_r = compare_exact_approx(df, output_dir / f"{base_name}_accuracy.png")
    print(f"  → Correlation accuracy: R = {corr_r:.4f}")
    print(f"  → Entropy accuracy: R = {entropy_r:.4f}")
    
    print("\n" + "-"*60)
    print(f"\n✓ Analysis complete! Plots saved to {output_dir}/")
    print("\n" + "="*60 + "\n")

if __name__ == "__main__":
    main()

