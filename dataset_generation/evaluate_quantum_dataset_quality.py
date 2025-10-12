#!/usr/bin/env python3
"""
通用量子多体数据集质量评估脚本
============================

支持评估所有量子多体模型的数据集质量：
- 2D横场Ising模型 (TFIM-2D)
- 1D Heisenberg模型
- 1D XXZ模型  
- 1D横场Ising模型 (TFIM-1D)
- 其他量子多体模型

使用方法：
    python evaluate_quantum_dataset_quality.py <csv_file>

示例：
    python evaluate_quantum_dataset_quality.py dataset_results/tfim_2d/n100|X(J,h,meas1000)_y(energy,entropy,corrs)_Nx3Ny3.csv
    python evaluate_quantum_dataset_quality.py dataset_results/heisenberg_1d/n100|X(coupling,meas128)_y(energy,entropy,corrs)_q127.csv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast
import sys
import re
from pathlib import Path

def parse_array_column(df, column_name):
    """解析存储为字符串的数组列"""
    def parse_value(val):
        if not isinstance(val, str):
            return val
        if val.startswith('Any[') and val.endswith(']'):
            val = val[3:]
        try:
            return ast.literal_eval(val)
        except (ValueError, SyntaxError):
            return []
    
    return [parse_value(val) for val in df[column_name]]

def detect_model_type(csv_file):
    """自动检测模型类型"""
    file_path = Path(csv_file)
    
    if 'tfim_2d' in str(file_path):
        return 'tfim_2d'
    elif 'heisenberg_1d' in str(file_path):
        return 'heisenberg_1d'
    elif 'xxz_1d' in str(file_path):
        return 'xxz_1d'
    elif 'ising_1d' in str(file_path):
        return 'ising_1d'
    elif 'cluster_ising' in str(file_path):
        return 'cluster_ising'
    else:
        df = pd.read_csv(csv_file, nrows=1)
        columns = df.columns.tolist()
        
        if 'Nx' in columns and 'Ny' in columns:
            return 'tfim_2d'
        elif 'coupling_matrix' in columns:
            return 'heisenberg_1d'
        elif 'delta' in columns:
            return 'xxz_1d'
        elif 'J' in columns and 'qubits_num' in columns:
            return 'ising_1d'
        else:
            return 'unknown'

def get_model_quality_criteria(model_type, df):
    """获取模型特定的质量评估标准"""
    criteria = {
        'energy_column': 'ground_state_energy',
        'entropy_columns': ['exact_entropy'],
        'correlation_columns': ['exact_correlation'],
        'parameter_columns': [],
        'system_size': 0,
        'energy_range': (-np.inf, 0),
        'entropy_range': (0, np.inf),
        'correlation_range': (-1, 1)
    }
    
    if model_type == 'tfim_2d':
        criteria.update({
            'energy_column': 'ground_state_energy',
            'entropy_columns': ['exact_entropy'],
            'correlation_columns': ['exact_correlation'],
            'parameter_columns': ['Nx', 'Ny', 'transverse_field', 'coupling_strength'],
            'system_size': df['Nx'].iloc[0] * df['Ny'].iloc[0] if 'Nx' in df.columns else 0,
            'energy_range': (-4 * criteria['system_size'], 0),
            'entropy_range': (0, np.log2(min(2**df['Nx'].iloc[0], 2**df['Ny'].iloc[0])) if 'Nx' in df.columns else (0, 10))
        })
        
    elif model_type == 'heisenberg_1d':
        criteria.update({
            'energy_column': 'ground_state_energy',
            'entropy_columns': ['exact_entropy'],
            'correlation_columns': ['exact_correlation'],
            'parameter_columns': ['coupling_matrix'],
            'system_size': len(parse_array_column(df, 'coupling_matrix')[0]) if 'coupling_matrix' in df.columns else 0,
            'energy_range': (-2 * criteria['system_size'], 0),
            'entropy_range': (0, np.log2(criteria['system_size']))
        })
        
    elif model_type == 'xxz_1d':
        criteria.update({
            'energy_column': 'energy',
            'entropy_columns': ['entropy'],
            'correlation_columns': ['corrzz'],
            'parameter_columns': ['nsites', 'delta'],
            'system_size': df['nsites'].iloc[0] if 'nsites' in df.columns else 0,
            'energy_range': (-2 * criteria['system_size'], 0),
            'entropy_range': (0, np.log2(criteria['system_size']))
        })
        
    elif model_type == 'ising_1d':
        criteria.update({
            'energy_column': 'energy',
            'entropy_columns': ['entropy'],
            'correlation_columns': ['corrzz'],
            'parameter_columns': ['qubits_num', 'J'],
            'system_size': df['qubits_num'].iloc[0] if 'qubits_num' in df.columns else 0,
            'energy_range': (-2 * criteria['system_size'], 0),
            'entropy_range': (0, np.log2(criteria['system_size']))
        })
    
    return criteria

def evaluate_physical_reasonableness(df, criteria):
    """评估物理合理性"""
    print("🔬 物理合理性检查")
    print("-" * 50)
    
    checks = []
    
    # 1. 能量检查
    energy_col = criteria['energy_column']
    if energy_col in df.columns:
        energies = df[energy_col].values
        E_min, E_max = criteria['energy_range']
        
        print(f"📊 能量统计:")
        print(f"  能量范围: [{energies.min():.3f}, {energies.max():.3f}]")
        print(f"  理论范围: [{E_min:.3f}, {E_max:.3f}]")
        
        if all(energies < 0):
            print("  ✅ 所有能量为负数（基态）")
            checks.append(True)
        else:
            print("  ❌ 存在正能量")
            checks.append(False)
        
        if E_min != -np.inf and E_max != 0:
            if all(E_min <= energies) and all(energies <= E_max):
                print("  ✅ 能量在合理范围内")
                checks.append(True)
            else:
                print("  ⚠️ 能量超出理论范围")
                checks.append(False)
        else:
            print("  ✅ 能量范围检查跳过（理论范围未定义）")
            checks.append(True)
    
    # 2. 参数检查
    print(f"\n🔧 参数检查:")
    for param in criteria['parameter_columns']:
        if param in df.columns:
            if param in ['coupling_matrix', 'coupling_strength']:
                try:
                    param_data = parse_array_column(df, param)
                    if param_data:
                        flat_data = np.concatenate([np.array(p).flatten() for p in param_data if len(p) > 0])
                        if len(flat_data) > 0:
                            print(f"  {param}: [{flat_data.min():.3f}, {flat_data.max():.3f}]")
                            if all(flat_data >= 0):
                                print(f"    ✅ 所有值为非负数")
                                checks.append(True)
                            else:
                                print(f"    ⚠️ 存在负值")
                                checks.append(False)
                except:
                    pass
            else:
                values = df[param].values
                if values.dtype in ['object', 'string']:
                    print(f"  {param}: {values.tolist()}")
                    checks.append(True)  # 字符串类型跳过检查
                else:
                    print(f"  {param}: [{values.min():.3f}, {values.max():.3f}]")
                    if all(values > 0):
                        print(f"    ✅ 所有值为正数")
                        checks.append(True)
                    else:
                        print(f"    ⚠️ 存在非正值")
                        checks.append(False)
    
    return {'physical_checks': checks}

def evaluate_data_completeness(df):
    """评估数据完整性"""
    print("\n📈 数据完整性检查")
    print("-" * 50)
    
    n_samples = len(df)
    print(f"样本数量: {n_samples}")
    
    # 样本数量评估
    if n_samples < 10:
        print("⚠️ 样本太少，难以进行统计分析")
        sample_quality = "poor"
    elif n_samples < 50:
        print("✅ 样本数量适中，适合初步分析")
        sample_quality = "moderate"
    elif n_samples < 200:
        print("✅ 样本数量充足，适合机器学习")
        sample_quality = "good"
    else:
        print("✅ 样本数量丰富，适合深度研究")
        sample_quality = "excellent"
    
    # 检查缺失值
    missing_data = df.isnull().sum()
    if missing_data.sum() == 0:
        print("✅ 无缺失数据")
        completeness = True
    else:
        print(f"❌ 存在缺失数据: {missing_data.sum()} 个")
        completeness = False
    
    return {
        'n_samples': n_samples,
        'sample_quality': sample_quality,
        'completeness': completeness
    }

def evaluate_entanglement_quality(df, criteria):
    """评估纠缠熵质量"""
    print("\n🔗 纠缠熵质量检查")
    print("-" * 50)
    
    entropy_cols = criteria['entropy_columns']
    valid_entropy_cols = [col for col in entropy_cols if col in df.columns]
    
    if not valid_entropy_cols:
        print("⚠️ 未找到纠缠熵列")
        return {'entropy_checks': [False], 'error': 'No entropy columns found'}
    
    checks = []
    
    for col in valid_entropy_cols:
        try:
            entropy_data = parse_array_column(df, col)
            if entropy_data:
                avg_entropies = [np.mean(e) for e in entropy_data if len(e) > 0]
                if avg_entropies:
                    print(f"{col} 统计:")
                    print(f"  平均熵范围: [{np.min(avg_entropies):.3f}, {np.max(avg_entropies):.3f}]")
                    print(f"  平均熵均值: {np.mean(avg_entropies):.3f}")
                    
                    # 熵合理性检查
                    if all(np.array(avg_entropies) >= 0):
                        print("  ✅ 所有熵值非负")
                        checks.append(True)
                    else:
                        print("  ❌ 存在负熵值")
                        checks.append(False)
                    
                    # 理论范围检查
                    S_min, S_max = criteria['entropy_range']
                    if S_max > 0:  # 只有当理论范围有效时才检查
                        if all(S_min <= avg_entropies) and all(avg_entropies <= S_max):
                            print(f"  ✅ 熵值在理论范围内 [{S_min:.2f}, {S_max:.2f}]")
                            checks.append(True)
                        else:
                            print(f"  ⚠️ 熵值超出理论范围")
                            checks.append(False)
                    else:
                        print(f"  ✅ 熵值范围检查跳过（理论范围未定义）")
                        checks.append(True)
        except Exception as e:
            print(f"⚠️ 纠缠熵检查失败: {e}")
            checks.append(False)
    
    return {'entropy_checks': checks}

def evaluate_correlation_quality(df, criteria):
    """评估关联函数质量"""
    print("\n🔗 关联函数质量检查")
    print("-" * 50)
    
    corr_cols = criteria['correlation_columns']
    valid_corr_cols = [col for col in corr_cols if col in df.columns]
    
    if not valid_corr_cols:
        print("⚠️ 未找到关联函数列")
        return {'correlation_checks': [False], 'error': 'No correlation columns found'}
    
    checks = []
    
    for col in valid_corr_cols:
        try:
            corr_data = parse_array_column(df, col)
            if corr_data:
                flat_corr = np.concatenate([np.array(c).flatten() for c in corr_data if len(c) > 0])
                if len(flat_corr) > 0:
                    print(f"{col} 统计:")
                    print(f"  关联值范围: [{flat_corr.min():.3f}, {flat_corr.max():.3f}]")
                    print(f"  关联值均值: {flat_corr.mean():.3f}")
                    
                    # 关联函数合理性检查
                    C_min, C_max = criteria['correlation_range']
                    if C_min != -1 or C_max != 1:  # 只有当范围不是默认的[-1,1]时才检查
                        if all(C_min <= flat_corr) and all(flat_corr <= C_max):
                            print(f"  ✅ 关联值在合理范围内 [{C_min}, {C_max}]")
                            checks.append(True)
                        else:
                            print(f"  ⚠️ 关联值超出合理范围")
                            checks.append(False)
                    else:
                        # 对于[-1,1]范围，检查是否在合理范围内
                        if all(-1.1 <= flat_corr) and all(flat_corr <= 1.1):  # 允许小的数值误差
                            print(f"  ✅ 关联值在合理范围内")
                            checks.append(True)
                        else:
                            print(f"  ⚠️ 关联值超出合理范围")
                            checks.append(False)
                    
                    # 对称性检查（如果是矩阵）
                    if len(corr_data[0]) > 1:
                        try:
                            # 检查第一个样本的对称性
                            first_corr = np.array(corr_data[0])
                            if first_corr.ndim == 2:
                                is_symmetric = np.allclose(first_corr, first_corr.T)
                                if is_symmetric:
                                    print("  ✅ 关联矩阵对称")
                                    checks.append(True)
                                else:
                                    print("  ⚠️ 关联矩阵不对称")
                                    checks.append(False)
                        except:
                            pass
        except Exception as e:
            print(f"⚠️ 关联函数检查失败: {e}")
            checks.append(False)
    
    return {'correlation_checks': checks}

def evaluate_shadow_accuracy(df):
    """评估Classical Shadow精度（如果存在）"""
    print("\n🎯 Classical Shadow精度评估")
    print("-" * 50)
    
    # 检查是否存在approx列
    approx_cols = [col for col in df.columns if 'approx' in col.lower()]
    
    if not approx_cols:
        print("⚠️ 未找到Classical Shadow数据")
        return {'accuracy_checks': [False], 'error': 'No shadow data found'}
    
    checks = []
    
    # 尝试计算精度
    try:
        if 'approx_correlation' in df.columns and 'exact_correlation' in df.columns:
            exact_corrs = parse_array_column(df, 'exact_correlation')
            approx_corrs = parse_array_column(df, 'approx_correlation')
            
            exact_corr_flat = np.concatenate([np.array(c).flatten() for c in exact_corrs])
            approx_corr_flat = np.concatenate([np.array(c).flatten() for c in approx_corrs])
            
            corr_r = np.corrcoef(exact_corr_flat, approx_corr_flat)[0, 1]
            print(f"关联函数精度: R = {corr_r:.4f}")
            
            if corr_r > 0.8:
                print("  ✅ 关联函数精度优秀")
                checks.append(True)
            elif corr_r > 0.6:
                print("  ✅ 关联函数精度良好")
                checks.append(True)
            else:
                print("  ⚠️ 关联函数精度需要改进")
                checks.append(False)
        
        if 'approx_entropy' in df.columns and 'exact_entropy' in df.columns:
            exact_entropies = parse_array_column(df, 'exact_entropy')
            approx_entropies = parse_array_column(df, 'approx_entropy')
            
            exact_entropy_flat = np.concatenate([np.array(e).flatten() for e in exact_entropies])
            approx_entropy_flat = np.concatenate([np.array(e).flatten() for e in approx_entropies])
            
            entropy_r = np.corrcoef(exact_entropy_flat, approx_entropy_flat)[0, 1]
            print(f"熵精度: R = {entropy_r:.4f}")
            
            if entropy_r > 0.7:
                print("  ✅ 熵精度优秀")
                checks.append(True)
            elif entropy_r > 0.5:
                print("  ✅ 熵精度良好")
                checks.append(True)
            else:
                print("  ⚠️ 熵精度需要改进")
                checks.append(False)
    
    except Exception as e:
        print(f"⚠️ 精度评估失败: {e}")
        checks.append(False)
    
    return {'accuracy_checks': checks}

def generate_quality_report(results, model_type):
    """生成质量报告"""
    print("\n" + "="*60)
    print("📋 数据集质量报告")
    print("="*60)
    
    # 计算总体质量分数
    total_checks = 0
    passed_checks = 0
    
    # 物理合理性
    if 'physical' in results and 'physical_checks' in results['physical']:
        total_checks += len(results['physical']['physical_checks'])
        passed_checks += sum(results['physical']['physical_checks'])
    
    # 数据完整性
    if results['completeness']['completeness']:
        passed_checks += 1
    total_checks += 1
    
    # 纠缠熵
    if 'entanglement' in results and 'entropy_checks' in results['entanglement']:
        total_checks += len(results['entanglement']['entropy_checks'])
        passed_checks += sum(results['entanglement']['entropy_checks'])
    
    # 关联函数
    if 'correlation' in results and 'correlation_checks' in results['correlation']:
        total_checks += len(results['correlation']['correlation_checks'])
        passed_checks += sum(results['correlation']['correlation_checks'])
    
    # 精度
    if 'shadow' in results and 'accuracy_checks' in results['shadow']:
        total_checks += len(results['shadow']['accuracy_checks'])
        passed_checks += sum(results['shadow']['accuracy_checks'])
    
    quality_score = passed_checks / total_checks if total_checks > 0 else 0
    
    print(f"\n总体质量分数: {quality_score:.1%} ({passed_checks}/{total_checks})")
    
    if quality_score >= 0.9:
        print("🏆 数据集质量优秀！")
    elif quality_score >= 0.7:
        print("✅ 数据集质量良好")
    elif quality_score >= 0.5:
        print("⚠️ 数据集质量一般，建议改进")
    else:
        print("❌ 数据集质量较差，需要重新生成")
    
    # 改进建议
    print(f"\n💡 改进建议:")
    if results['completeness']['n_samples'] < 50:
        print("  - 增加样本数量到50+")
    
    if 'shadow' in results and results['shadow'].get('accuracy_checks', [False]) and not all(results['shadow']['accuracy_checks']):
        print("  - 增加Classical Shadow测量次数")
    
    if 'physical' in results and not all(results['physical'].get('physical_checks', [True])):
        print("  - 检查物理参数设置")
    
    print(f"\n模型类型: {model_type}")
    print("\n" + "="*60)

def main():
    if len(sys.argv) < 2:
        print("Usage: python evaluate_quantum_dataset_quality.py <csv_file>")
        print("\nSupported models:")
        print("  - 2D横场Ising模型 (TFIM-2D)")
        print("  - 1D Heisenberg模型")
        print("  - 1D XXZ模型")
        print("  - 1D横场Ising模型 (TFIM-1D)")
        print("  - 其他量子多体模型")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    
    if not Path(csv_file).exists():
        print(f"Error: File not found: {csv_file}")
        sys.exit(1)
    
    print("="*60)
    print("通用量子多体数据集质量评估")
    print("="*60)
    print(f"\n分析文件: {csv_file}")
    
    # 读取数据
    df = pd.read_csv(csv_file)
    print(f"✓ 加载 {len(df)} 个样本")
    
    # 检测模型类型
    model_type = detect_model_type(csv_file)
    print(f"✓ 检测到模型类型: {model_type}")
    
    # 获取质量评估标准
    criteria = get_model_quality_criteria(model_type, df)
    print(f"✓ 系统大小: {criteria['system_size']}")
    
    # 执行各项评估
    results = {}
    
    results['physical'] = evaluate_physical_reasonableness(df, criteria)
    results['completeness'] = evaluate_data_completeness(df)
    results['entanglement'] = evaluate_entanglement_quality(df, criteria)
    results['correlation'] = evaluate_correlation_quality(df, criteria)
    results['shadow'] = evaluate_shadow_accuracy(df)
    
    # 生成质量报告
    generate_quality_report(results, model_type)

if __name__ == "__main__":
    main()
