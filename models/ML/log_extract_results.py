#!/usr/bin/env python3
"""
Extract Test RMSE results from all log files and generate a summary report.
This script searches for .log files in the current directory and extracts
the Test RMSE values, then generates a task_exp1.txt summary file.
"""

import os
import re
from pathlib import Path
from collections import defaultdict


def extract_tasks_from_log(log_file_path):
    """
    Extract Test RMSE results for all tasks present in a log file. If a task is
    detected as running but has no RMSE yet, mark it as '运行中'.
    
    Args:
        log_file_path: Path to the log file
        
    Returns:
        dict: Mapping of task -> value where value is either RMSE string or '运行中'
    """
    try:
        with open(log_file_path, 'r') as f:
            content = f.read()

        task_to_value = {}

        # Strategy 1: Prefer parsing explicit Final Results blocks (most precise)
        final_results_pattern = re.compile(
            r"Final Results\s*—\s*Task:\s*(correlation|entropy)[\s\S]*?Test RMSE\s*[:=]\s*([0-9]+\.[0-9]+)",
            re.IGNORECASE
        )
        for m in final_results_pattern.finditer(content):
            task = m.group(1).lower()
            rmse_value = m.group(2)
            task_to_value[task] = rmse_value

        # Strategy 2: Fallback — for each Task: <name> block, find following Test RMSE
        if not task_to_value:
            task_iter = list(re.finditer(r"Task:\s*(correlation|entropy)", content, re.IGNORECASE))
            for idx, tm in enumerate(task_iter):
                task = tm.group(1).lower()
                start = tm.end()
                end = task_iter[idx + 1].start() if idx + 1 < len(task_iter) else len(content)
                block = content[start:end]
                rm = re.search(r"Test RMSE\s*[:=]\s*([0-9]+\.[0-9]+)", block)
                if rm:
                    task_to_value[task] = rm.group(1)

        # Strategy 3: Detect running tasks (only if not already recorded with RMSE)
        running_matches = re.findall(r"Running\s+(correlation|entropy)\b", content, flags=re.IGNORECASE)
        for task in running_matches:
            task_lower = task.lower()
            if task_lower not in task_to_value:
                task_to_value[task_lower] = '运行中'

        return task_to_value
    except Exception as e:
        print(f"Warning: Error reading {log_file_path}: {e}")
        return {}


def parse_log_filename(filename):
    """
    Parse the log filename to extract experiment parameters.
    
    Expected format: tfim_example55_512_40_correlation.log
    or: tfim_example55_512_40.log (for older format)
    
    Args:
        filename: The log filename
        
    Returns:
        dict: Parsed parameters or None if format doesn't match
    """
    # Remove .log extension
    base_name = filename.replace('.log', '')
    
    # Try to match the pattern: tfim_example{grid}{grid}_{meas}_{train}_{task}
    # or: tfim_example{grid}{grid}_{meas}_{train}
    pattern1 = r'tfim_example(\d)(\d)_(\d+)_(\d+)_(correlation|entropy)'
    pattern2 = r'tfim_example(\d)(\d)_(\d+)_(\d+)'
    
    match = re.match(pattern1, base_name)
    if match:
        grid1, grid2, meas, train, task = match.groups()
        return {
            'grid': f"{grid1}x{grid2}",
            'meas': meas,
            'train': train,
            'task': task,
            'exp_name': f"tfim_example{grid1}{grid2}_{meas}_{train}"
        }
    
    match = re.match(pattern2, base_name)
    if match:
        grid1, grid2, meas, train = match.groups()
        return {
            'grid': f"{grid1}x{grid2}",
            'meas': meas,
            'train': train,
            'task': None,  # Will be determined from log content
            'exp_name': f"tfim_example{grid1}{grid2}_{meas}_{train}"
        }
    
    return None


def collect_results(log_directory='.'):
    """
    Collect all results from log files in the specified directory.
    
    Args:
        log_directory: Directory containing log files
        
    Returns:
        dict: Nested dictionary organized by experiment name and task type
    """
    results = defaultdict(dict)
    
    # Find all .log files
    log_files = list(Path(log_directory).glob('*.log'))
    
    print(f"Found {len(log_files)} log files in {log_directory}")
    
    for log_file in sorted(log_files):
        filename = log_file.name
        parsed = parse_log_filename(filename)
        
        if parsed is None:
            print(f"Skipping {filename}: doesn't match expected format")
            continue
        
        # Extract all task results from the log
        task_values = extract_tasks_from_log(log_file)

        if not task_values:
            print(f"Skipping {filename}: couldn't extract any task status or RMSE")
            continue

        # Store results per task (correlation/entropy)
        exp_name = parsed['exp_name']
        for task_type, value in task_values.items():
            results[exp_name][task_type] = value
            if value == '运行中':
                print(f"  {exp_name} - {task_type}: 运行中")
            else:
                print(f"  {exp_name} - {task_type}: Test RMSE = {value}")
    
    return results


def generate_summary_file(results, output_file='task_exp1.txt'):
    """
    Generate a formatted summary file from the collected results.
    
    Args:
        results: Dictionary of results from collect_results()
        output_file: Path to output summary file
    """
    with open(output_file, 'w') as f:
        for exp_name in sorted(results.keys()):
            f.write(f"{exp_name}\n")
            
            # Write correlation results/status if available
            if 'correlation' in results[exp_name]:
                value = results[exp_name]['correlation']
                if value == '运行中':
                    f.write(f"   - correlation : 运行中\n")
                else:
                    f.write(f"   - correlation : Test RMSE = {value}\n")
            
            # Write entropy results/status if available
            if 'entropy' in results[exp_name]:
                value = results[exp_name]['entropy']
                if value == '运行中':
                    f.write(f"   - entropy : 运行中\n")
                else:
                    f.write(f"   - entropy : Test RMSE = {value}\n")
    
    print(f"\nSummary written to {output_file}")


def main():
    """Main function to run the extraction and summary generation."""
    print("=" * 60)
    print("Extracting Test RMSE results from log files")
    print("=" * 60)
    print()
    
    # Get the current directory
    current_dir = os.getcwd()
    print(f"Working directory: {current_dir}\n")
    
    # Collect results from current directory
    results = collect_results('.')
    
    if not results:
        print("\nNo valid results found!")
        return
    
    print("\n" + "=" * 60)
    print(f"Collected results from {len(results)} experiments")
    print("=" * 60)
    
    # Generate summary file
    generate_summary_file(results, 'task_exp1.txt')
    
    print("\nDone!")


if __name__ == '__main__':
    main()

