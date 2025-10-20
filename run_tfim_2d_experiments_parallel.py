#!/usr/bin/env python3
"""
2D TFIM Dataset Generation Script (Parallel Version)
This script runs generation_tfim_2d_digital_pro.jl with various parameter combinations in parallel
"""

import subprocess
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import os
import logging
import psutil
from datetime import datetime
from typing import List, Tuple, Dict, Any
import argparse

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('multiprocessor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def get_max_workers() -> int:
    """
    Use maximum cores for fastest completion
    Aggressive allocation for maximum speed
    """
    # Get system information
    cpu_count_logical = psutil.cpu_count(logical=True)  # Logical cores
    cpu_count_physical = psutil.cpu_count(logical=False)  # Physical cores
    memory_gb = psutil.virtual_memory().total / (1024**3)
    
    # Ultra-aggressive strategy for maximum speed
    # Strategy 1: Use 98% of logical cores (leave only 2% for system)
    workers_by_cores_aggressive = int(cpu_count_logical * 0.98)
    
    # Strategy 2: Leave only 1 core for system (minimal safety)
    workers_by_cores_min = max(1, cpu_count_logical - 1)
    
    # Strategy 3: Use 95% of physical cores (ultra-aggressive)
    workers_by_physical = int(cpu_count_physical * 0.95)
    
    # Memory-based calculation (0.5GB per process for maximum speed)
    workers_by_memory = int(memory_gb / 0.5)
    
    # Take the most aggressive approach
    max_workers = min(workers_by_cores_aggressive, workers_by_cores_min, workers_by_physical, workers_by_memory)
    
    # Ultra-high caps for maximum speed
    if cpu_count_logical >= 128:
        max_workers = min(max_workers, 200)  # Ultra-high cap for very large systems
    elif cpu_count_logical >= 64:
        max_workers = min(max_workers, 120)  # Ultra-high cap for large systems
    elif cpu_count_logical >= 32:
        max_workers = min(max_workers, 60)   # Ultra-high cap for medium systems
    
    # Minimum safety bound
    max_workers = max(1, max_workers)
    
    logger.info(f"System resources (maximum speed mode):")
    logger.info(f"  Physical CPU cores: {cpu_count_physical}")
    logger.info(f"  Logical CPU cores: {cpu_count_logical}")
    logger.info(f"  Available memory: {memory_gb:.1f} GB")
    logger.info(f"  Workers by cores (98% ultra-aggressive): {workers_by_cores_aggressive}")
    logger.info(f"  Workers by cores (leave 1): {workers_by_cores_min}")
    logger.info(f"  Workers by physical (95%): {workers_by_physical}")
    logger.info(f"  Workers by memory (0.5GB each): {workers_by_memory}")
    logger.info(f"  Using maximum workers: {max_workers}")
    logger.info(f"  Cores left for system: {cpu_count_logical - max_workers}")
    
    return max_workers

def check_system_resources() -> bool:
    """
    Check if system resources are healthy before running experiments
    """
    try:
        # Relaxed checks for maximum speed
        cpu_percent = psutil.cpu_percent(interval=1)
        if cpu_percent > 95:
            logger.warning(f"Very high CPU usage detected: {cpu_percent}%")
        
        # Check memory usage
        memory = psutil.virtual_memory()
        if memory.percent > 95:
            logger.warning(f"Very high memory usage detected: {memory.percent}%")
        
        # Check disk space
        disk = psutil.disk_usage('/')
        if disk.percent > 95:
            logger.warning(f"Very high disk usage detected: {disk.percent}%")
        
        # System is healthy if no critical issues (relaxed thresholds)
        if cpu_percent < 95 and memory.percent < 95 and disk.percent < 95:
            logger.info(f"System health check passed - CPU: {cpu_percent}%, Memory: {memory.percent}%, Disk: {disk.percent}%")
            return True
        else:
            logger.warning("System health check failed - very high resource usage detected")
            return False
            
    except Exception as e:
        logger.error(f"System health check failed: {e}")
        return False

def run_single_experiment(params: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Run a single TFIM experiment
    
    Args:
        params: Dictionary containing experiment parameters
        
    Returns:
        Tuple of (success, message, params)
    """
    nx, ny = params['nx'], params['ny']
    samples = params['samples']
    shots = params['shots']
    
    cmd = [
        'julia', 
        '--threads=2',     # Limit threads per Julia process to avoid over-competition
        '--optimize=3',    # Maximum optimization level
        '--project=dataset_generation',  # Use project environment
        'dataset_generation/generation_tfim_2d_digital_pro.jl',
        '-n', str(samples),
        '-s', str(shots),
        '--Nx', str(nx),
        '--Ny', str(ny)
    ]
    
    start_time = time.time()
    experiment_id = f"Nx={nx}, Ny={ny}, samples={samples}, shots={shots}"
    
    try:
        logger.info(f"Starting experiment: {experiment_id}")
        
        # Run the Julia script (no timeout for maximum speed)
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=os.getcwd()
        )
        
        elapsed_time = time.time() - start_time
        
        if result.returncode == 0:
            logger.info(f"✅ Completed: {experiment_id} (took {elapsed_time:.1f}s)")
            return True, f"Success: {experiment_id}", params
        else:
            error_msg = f"❌ Failed: {experiment_id} - {result.stderr}"
            logger.error(error_msg)
            return False, error_msg, params
            
    except subprocess.TimeoutExpired:
        error_msg = f"⏰ Timeout: {experiment_id} (unexpected timeout)"
        logger.error(error_msg)
        return False, error_msg, params
    except Exception as e:
        error_msg = f"💥 Exception: {experiment_id} - {str(e)}"
        logger.error(error_msg)
        return False, error_msg, params

def generate_experiment_parameters() -> List[Dict[str, Any]]:
    """Generate all experiment parameter combinations"""
    experiments = []
    
    # Grid sizes to test
    grid_sizes = [(5, 5)]  # Add (8, 8) if needed
    
    # Part 1: Run with -n 100 for different shot numbers
    shots_n100 = [64,128,256,512]

    for nx, ny in grid_sizes:
        for samples in [100, 200]:
            for shots in shots_n100:
                experiments.append({
                    'nx': nx,
                    'ny': ny,
                    'samples': samples,
                    'shots': shots,
                    'part': 'Part1'
                })
    
    #Part 2: Run with -s 512 for different sample numbers (uncomment if needed)
    samples_s512 = [20, 40, 60, 80, 100]
    for nx, ny in grid_sizes:
        for samples in samples_s512:
            experiments.append({
                'nx': nx,
                'ny': ny,
                'samples': samples,
                'shots': 512,
                'part': 'Part2'
            })
    
    return experiments

def run_experiments_parallel(max_workers: int = 4):
    """
    Run all experiments in parallel
    
    Args:
        max_workers: Maximum number of parallel processes
    """
    experiments = generate_experiment_parameters()
    
    logger.info(f"Starting {len(experiments)} experiments with max {max_workers} parallel workers")
    logger.info(f"Start time: {datetime.now()}")
    
    successful = 0
    failed = 0
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all experiments
        future_to_params = {
            executor.submit(run_single_experiment, exp): exp 
            for exp in experiments
        }
        
        # Process completed experiments
        for future in as_completed(future_to_params):
            params = future_to_params[future]
            try:
                success, message, _ = future.result()
                if success:
                    successful += 1
                else:
                    failed += 1
                    logger.error(f"Failed experiment: {params}")
            except Exception as e:
                failed += 1
                logger.error(f"Exception in experiment {params}: {e}")
    
    logger.info(f"All experiments completed!")
    logger.info(f"Successful: {successful}, Failed: {failed}")
    logger.info(f"End time: {datetime.now()}")


def main():
    parser = argparse.ArgumentParser(description='Run 2D TFIM experiments with optimal parallelism')
    parser.add_argument('--dry-run', action='store_true',
                       help='Print experiment parameters without running them')
    parser.add_argument('--skip-health-check', action='store_true',
                       help='Skip system health check before running')
    
    args = parser.parse_args()
    
    if args.dry_run:
        experiments = generate_experiment_parameters()
        print(f"Would run {len(experiments)} experiments:")
        for i, exp in enumerate(experiments, 1):
            print(f"  {i}. {exp}")
        return
    
    # Check system health before starting
    if not args.skip_health_check:
        logger.info("Performing system health check...")
        if not check_system_resources():
            logger.warning("System health check failed. Use --skip-health-check to override.")
            logger.warning("Continuing anyway, but monitor system resources carefully.")
    
    # Use maximum cores for fastest completion
    max_workers = get_max_workers()
    
    experiments = generate_experiment_parameters()
    logger.info(f"Running {len(experiments)} experiments with maximum {max_workers} workers for fastest completion")
    run_experiments_parallel(max_workers=max_workers)

if __name__ == "__main__":
    main()
