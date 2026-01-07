#!/usr/bin/env python
"""
Quick experiment to compare success rates with different n_action_steps.
Tests shorter periods (more reactive) vs longer periods (more efficient).
"""

import subprocess
import sys
import json
from pathlib import Path


def run_quick_eval(n_action_steps: int, n_episodes: int = 5, task: str = "libero_spatial") -> dict:
    """Run a quick evaluation with specified n_action_steps."""
    
    output_dir = f"./eval_logs/logs/{task}_quick_eval_n{n_action_steps}"
    
    cmd = [
        "lerobot-eval",
        "--policy.path=lerobot/pi05_libero_finetuned",
        f"--policy.n_action_steps={n_action_steps}",
        "--env.type=libero",
        f"--env.task={task}",
        f"--eval.n_episodes={n_episodes}",
        "--eval.batch_size=1",
        f"--output_dir={output_dir}",
        "--env.max_parallel_tasks=1",
    ]
    
    print(f"\n{'='*80}")
    print(f"Testing n_action_steps={n_action_steps} (period={n_action_steps*50}ms)")
    print(f"{'='*80}")
    print(f"Command: {' '.join(cmd)}")
    print()
    
    try:
        # Run without capturing output so progress bars show in real-time
        # Timeout set to 20 minutes (1200s) for 10 tasks × 5 episodes
        result = subprocess.run(cmd, timeout=1200)
        
        if result.returncode == 0:
            print(f"\n✓ Completed n_action_steps={n_action_steps}")
            return {
                "n_action_steps": n_action_steps,
                "period_ms": n_action_steps * 50,
                "success": True,
                "output_dir": output_dir,
            }
        else:
            print(f"\n✗ Failed with exit code {result.returncode}")
            return {
                "n_action_steps": n_action_steps,
                "period_ms": n_action_steps * 50,
                "success": False,
                "error": f"exit_code_{result.returncode}",
            }
    
    except subprocess.TimeoutExpired:
        print(f"\n⚠️  Timeout after 10 minutes")
        return {
            "n_action_steps": n_action_steps,
            "period_ms": n_action_steps * 50,
            "success": False,
            "error": "timeout",
        }
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return {
            "n_action_steps": n_action_steps,
            "period_ms": n_action_steps * 50,
            "success": False,
            "error": str(e),
        }


def main():
    """Run comparison experiment."""
    print("\n" + "="*80)
    print("ACTION CHUNK PERIOD COMPARISON EXPERIMENT")
    print("="*80)
    print("\nHypothesis: Shorter periods (lower n_action_steps) → Better success rate")
    print("Reason: Faster reaction to environmental changes\n")
    
    # Test configurations from very reactive to efficient
    configs = [
        (5, "Very Reactive - 250ms period, 4Hz re-planning"),
        (10, "Default - 500ms period, 2Hz re-planning"),
        (15, "Balanced - 750ms period, 1.33Hz re-planning"),
        (20, "Efficient - 1000ms period, 1Hz re-planning"),
        (40, "Efficient - 1000ms period, 1Hz re-planning"),
    ]
    tasks = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]  # Tasks to evaluate

    n_episodes = 10  # Quick test with 5 episodes each
    
    print(f"Running {n_episodes} episodes per configuration...")
    print(f"Total configurations: {len(configs)}")
    print(f"Estimated time: {len(configs) * n_episodes * 2} minutes\n")
    
    print("Starting experiment in 3 seconds...")
    import time
    time.sleep(3)
    
    results = []
    
    for n_steps, description in configs:
        for task in tasks:
            print(f"\n📊 Configuration {len(results)+1}/{len(configs)}: {description} - Task: {task}")
            result = run_quick_eval(n_steps, n_episodes, task)
            results.append(result)
    
    # Print summary
    print("\n" + "="*80)
    print("EXPERIMENT RESULTS SUMMARY")
    print("="*80)
    print()
    
    print(f"{'n_action_steps':<15} {'Period':<12} {'Frequency':<12} {'Status':<10}")
    print("-"*80)
    
    for result in results:
        n_steps = result['n_action_steps']
        period = result['period_ms']
        freq = 1000 / period
        status = "✓ Completed" if result['success'] else "✗ Failed"
        
        print(f"{n_steps:<15} {period:<12} {freq:<12.2f} {status:<10}")
    
    print()
    print("="*80)
    print("DETAILED RESULTS")
    print("="*80)
    
    for result in results:
        print(f"\nn_action_steps={result['n_action_steps']} ({result['period_ms']}ms period):")
        if result['success']:
            print("  Check output in: ./quick_eval_n{}/".format(result['n_action_steps']))
            print("  Run: cat ./quick_eval_n{}/metrics.json".format(result['n_action_steps']))
        else:
            print(f"  Error: {result.get('error', 'Unknown')}")
    
    print("\n" + "="*80)
    print("ANALYSIS")
    print("="*80)
    print("""
To analyze the results, check the metrics files:
    
    for dir in quick_eval_n*/; do
        echo "=== $dir ==="
        if [ -f "$dir/metrics.json" ]; then
            cat "$dir/metrics.json"
        fi
    done

Expected pattern:
- Lower n_action_steps (5-10) → Higher success rate (more reactive)
- Higher n_action_steps (15-20) → Lower success rate (less reactive)

Trade-offs:
- Short periods: Better adaptation, more computation
- Long periods: Less computation, slower adaptation
""")
    
    # Save results
    summary_file = Path("./experiment_summary.json")
    with open(summary_file, "w") as f:
        json.dump({
            "experiment": "action_chunk_period_comparison",
            "n_episodes_per_config": n_episodes,
            "results": results,
        }, f, indent=2)
    
    print(f"\nResults saved to: {summary_file}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Experiment interrupted by user")
        sys.exit(1)
