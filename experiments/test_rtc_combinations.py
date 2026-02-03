#!/usr/bin/env python

"""Test multiple combinations of execution horizon and inference delay for PI05 on LIBERO.

Usage example:
python test_rtc_combinations.py \
    --policy.path=lerobot/pi05_libero_finetuned \
    --env.task=libero_spatial \
    --output_dir=./rtc_sweep_results/ \
    --execution_horizons 10,20,40 \
    --inference_delays 0,2,4,6,8
"""

import argparse
import csv
import logging
import sys
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from pprint import pformat

import torch
from termcolor import colored

from lerobot.configs import parser
from lerobot.configs.eval import EvalPipelineConfig
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.envs.utils import close_envs
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import get_safe_torch_device, init_logging

from lerobot.scripts.lerobot_eval import eval_policy_all


def parse_rtc_args():
    """Parse RTC-specific arguments that aren't part of EvalPipelineConfig."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--execution_horizons",
        type=str,
        default="10,20,40",
        help="Comma-separated list of execution horizons to test (e.g., '10,20,40')",
    )
    parser.add_argument(
        "--inference_delays",
        type=str,
        default="0,2,4,6,8",
        help="Comma-separated list of inference delays to test (e.g., '0,2,4,6,8')",
    )
    args, remaining = parser.parse_known_args()
    
    # Parse comma-separated strings into integer lists
    execution_horizons = [int(x.strip()) for x in args.execution_horizons.split(",")]
    inference_delays = [int(x.strip()) for x in args.inference_delays.split(",")]
    
    # Update sys.argv to remove our custom args so parser.wrap() doesn't see them
    sys.argv = [sys.argv[0]] + remaining
    
    return execution_horizons, inference_delays


@parser.wrap()
def main(cfg: EvalPipelineConfig, execution_horizons=None, inference_delays=None):
    logging.info(pformat(asdict(cfg)))

    # Check device is available
    device = get_safe_torch_device(cfg.policy.device, log=True)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    set_seed(cfg.seed)

    logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")

    # Log the RTC parameter ranges being tested
    logging.info(
        colored("RTC parameter sweep:", "yellow", attrs=["bold"])
        + f" execution_horizons={execution_horizons}, inference_delays={inference_delays}"
    )

    rtc_settings = [True]  # RTC on/off

    action_chunk_size_candidates = [
        getattr(cfg.policy, "chunk_size", None),
        getattr(cfg.policy, "n_action_steps", None),
        getattr(cfg.policy, "action_chunk_size", None),
    ]
    action_chunk_size = next((value for value in action_chunk_size_candidates if value is not None), None)

    total_requested = len(execution_horizons) * len(inference_delays) * len(rtc_settings)

    if action_chunk_size is not None:
        def _is_valid_horizon(eh: int, delay: int) -> bool:
            min_horizon = max(1, delay)
            max_horizon = action_chunk_size - delay
            return max_horizon >= min_horizon and min_horizon <= eh <= max_horizon

        combinations = [
            (rtc_enabled, execution_horizon, inference_delay)
            for rtc_enabled in rtc_settings
            for execution_horizon in execution_horizons
            for inference_delay in inference_delays
            if _is_valid_horizon(execution_horizon, inference_delay)
        ]

        skipped = total_requested - len(combinations)
        if skipped > 0:
            logging.info(
                "Skipping %d invalid combinations due to execution_horizon constraints "
                "(action_chunk_size=%d).",
                skipped,
                action_chunk_size,
            )
    else:
        logging.warning(
            "Could not determine action_chunk_size; execution_horizon constraints will not be enforced."
        )
        combinations = [
            (rtc_enabled, execution_horizon, inference_delay)
            for rtc_enabled in rtc_settings
            for execution_horizon in execution_horizons
            for inference_delay in inference_delays
        ]

    if not combinations:
        logging.warning("No valid RTC combinations after applying execution_horizon constraints. Exiting.")
        return

    # Create output directory
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "rtc_combinations_results.csv"

    # Prepare CSV file (append if it already exists)
    fieldnames = [
        "rtc_enabled",
        "execution_horizon",
        "inference_delay",
        "avg_sum_reward",
        "avg_max_reward",
        "pc_success",
        "eval_s",
        "eval_ep_s",
    ]
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with open(csv_path, "a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        logging.info("Making environment.")
        envs = make_env(
            cfg.env,
            n_envs=cfg.eval.batch_size,
            use_async_envs=cfg.eval.use_async_envs,
            trust_remote_code=cfg.trust_remote_code,
        )

        total_combinations = len(combinations)

        for current_combination, (rtc_enabled, execution_horizon, inference_delay) in enumerate(
            combinations, start=1
        ):
            logging.info(
                colored(
                    f"\n{'=' * 80}\n"
                    f"Testing combination {current_combination}/{total_combinations}:\n"
                    f"  RTC Enabled: {rtc_enabled}\n"
                    f"  Execution Horizon: {execution_horizon}\n"
                    f"  Inference Delay: {inference_delay}\n"
                    f"{'=' * 80}",
                    "cyan",
                    attrs=["bold"],
                )
            )

            # Create fresh policy for each combination
            logging.info("Making policy.")
            policy = make_policy(
                cfg=cfg.policy,
                env_cfg=cfg.env,
                rename_map=cfg.rename_map,
            )

            # Update RTC configuration
            if hasattr(policy.config, "rtc_config"):
                if rtc_enabled:
                    if policy.config.rtc_config is None:
                        policy.config.rtc_config = RTCConfig()
                    policy.config.rtc_config.enabled = True
                    policy.config.rtc_config.execution_horizon = execution_horizon
                    policy.config.rtc_config.inference_delay = inference_delay
                    logging.info(
                        f"Updated RTCConfig: enabled={policy.config.rtc_config.enabled}, "
                        f"execution_horizon={policy.config.rtc_config.execution_horizon}, "
                        f"inference_delay={policy.config.rtc_config.inference_delay}"
                    )
                else:
                    if policy.config.rtc_config is None:
                        policy.config.rtc_config = RTCConfig()
                    policy.config.rtc_config.enabled = False
                    logging.info("RTC disabled for this run")
            else:
                logging.warning("Policy does not have rtc_config attribute")

            # Reinitialize RTC processor with new config
            if hasattr(policy, "init_rtc_processor"):
                policy.init_rtc_processor()

            policy.eval()

            # Create preprocessors
            preprocessor_overrides = {
                "device_processor": {"device": str(policy.config.device)},
                "rename_observations_processor": {"rename_map": cfg.rename_map},
            }

            preprocessor, postprocessor = make_pre_post_processors(
                policy_cfg=cfg.policy,
                pretrained_path=cfg.policy.pretrained_path,
                preprocessor_overrides=preprocessor_overrides,
            )

            env_preprocessor, env_postprocessor = make_env_pre_post_processors(
                env_cfg=cfg.env, policy_cfg=cfg.policy
            )

            # Determine whether to use predict_action_chunk
            use_predict_action_chunk = rtc_enabled

            # Run evaluation
            with (
                torch.no_grad(),
                torch.autocast(device_type=device.type) if cfg.policy.use_amp else nullcontext(),
            ):
                info = eval_policy_all(
                    envs=envs,
                    policy=policy,
                    env_preprocessor=env_preprocessor,
                    env_postprocessor=env_postprocessor,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    n_episodes=cfg.eval.n_episodes,
                    max_episodes_rendered=0,  # Disable video rendering for speed
                    videos_dir=None,
                    start_seed=cfg.seed,
                    max_parallel_tasks=cfg.env.max_parallel_tasks,
                    use_predict_action_chunk=use_predict_action_chunk,
                )

            # Extract overall metrics
            overall_metrics = info["overall"]
            result = {
                "rtc_enabled": rtc_enabled,
                "execution_horizon": execution_horizon,
                "inference_delay": inference_delay,
                "avg_sum_reward": overall_metrics["avg_sum_reward"],
                "avg_max_reward": overall_metrics["avg_max_reward"],
                "pc_success": overall_metrics["pc_success"],
                "eval_s": overall_metrics["eval_s"],
                "eval_ep_s": overall_metrics["eval_ep_s"],
            }

            # Write to CSV
            writer.writerow(result)
            csvfile.flush()  # Ensure results are written immediately

            # Log results
            logging.info(
                colored(
                    f"Results: Success Rate = {result['pc_success']:.2f}%, "
                    f"Avg Sum Reward = {result['avg_sum_reward']:.2f}",
                    "green",
                    attrs=["bold"],
                )
            )

            # Clean up policy to free memory
            del policy
            torch.cuda.empty_cache()

        # Close environments
        close_envs(envs)

    logging.info(colored(f"\n{'=' * 80}\nAll combinations tested!", "green", attrs=["bold"]))
    logging.info(colored(f"Results saved to: {csv_path}", "yellow", attrs=["bold"]))


if __name__ == "__main__":
    init_logging()
    register_third_party_plugins()
    
    # Parse RTC-specific arguments first
    execution_horizons, inference_delays = parse_rtc_args()
    
    # Call main with parsed RTC parameters
    main(execution_horizons=execution_horizons, inference_delays=inference_delays)
