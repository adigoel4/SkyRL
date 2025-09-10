"""
Trajectory logging utilities for debugging and analysis during training.

This module provides a flexible framework for logging complete trajectories
including model outputs and environment observations/feedback.
"""

import os
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
import pandas as pd
from omegaconf import DictConfig
from skyrl_train.inference_engines.base import ConversationType


class TrajectoryLogger(ABC):
    """
    Abstract base class for trajectory logging.
    
    Subclasses should implement the log method to handle trajectory
    data in their specific format (e.g., wandb tables, CSV files, etc.).
    
    TODO: Allow users to bring a custom trajectory logger. They should be able to
    define their own class (outside of the skyrl-train package) and add it to
    a registry (see AdvantageEstimatorRegistry for an example) so that it can
    be referenced by name in the config.
    """
    
    def __init__(self, max_trajectories: int = -1):
        """
        Initialize the trajectory logger.
        
        Args:
            max_trajectories: Maximum number of trajectories to log per batch (-1 for unlimited)
        """
        self.max_trajectories = max_trajectories
        self.logged_count = 0
    
    @abstractmethod
    def log(
        self, 
        prompts: List[str],
        responses: List[str], 
        rewards: List[float],
        step: int,
        prefix: str = "train"
    ) -> None:
        """
        Log a batch of trajectories.
        
        Args:
            prompts: List of prompts
            responses: List of responses
            rewards: List of rewards
            step: Current training step
            prefix: Prefix for logging (e.g., "train", "eval")
        """
        raise NotImplementedError()
    
    def _format_conversation(self, conversation: ConversationType) -> str:
        """
        Format a conversation history into a readable string.
        
        Args:
            conversation: List of message dictionaries
            
        Returns:
            Formatted conversation string
        """
        formatted = []
        for msg in conversation:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            formatted.append(f"[{role}]: {content}")
        return "\n".join(formatted)
    
    def to_dataframe(self, prompts: List[str], responses: List[str], rewards: List[float]) -> pd.DataFrame:
        """
        Convert trajectories to a pandas DataFrame for analysis.
        
        Args:
            prompts: List of prompts
            responses: List of responses 
            rewards: List of rewards
            
        Returns:
            DataFrame with trajectory data
        """
        data = {
            "prompt": prompts,
            "response": responses,
            "reward": rewards
        }
        
        return pd.DataFrame(data)


class WandbTableTrajectoryLogger(TrajectoryLogger):
    """
    Trajectory logger that uploads trajectories as tables to Weights & Biases.
    """
    
    def __init__(
        self, 
        max_trajectories: int = -1
    ):
        """
        Initialize the WandB table trajectory logger.
        
        Args:
            max_trajectories: Maximum number of trajectories to log per batch (-1 for unlimited)
        """
        super().__init__(max_trajectories)
        
        import wandb
        self.wandb = wandb
    
    def log(
        self, 
        prompts: List[str],
        responses: List[str], 
        rewards: List[float],
        step: int,
        prefix: str = "train"
    ) -> None:
        """
        Log trajectories to wandb as a table.
        
        Args:
            prompts: List of prompts
            responses: List of responses
            rewards: List of rewards
            step: Current training step
            prefix: Prefix for logging (e.g., "train", "eval")
        """
        # Limit number of trajectories if specified
        if self.max_trajectories > 0 and len(prompts) > self.max_trajectories:
            prompts = prompts[:self.max_trajectories]
            responses = responses[:self.max_trajectories]
            rewards = rewards[:self.max_trajectories]
        
        # Define columns
        columns = ["step", "prompt", "response", "reward"]
        
        # Create table
        table = self.wandb.Table(columns=columns)
        
        # Add rows for each trajectory
        for prompt, response, reward in zip(prompts, responses, rewards):
            row_data = [step, prompt, response, reward]
            table.add_data(*row_data)
        
        # Log to wandb
        self.wandb.log({f"{prefix}/trajectories": table}, step=step)
        self.logged_count += len(prompts)


class CSVTrajectoryLogger(TrajectoryLogger):
    """
    Trajectory logger that saves trajectories to CSV files for offline analysis.
    """
    
    def __init__(
        self,
        output_dir: str,
        max_trajectories: int = -1
    ):
        """
        Initialize the CSV trajectory logger.
        
        Args:
            output_dir: Directory to save CSV files
            max_trajectories: Maximum number of trajectories to log per batch (-1 for unlimited)
        """
        super().__init__(max_trajectories)
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def log(
        self,
        prompts: List[str],
        responses: List[str], 
        rewards: List[float],
        step: int,
        prefix: str = "train"
    ) -> None:
        """
        Save trajectories to a CSV file.
        
        Args:
            prompts: List of prompts
            responses: List of responses
            rewards: List of rewards
            step: Current training step
            prefix: Prefix for logging (e.g., "train", "eval")
        """
        # Limit number of trajectories if specified
        if self.max_trajectories > 0 and len(prompts) > self.max_trajectories:
            prompts = prompts[:self.max_trajectories]
            responses = responses[:self.max_trajectories]
            rewards = rewards[:self.max_trajectories]
        
        # Convert to dataframe
        df = self.to_dataframe(prompts, responses, rewards)
        df["step"] = step
        df["prefix"] = prefix
        
        # Save to CSV
        filename = os.path.join(self.output_dir, f"{prefix}_trajectories_step_{step}.csv")
        df.to_csv(filename, index=False)
        
        self.logged_count += len(prompts)


def create_trajectory_logger_from_config(
    logging_cfg: DictConfig,
    export_path: Optional[str] = None
) -> Optional[TrajectoryLogger]:
    """
    Create trajectory logger from configuration following repo factory pattern.
    
    Args:
        logging_cfg: Trajectory logging configuration
        export_path: Base export path from config
        
    Returns:
        TrajectoryLogger instance or None if disabled
    """
    assert logging_cfg.get('enabled', False), "Trajectory logging must be enabled to create logger"
        
    logger_type = logging_cfg.get('type', 'wandb')
    
    if logger_type == 'wandb':
        return WandbTableTrajectoryLogger(
            max_trajectories=logging_cfg.get('max_trajectories', -1)
        )
    elif logger_type == 'csv':
        # Use export_path + '/trajectory_logs' as default if not specified
        if export_path and 'output_dir' not in logging_cfg:
            output_dir = os.path.join(export_path, 'trajectory_logs')
        else:
            output_dir = logging_cfg.get('output_dir', './trajectory_logs')
        return CSVTrajectoryLogger(
            output_dir=output_dir,
            max_trajectories=logging_cfg.get('max_trajectories', -1)
        )
    else:
        raise ValueError(f"Unknown trajectory logger type: {logger_type}")