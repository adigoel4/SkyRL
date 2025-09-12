"""
This module provides a framework for logging prompts, responses, and rewards during
RL training. Built-in loggers support Weights & Biases tables and CSV files.

Example:
    # In your config:
    trajectory_logging:
      enabled: true
      type: "wandb"  # or "csv"
      max_trajectories: 100
      
    # Custom logger:
    @register_trajectory_logger("custom")
    class MyLogger(TrajectoryLogger):
        def log(self, prompts, responses, rewards, step, prefix="train"):
            # Your custom logging logic here
            pass
"""

import os
from abc import ABC, abstractmethod
from typing import List, Optional
import pandas as pd
from omegaconf import DictConfig


class TrajectoryLogger(ABC):
    """
    Abstract base class for trajectory logging.
    
    Custom loggers should inherit from this class and implement the log() method.
    The logger will be called once per training batch with prompts, responses, and rewards.
    """
    
    def __init__(self, max_trajectories: int = -1):
        self.max_trajectories = max_trajectories
    
    @abstractmethod
    def log(
        self, 
        prompts: List[str],
        responses: List[str], 
        rewards: List[float],
        step: int,
        prefix: str = "train"
    ) -> None:
        pass


class WandbTableTrajectoryLogger(TrajectoryLogger):
    
    def __init__(self, max_trajectories: int = -1):
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
        if self.max_trajectories > 0:
            prompts = prompts[:self.max_trajectories]
            responses = responses[:self.max_trajectories]
            rewards = rewards[:self.max_trajectories]
        
        table = self.wandb.Table(columns=["step", "prompt", "response", "reward"])
        for prompt, response, reward in zip(prompts, responses, rewards):
            table.add_data(step, prompt, response, reward)
        
        self.wandb.log({f"{prefix}/trajectories": table}, step=step)


class CSVTrajectoryLogger(TrajectoryLogger):
    
    def __init__(self, output_dir: str, max_trajectories: int = -1):
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
        if self.max_trajectories > 0:
            prompts = prompts[:self.max_trajectories]
            responses = responses[:self.max_trajectories]
            rewards = rewards[:self.max_trajectories]
        
        df = pd.DataFrame({
            "step": step,
            "prefix": prefix,
            "prompt": prompts,
            "response": responses,
            "reward": rewards
        })
        
        filename = os.path.join(self.output_dir, f"{prefix}_trajectories_step_{step}.csv")
        df.to_csv(filename, index=False)


class TrajectoryLoggerRegistry:
    """
    Allows users to register custom trajectory loggers without modifying skyrl_train.
    Built-in loggers ("wandb", "csv") are pre-registered.
    """

    _loggers = {}

    @classmethod
    def register(cls, name: str, logger_class):
        """Register a trajectory logger class."""
        if name in cls._loggers:
            raise ValueError(f"Trajectory logger '{name}' already registered")
        cls._loggers[name] = logger_class

    @classmethod
    def get(cls, name: str):
        """Get a registered trajectory logger class by name."""
        if name not in cls._loggers:
            available = list(cls._loggers.keys())
            raise ValueError(f"Unknown trajectory logger '{name}'. Available: {available}")
        return cls._loggers[name]

    @classmethod
    def list_available(cls):
        """List all registered trajectory logger names."""
        return list(cls._loggers.keys())


def register_trajectory_logger(name: str):
    """Decorator to register a trajectory logger class."""

    def decorator(cls):
        TrajectoryLoggerRegistry.register(name, cls)
        return cls

    return decorator


# Register built-in trajectory loggers
TrajectoryLoggerRegistry.register("wandb", WandbTableTrajectoryLogger)
TrajectoryLoggerRegistry.register("csv", CSVTrajectoryLogger)


def create_trajectory_logger_from_config(logging_cfg: DictConfig) -> TrajectoryLogger:
    assert logging_cfg.enabled
    
    # Get the logger class from the registry
    logger_class = TrajectoryLoggerRegistry.get(logging_cfg.type)
    
    # Prepare kwargs for the logger constructor
    # Pass all config parameters except 'enabled' and 'type'
    kwargs = {k: v for k, v in logging_cfg.items() if k not in ['enabled', 'type']}
    
    # Create and return the logger instance
    return logger_class(**kwargs)