"""Difficulty adjustment recommendation as a bounded enum."""

from enum import Enum


class DifficultyRecommendation(str, Enum):
    INCREASE = "increase_difficulty"
    MAINTAIN = "maintain_difficulty"
    DECREASE = "decrease_difficulty"
