"""Shared config/helpers for the LIAR fake-news-detection notebooks.

Import this from every v2 notebook so the random seed (and, later, other
shared preprocessing) can never diverge between the baseline and proposed
pipelines. See claude-workspace/ISSUE_PLAN.md Phase 0/1.
"""

RANDOM_STATE = 42
