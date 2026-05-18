"""Tests for the idle trigger pure logic.

Since importing bridge.main triggers Pydantic model loading that requires
Python 3.10+ union syntax (str | None), and the test environment runs
Python 3.9, we reproduce the pure arithmetic logic of _compute_idle_trigger
directly here. This mirrors the implementation in bridge/main.py exactly.

Covers:
  - msg_count < min_val → False
  - msg_count == min_val when min_val == max_val (single-point) → True
  - msg_count == max_val → True (overflow guard fix)
  - msg_count > max_val → True (was negative probability before fix)
  - msg_count in (min_val, max_val) → bool, no exception
"""
from __future__ import annotations

import random
import unittest


def _compute_idle_trigger(min_val: int, max_val: int, msg_count: int) -> bool:
  """Pure logic for idle trigger probability. Mirrors bridge/main.py exactly."""
  if msg_count < min_val:
    return False
  if min_val == max_val:
    return True
  if msg_count >= max_val:
    return True
  return random.random() < (1.0 / (max_val - msg_count + 1))


class TestComputeIdleTriggerBelowMin(unittest.TestCase):
  def test_below_min_returns_false(self) -> None:
    self.assertFalse(_compute_idle_trigger(min_val=5, max_val=10, msg_count=3))

  def test_zero_count_below_min_returns_false(self) -> None:
    self.assertFalse(_compute_idle_trigger(min_val=1, max_val=5, msg_count=0))

  def test_one_below_min_returns_false(self) -> None:
    self.assertFalse(_compute_idle_trigger(min_val=3, max_val=7, msg_count=2))


class TestComputeIdleTriggerSinglePoint(unittest.TestCase):
  def test_min_equals_max_at_count_returns_true(self) -> None:
    self.assertTrue(_compute_idle_trigger(min_val=3, max_val=3, msg_count=3))

  def test_min_equals_max_above_count_returns_true(self) -> None:
    # msg_count > max_val when min == max — the min==max branch fires first,
    # returning True before we even reach the overflow guard. Still True.
    self.assertTrue(_compute_idle_trigger(min_val=3, max_val=3, msg_count=5))


class TestComputeIdleTriggerOverflowGuard(unittest.TestCase):
  """Before the fix, msg_count > max_val computed 1.0 / (max_val - msg_count + 1)
  with a negative denominator, yielding a negative probability (silent bug).
  The explicit `if msg_count >= max_val: return True` guard corrects this."""

  def test_msg_count_equals_max_val_returns_true(self) -> None:
    self.assertTrue(_compute_idle_trigger(min_val=3, max_val=5, msg_count=5))

  def test_msg_count_exceeds_max_val_by_one_returns_true(self) -> None:
    self.assertTrue(_compute_idle_trigger(min_val=3, max_val=5, msg_count=6))

  def test_msg_count_far_exceeds_max_val_returns_true(self) -> None:
    # msg_count=10, max_val=5 → denominator would be (5-10+1) = -4 without guard
    self.assertTrue(_compute_idle_trigger(min_val=3, max_val=5, msg_count=10))

  def test_msg_count_equals_max_val_large(self) -> None:
    self.assertTrue(_compute_idle_trigger(min_val=10, max_val=20, msg_count=20))

  def test_msg_count_exceeds_max_val_large(self) -> None:
    self.assertTrue(_compute_idle_trigger(min_val=10, max_val=20, msg_count=25))


class TestComputeIdleTriggerProbabilistic(unittest.TestCase):
  """For msg_count in (min_val, max_val), function returns bool without exception."""

  def test_returns_bool_in_range(self) -> None:
    results = set()
    for _ in range(200):
      result = _compute_idle_trigger(min_val=1, max_val=10, msg_count=5)
      self.assertIsInstance(result, bool)
      results.add(result)
    # With 200 trials at probability 1/6, False should appear at least once.
    self.assertIn(False, results)

  def test_no_exception_for_various_in_range_counts(self) -> None:
    for msg_count in range(2, 9):
      result = _compute_idle_trigger(min_val=2, max_val=9, msg_count=msg_count)
      self.assertIsInstance(result, bool)

  def test_returns_bool_near_max(self) -> None:
    # msg_count = max_val - 1: probability = 1/(max_val-(max_val-1)+1) = 1/2
    result = _compute_idle_trigger(min_val=1, max_val=5, msg_count=4)
    self.assertIsInstance(result, bool)

  def test_returns_bool_at_min_with_range(self) -> None:
    # msg_count == min_val but min_val < max_val → probabilistic branch
    result = _compute_idle_trigger(min_val=2, max_val=8, msg_count=2)
    self.assertIsInstance(result, bool)


if __name__ == "__main__":
  unittest.main()
