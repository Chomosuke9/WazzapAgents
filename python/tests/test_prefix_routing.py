from __future__ import annotations

import os
import random
import sys
import unittest
from pathlib import Path


PYTHON_DIR = Path(__file__).resolve().parents[1]
if str(PYTHON_DIR) not in sys.path:
  sys.path.insert(0, str(PYTHON_DIR))

from bridge import history
from bridge.history import DEFAULT_ASSISTANT_NAME
from bridge.messaging.filtering import (
  _match_prefix_reason,
  _message_matches_prefix,
  _payload_triggers_llm1,
)


def _idle_fires(cfg, msg_count: int, rng=random) -> bool:
  """Standalone mirror of ``_should_idle_trigger`` from ``main.connection_loop``.

  ``_should_idle_trigger`` is defined as a closure inside ``connection_loop``
  and cannot be imported. This helper duplicates its body verbatim so the
  gating invariants can be exercised under unit tests. The ``rng`` parameter
  defaults to the ``random`` module so callers can pass ``random.Random(42)``
  for determinism. Keep this function in lock-step with the closure body in
  ``python/bridge/main.py``: any behavior change there must be mirrored here.
  """
  if cfg is None:
    return False
  min_val, max_val = cfg
  if msg_count < min_val:
    return False
  if min_val == max_val:
    return True
  probability = 1.0 / (max_val - msg_count + 1)
  return rng.random() < probability


def reset_name(name):
  """Point ASSISTANT_NAME at ``name`` and invalidate history's caches."""
  if name is None:
    os.environ.pop("ASSISTANT_NAME", None)
  else:
    os.environ["ASSISTANT_NAME"] = name
  history._last_env_value = object()
  history._cached_names = []
  history._cached_pattern = None


class _HistoryCacheResetMixin:
  """Ensure each test starts (and ends) with a clean history module cache.

  Several tests mutate ``ASSISTANT_NAME`` or poke ``_cached_names`` directly,
  so we defensively reset the module-level cache in both ``setUp`` and
  ``tearDown`` to keep state from leaking across tests.
  """

  _saved_env_value = None

  def setUp(self) -> None:  # type: ignore[override]
    super().setUp()  # type: ignore[misc]
    self._saved_env_value = os.environ.get("ASSISTANT_NAME")
    history._last_env_value = object()
    history._cached_names = []
    history._cached_pattern = None

  def tearDown(self) -> None:  # type: ignore[override]
    if self._saved_env_value is None:
      os.environ.pop("ASSISTANT_NAME", None)
    else:
      os.environ["ASSISTANT_NAME"] = self._saved_env_value
    history._last_env_value = object()
    history._cached_names = []
    history._cached_pattern = None
    super().tearDown()  # type: ignore[misc]


class MessageMatchesPrefixTests(_HistoryCacheResetMixin, unittest.TestCase):
  def test_empty_triggers_never_match(self) -> None:
    payload = {
      "botMentioned": True,
      "repliedToBot": True,
      "messageType": "groupParticipantsUpdate",
      "text": "LLM please",
    }
    self.assertFalse(_message_matches_prefix(payload, set()))

  def test_tag_matches_when_bot_mentioned_and_tag_enabled(self) -> None:
    payload = {"botMentioned": True, "text": "hi"}
    self.assertTrue(_message_matches_prefix(payload, {"tag"}))

  def test_tag_does_not_match_when_tag_disabled(self) -> None:
    payload = {"botMentioned": True, "text": "hi"}
    self.assertFalse(_message_matches_prefix(payload, {"reply", "name", "join"}))

  def test_reply_matches_when_replied_to_bot_and_reply_enabled(self) -> None:
    payload = {"repliedToBot": True, "text": "yo"}
    self.assertTrue(_message_matches_prefix(payload, {"reply"}))

  def test_reply_does_not_match_when_reply_disabled(self) -> None:
    payload = {"repliedToBot": True, "text": "yo"}
    self.assertFalse(_message_matches_prefix(payload, {"tag", "name", "join"}))

  def test_join_matches_group_participants_update_when_join_enabled(self) -> None:
    payload = {"messageType": "groupParticipantsUpdate"}
    self.assertTrue(_message_matches_prefix(payload, {"join"}))

  def test_join_does_not_match_when_join_disabled(self) -> None:
    payload = {"messageType": "groupParticipantsUpdate"}
    self.assertFalse(_message_matches_prefix(payload, {"tag", "reply", "name"}))

  def test_name_matches_text_with_assistant_name(self) -> None:
    reset_name("Vivy")
    payload = {"text": "hey Vivy are you there"}
    self.assertTrue(_message_matches_prefix(payload, {"name"}))

  def test_name_respects_word_boundary(self) -> None:
    reset_name("vivy")
    payload = {"text": "meet vivyana tomorrow"}
    self.assertFalse(_message_matches_prefix(payload, {"name"}))

  def test_name_does_not_fire_on_empty_or_none_text(self) -> None:
    reset_name("Vivy")
    for value in ("", "   ", None):
      with self.subTest(text=repr(value)):
        payload = {"text": value}
        self.assertFalse(_message_matches_prefix(payload, {"name"}))

  def test_name_is_case_insensitive(self) -> None:
    reset_name("Vivy")
    for text in ("VIVY come here", "vivy come here", "ViVy come here"):
      with self.subTest(text=text):
        self.assertTrue(_message_matches_prefix({"text": text}, {"name"}))

  def test_name_multi_alias_matches_any_alias(self) -> None:
    reset_name("Vivy,ivy,vivi")
    for text in ("hello ivy", "sup vivi", "yo Vivy"):
      with self.subTest(text=text):
        self.assertTrue(_message_matches_prefix({"text": text}, {"name"}))
    self.assertFalse(
      _message_matches_prefix({"text": "hello world"}, {"name"})
    )

  def test_name_falls_back_to_default_when_env_unset(self) -> None:
    reset_name(None)
    self.assertEqual(history.assistant_name(), DEFAULT_ASSISTANT_NAME)
    self.assertTrue(
      _message_matches_prefix({"text": "ask LLM please"}, {"name"})
    )
    self.assertFalse(
      _message_matches_prefix({"text": "hello world"}, {"name"})
    )

  def test_name_falls_back_to_default_when_env_empty_and_all_aliases_empty(self) -> None:
    reset_name(",, ,")
    self.assertEqual(history.assistant_name(), DEFAULT_ASSISTANT_NAME)
    self.assertTrue(
      _message_matches_prefix({"text": "LLM time"}, {"name"})
    )
    self.assertFalse(
      _message_matches_prefix({"text": "hello world"}, {"name"})
    )


class MatchPrefixReasonTests(_HistoryCacheResetMixin, unittest.TestCase):
  def test_tag_reported_as_tag(self) -> None:
    self.assertEqual(
      _match_prefix_reason({"botMentioned": True, "text": "hey"}, {"tag"}),
      "tag",
    )

  def test_reply_reported_as_reply(self) -> None:
    self.assertEqual(
      _match_prefix_reason({"repliedToBot": True, "text": "sup"}, {"reply"}),
      "reply",
    )

  def test_join_reported_as_join(self) -> None:
    self.assertEqual(
      _match_prefix_reason(
        {"messageType": "groupParticipantsUpdate"}, {"join"}
      ),
      "join",
    )

  def test_name_reported_with_matched_substring(self) -> None:
    reset_name("Vivy")
    reason = _match_prefix_reason(
      {"text": "hey Vivy please help"}, {"name"}
    )
    self.assertIsNotNone(reason)
    assert reason is not None
    self.assertTrue(reason.startswith("name:"))
    self.assertEqual(reason, "name:Vivy")

  def test_branch_order_tag_beats_name(self) -> None:
    # Both tag and name would match; tag must be reported because it is
    # checked first. Any reordering of branches in _message_matches_prefix
    # must be mirrored in _match_prefix_reason and must update this test.
    reset_name("Vivy")
    payload = {"botMentioned": True, "text": "hey Vivy are you there"}
    self.assertEqual(
      _match_prefix_reason(payload, {"tag", "name"}),
      "tag",
    )

  def test_returns_none_when_nothing_matches(self) -> None:
    reset_name("Vivy")
    payload = {
      "botMentioned": False,
      "repliedToBot": False,
      "messageType": "conversation",
      "text": "regular chat",
    }
    self.assertIsNone(
      _match_prefix_reason(payload, {"tag", "reply", "join", "name"})
    )

  def test_returns_none_when_triggers_empty(self) -> None:
    self.assertIsNone(
      _match_prefix_reason(
        {"botMentioned": True, "repliedToBot": True}, set()
      )
    )


class ShouldIdleTriggerTests(unittest.TestCase):
  def test_none_config_never_fires(self) -> None:
    for msg_count in (0, 1, 10, 10_000):
      with self.subTest(msg_count=msg_count):
        self.assertFalse(_idle_fires(None, msg_count))

  def test_below_min_never_fires(self) -> None:
    self.assertFalse(_idle_fires((5, 10), 0))
    self.assertFalse(_idle_fires((5, 10), 4))

  def test_min_equals_max_always_fires_at_or_above_min(self) -> None:
    self.assertTrue(_idle_fires((3, 3), 3))
    self.assertTrue(_idle_fires((3, 3), 4))
    self.assertTrue(_idle_fires((3, 3), 500))
    # Below min still returns False even when min==max.
    self.assertFalse(_idle_fires((3, 3), 2))

  def test_1_500_distribution_fires_between_20_and_80_times_in_10_000_msgs(self) -> None:
    # Pin the observed ~40-fire-per-10_000-messages behavior of idle(1,500)
    # so a regression that widens the fire condition (e.g. fire-every-message
    # or fire-every-other-message) breaks loudly. With random.Random(42) and
    # the reset-on-fire counter model, this simulation produces exactly 40
    # fires today; we allow [20, 80] for margin across Python versions.
    rng = random.Random(42)
    msg_count = 0
    fires = 0
    cfg = (1, 500)
    for _ in range(10_000):
      msg_count += 1
      if _idle_fires(cfg, msg_count, rng=rng):
        fires += 1
        msg_count = 0
    self.assertGreaterEqual(fires, 20)
    self.assertLessEqual(fires, 80)
    # ~1 fire per 250 messages is the invariant documented in the
    # investigation; sanity-check we are nowhere near "fire every message".
    self.assertLess(fires, 500)


class PrefixModeEndToEndTests(_HistoryCacheResetMixin, unittest.TestCase):
  """Encode the prefix-mode decision invariant as an end-to-end scenario.

  For each payload we compute ``prefix_matched_payloads`` exactly like
  ``process_message_batch`` does and assert whether LLM2 would fire under
  the prefix-mode rules: fire iff any payload in the batch matches.
  """

  def test_prefix_mode_fires_only_on_matching_payloads(self) -> None:
    reset_name("Vivy")
    triggers = {"tag", "reply", "name", "join"}
    sequence = [
      # (payload, expected_fire, label)
      ({"text": "hello world", "botMentioned": False}, False, "plain"),
      ({"text": "hey Vivy", "botMentioned": False}, True, "name"),
      ({"text": "@bot please", "botMentioned": True}, True, "tag"),
      ({"text": "thanks", "repliedToBot": True}, True, "reply"),
      (
        {"text": "", "messageType": "groupParticipantsUpdate"},
        True,
        "join",
      ),
      ({"text": "lunch time"}, False, "plain2"),
      ({"text": "meet vivyana later"}, False, "boundary-miss"),
      ({"text": "VIVY come back"}, True, "case"),
      ({"text": "nothing here"}, False, "plain3"),
      ({"text": "morning everyone"}, False, "plain4"),
    ]
    for payload, expected_fire, label in sequence:
      with self.subTest(label=label):
        matches = [p for p in [payload] if _message_matches_prefix(p, triggers)]
        fires = bool(matches)
        self.assertEqual(fires, expected_fire)

  def test_prefix_mode_with_only_name_trigger_ignores_tag(self) -> None:
    reset_name("Vivy")
    triggers = {"name"}
    tag_only = {"botMentioned": True, "text": "nothing relevant"}
    name_hit = {"botMentioned": False, "text": "hey Vivy"}
    self.assertFalse(_message_matches_prefix(tag_only, triggers))
    self.assertTrue(_message_matches_prefix(name_hit, triggers))

  def test_payload_triggers_llm1_filters_reaction_messages(self) -> None:
    # Belt-and-suspenders: confirm the prefix-mode batch filter that
    # _payload_triggers_llm1 implements excludes reaction messages. The
    # prefix decision only looks at payloads that pass this filter, so a
    # regression that starts routing reactions into LLM1 would silently
    # widen the firing surface.
    reaction = {"messageType": "reactionMessage", "text": "emoji"}
    regular = {"messageType": "conversation", "text": "hi"}
    self.assertFalse(_payload_triggers_llm1(reaction))
    self.assertTrue(_payload_triggers_llm1(regular))


class NameTriggerResilienceTests(_HistoryCacheResetMixin, unittest.TestCase):
  def test_empty_alias_injection_does_not_match_unrelated_text(self) -> None:
    # Directly poke _cached_names to simulate a corrupted/stale cache and
    # freeze _last_env_value so _parse_assistant_names short-circuits
    # without rebuilding. If assistant_name_pattern() is NOT self-defending
    # the resulting regex r'(?i)\b(?:Vivy|)\b' matches at every word
    # boundary, including plain text like "hello world", which would turn
    # every prefix-mode message into a name-trigger fire. The fix lives in
    # history.assistant_name_pattern() and must filter empties out of the
    # alias list before compiling the regex.
    os.environ["ASSISTANT_NAME"] = "Vivy"
    history._last_env_value = "Vivy"
    history._cached_names = ["Vivy", ""]
    history._cached_pattern = None
    pattern = history.assistant_name_pattern()
    self.assertIsNone(pattern.search("hello world"))
    # The legitimate alias still matches.
    self.assertIsNotNone(pattern.search("hey Vivy please"))

  def test_parse_assistant_names_filters_empty_aliases_from_weird_env(self) -> None:
    # Belt-and-suspenders: even before the pattern builder self-defends,
    # _parse_assistant_names already drops empties from pathological
    # comma-separated values. Regressions that drop this filter would be
    # caught here.
    reset_name(",,,Vivy,,")
    self.assertEqual(history._parse_assistant_names(), ["Vivy"])

  def test_all_empty_aliases_fall_back_to_default(self) -> None:
    # Direct-cache-poisoning path: if every alias in _cached_names is
    # empty, the pattern builder must still produce something safe. The
    # hardening in assistant_name_pattern() falls back to the default
    # assistant name rather than compiling an unsafe alternation.
    os.environ["ASSISTANT_NAME"] = "whatever"
    history._last_env_value = "whatever"
    history._cached_names = ["", "  "]
    history._cached_pattern = None
    pattern = history.assistant_name_pattern()
    self.assertIsNone(pattern.search("hello world"))
    self.assertIsNotNone(pattern.search(f"ping {DEFAULT_ASSISTANT_NAME} now"))


class IdleCounterInteractionTests(unittest.TestCase):
  """Guard the counter-reset invariant that the prefix-mode branch relies on.

  The real branch in ``process_message_batch`` increments ``idle_msg_count``
  when no prefix match fires, calls ``_should_idle_trigger``, and resets the
  counter to zero on either a prefix match OR an idle fire. A regression that
  reorders those steps can let the counter grow unboundedly and eventually
  cross ``max_val``, producing the "fires on every message" symptom.
  """

  def test_counter_resets_on_prefix_match(self) -> None:
    cfg = (1, 500)
    rng = random.Random(1)
    msg_count = 0
    observed = []
    # Pattern: 3 plain, 1 match, 3 plain, 1 match, ...
    for i in range(12):
      matched = (i + 1) % 4 == 0
      if matched:
        msg_count = 0
      else:
        msg_count += 1
        if _idle_fires(cfg, msg_count, rng=rng):
          msg_count = 0
      observed.append(msg_count)
    # With the 1-in-4 match cadence and reset-on-match logic, the counter
    # never exceeds the 3 plain messages between matches.
    self.assertLessEqual(max(observed), 3)

  def test_counter_never_exceeds_max_val_during_normal_operation(self) -> None:
    cfg = (1, 500)
    rng = random.Random(42)
    msg_count = 0
    max_seen = 0
    for _ in range(10_000):
      msg_count += 1
      max_seen = max(max_seen, msg_count)
      if _idle_fires(cfg, msg_count, rng=rng):
        msg_count = 0
    self.assertLessEqual(max_seen, cfg[1])

  def test_counter_reset_on_idle_fire(self) -> None:
    # A forced fire (via min==max) must always drop the counter back to 0
    # so the idle cadence restarts cleanly. If the reset were moved to
    # "only on prefix match" the counter would climb forever.
    cfg = (1, 1)
    msg_count = 0
    for _ in range(5):
      msg_count += 1
      if _idle_fires(cfg, msg_count):
        msg_count = 0
    self.assertEqual(msg_count, 0)


if __name__ == "__main__":
  unittest.main()
