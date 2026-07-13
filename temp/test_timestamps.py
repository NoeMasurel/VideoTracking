"""
test_timestamps.py — Unit tests for timestamps.py

Timestamps are stored as **frame numbers** (integers). Conversion to
human-readable mm:ss strings only happens in format_timestamps(), which
requires an explicit fps argument, and in the overlay renderer.

CSV and JSON outputs therefore contain raw frame integers for start/end,
not formatted time strings.
"""

import json
import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import MagicMock, patch, call

from timestamps import (
    is_valid_annotation,
    format_seconds,
    format_timestamps,
    build_clips,
    update_csv,
    update_json,
    draw_overlay,
    handle_annotation_key,
    handle_paused_key,
    handle_playback_key,
    _handle_start_key,
    _handle_end_key,
    collect_missing_annotations,
    finalize_timestamps,
    Clip,
    PlaybackState,
    KEY_NONE,
    KEY_ENTER,
    KEY_BACKSPACE,
    KEY_SPACE,
    KEY_Q,
    KEY_ESC,
    KEY_P,
    KEY_B,
    KEY_N,
    KEY_S,
    KEY_E,
)


# ---------------------------------------------------------------------------
# is_valid_annotation
# ---------------------------------------------------------------------------

class TestIsValidAnnotation:
    def test_valid_uppercase(self):
        assert is_valid_annotation("A1") is True

    def test_valid_lowercase_letter(self):
        assert is_valid_annotation("b3") is True

    def test_valid_zero_digit(self):
        assert is_valid_annotation("Z0") is True

    def test_too_short(self):
        assert is_valid_annotation("A") is False

    def test_too_long(self):
        assert is_valid_annotation("A12") is False

    def test_empty(self):
        assert is_valid_annotation("") is False

    def test_digit_first(self):
        assert is_valid_annotation("1A") is False

    def test_two_letters(self):
        assert is_valid_annotation("AB") is False

    def test_two_digits(self):
        assert is_valid_annotation("12") is False

    def test_special_char(self):
        assert is_valid_annotation("A!") is False


# ---------------------------------------------------------------------------
# format_seconds
# ---------------------------------------------------------------------------

class TestFormatSeconds:
    def test_zero(self):
        assert format_seconds(0) == "00:00"

    def test_one_minute(self):
        assert format_seconds(60) == "01:00"

    def test_mixed(self):
        assert format_seconds(75) == "01:15"

    def test_large(self):
        assert format_seconds(3661) == "61:01"

    def test_single_digit_seconds(self):
        assert format_seconds(5) == "00:05"


# ---------------------------------------------------------------------------
# format_timestamps
# ---------------------------------------------------------------------------

class TestFormatTimestamps:
    # fps=1.0 makes frames == seconds, keeping expected strings readable.
    # fps=30.0 tests realistic conversion (e.g. 900 frames → 00:30).

    def test_empty(self):
        assert format_timestamps([], fps=30.0) == ""

    def test_one_complete_segment_fps1(self):
        assert format_timestamps([0, 30], fps=1.0) == "00:00-00:30"

    def test_one_complete_segment_fps30(self):
        # 0 frames → 00:00, 900 frames at 30fps → 00:30
        assert format_timestamps([0, 900], fps=30.0) == "00:00-00:30"

    def test_two_complete_segments_fps1(self):
        assert format_timestamps([0, 30, 60, 90], fps=1.0) == "00:00-00:30 01:00-01:30"

    def test_two_complete_segments_fps30(self):
        # 0, 900, 1800, 2700 frames at 30fps → 0s, 30s, 60s, 90s
        assert format_timestamps([0, 900, 1800, 2700], fps=30.0) == "00:00-00:30 01:00-01:30"

    def test_open_trailing_segment(self):
        result = format_timestamps([0, 30, 60], fps=1.0)
        assert result == "00:00-00:30 01:00-??:??"

    def test_single_open_segment(self):
        result = format_timestamps([10], fps=1.0)
        assert result == "00:10-??:??"


# ---------------------------------------------------------------------------
# build_clips
# ---------------------------------------------------------------------------

class TestBuildClips:
    # build_clips stores raw frame integers in Clip.start / Clip.end.
    # No fps conversion happens here.

    def test_empty(self):
        assert build_clips([]) == []

    def test_one_clip_stores_frames(self):
        clips = build_clips([0, 30])
        assert len(clips) == 1
        assert clips[0].start == 0     # raw frame number
        assert clips[0].end == 30      # raw frame number

    def test_two_clips_stores_frames(self):
        clips = build_clips([0, 30, 60, 90])
        assert len(clips) == 2
        assert clips[1].start == 60
        assert clips[1].end == 90

    def test_ignores_unclosed_trailing(self):
        clips = build_clips([0, 30, 60])
        assert len(clips) == 1


# ---------------------------------------------------------------------------
# update_csv
# ---------------------------------------------------------------------------

def make_clips(pairs: list[tuple[int, int]]) -> list[Clip]:
    return [Clip(start=s, end=e) for s, e in pairs]


def make_source(name: str = "video.mp4") -> Path:
    p = MagicMock(spec=Path)
    p.name = name
    return p

def read(csv_path: Path) -> pd.DataFrame:
    return pd.read_csv(csv_path)

class TestUpdateCsv:
    # start/end are persisted as raw frame integers (not mm:ss strings).

    def test_creates_new_file(self, tmp_path):
        csv_path = tmp_path / "out.csv"
        update_csv(csv_path, ["A1"], [0, 30], Path("video.mp4"))
        df = pd.read_csv(csv_path)
        assert len(df) == 1
        assert df.iloc[0]["segment"] == "A1"
        assert df.iloc[0]["start"] == 0    # frame integer
        assert df.iloc[0]["end"] == 30     # frame integer
        assert df.iloc[0]["video"] == "video.mp4"
    
    def test_creates_csv_when_missing(self, tmp_path):
        """No existing CSV → file is created with the new rows."""
        csv_path = tmp_path / "out.csv"
        clips = make_clips([(0, 10), (20, 30)])

        with patch("timestamps.build_clips", return_value=clips):
            update_csv(csv_path, ["a", "b"], [0, 10, 20, 30], make_source())

        df = read(csv_path)
        assert len(df) == 2
        assert list(df["segment"]) == ["a", "b"]

    def test_appends_to_existing_csv(self, tmp_path):
        """Existing CSV rows are preserved and new rows are appended."""
        csv_path = tmp_path / "out.csv"
        pd.DataFrame([{"video": "video.mp4", "segment": "a", "start": 0, "end": 10}]).to_csv(csv_path, index=False)

        clips = make_clips([(50, 60)])

        with patch("timestamps.build_clips", return_value=clips):
            update_csv(csv_path, ["b"], [50, 60], make_source())

        df = read(csv_path)
        assert len(df) == 2
        assert set(df["segment"]) == {"a", "b"}

    def test_close_duplicate_is_skipped(self, tmp_path):
        """New clip within 10s of an existing one with same segment is not added."""
        csv_path = tmp_path / "out.csv"
        pd.DataFrame([{"video": "video.mp4", "segment": "a", "start": 10, "end": 20}]).to_csv(csv_path, index=False)

        clips = make_clips([(12, 19)])  # within 10s on both start and end

        with patch("timestamps.build_clips", return_value=clips):
            update_csv(csv_path, ["a"], [12, 19], make_source())

        df = read(csv_path)
        assert len(df) == 1  # no new row added
    
    def test_far_duplicate_is_kept(self, tmp_path):
        """New clip far from an existing one with same segment is added."""
        csv_path = tmp_path / "out.csv"
        pd.DataFrame([{"video": "video.mp4", "segment": "a", "start": 10, "end": 20}]).to_csv(csv_path, index=False)

        clips = make_clips([(25, 40)])  # more than 10s apart

        with patch("timestamps.build_clips", return_value=clips):
            update_csv(csv_path, ["a"], [25, 40], make_source())

        df = read(csv_path)
        assert len(df) == 2

    def test_boundary_exactly_10s_apart_is_kept(self, tmp_path):
        """Clips exactly 10s apart are NOT considered close (strict < 10)."""
        csv_path = tmp_path / "out.csv"
        pd.DataFrame([{"video": "video.mp4", "segment": "a", "start": 0, "end": 20}]).to_csv(csv_path, index=False)

        clips = make_clips([(10, 30)])  # exactly 10s on both → not close

        with patch("timestamps.build_clips", return_value=clips):
            update_csv(csv_path, ["a"], [10, 30], make_source())

        df = read(csv_path)
        assert len(df) == 2

    def test_different_segment_same_times_is_kept(self, tmp_path):
        """Same timestamps but different segment name → always added."""
        csv_path = tmp_path / "out.csv"
        pd.DataFrame([{"video": "video.mp4", "segment": "a", "start": 10, "end": 20}]).to_csv(csv_path, index=False)

        clips = make_clips([(12, 19)])

        with patch("timestamps.build_clips", return_value=clips):
            update_csv(csv_path, ["b"], [12, 19], make_source())  # segment "b", not "a"

        df = read(csv_path)
        assert len(df) == 2


    def test_different_video_same_segment_and_times_is_kept(self, tmp_path):
        """Same segment and times but different video → always added."""
        csv_path = tmp_path / "out.csv"
        pd.DataFrame([{"video": "other.mp4", "segment": "a", "start": 10, "end": 20}]).to_csv(csv_path, index=False)

        clips = make_clips([(12, 19)])

        with patch("timestamps.build_clips", return_value=clips):
            update_csv(csv_path, ["a"], [12, 19], make_source("video.mp4"))

        df = read(csv_path)
        assert len(df) == 2


    def test_mismatched_annotations_and_clips_raises(self, tmp_path):
        """Mismatched annotations / clips lengths raise ValueError before any I/O."""
        csv_path = tmp_path / "out.csv"
        clips = make_clips([(0, 10), (20, 30)])

        with patch("timestamps.build_clips", return_value=clips):
            with pytest.raises(ValueError, match="annotations and clips must match"):
                update_csv(csv_path, ["only_one"], [0, 10, 20, 30], make_source())

        assert not csv_path.exists()


    def test_corrupted_csv_is_treated_as_empty(self, tmp_path):
        """A CSV that can't be parsed is silently replaced."""
        csv_path = tmp_path / "out.csv"
        csv_path.write_bytes(b"\xff\xfe" + b"\x00" * 100)  # binary garbage, not parseable

        clips = make_clips([(0, 10)])

        with patch("timestamps.build_clips", return_value=clips):
            update_csv(csv_path, ["a"], [0, 10], make_source())

        df = read(csv_path)
        assert len(df) == 1
        assert df.iloc[0]["segment"] == "a"

    def test_appends_to_existing(self, tmp_path):
        csv_path = tmp_path / "out.csv"
        update_csv(csv_path, ["A1"], [0, 30], Path("video.mp4"))
        update_csv(csv_path, ["B2"], [60, 90], Path("video.mp4"))
        df = pd.read_csv(csv_path)
        assert len(df) == 2

    def test_deduplicates_by_video_and_segment(self, tmp_path):
        csv_path = tmp_path / "out.csv"
        update_csv(csv_path, ["A1"], [0, 30], Path("video.mp4"))
        # Same video + segment with updated frame range — should replace
        update_csv(csv_path, ["A1"], [5, 35], Path("video.mp4"))
        df = pd.read_csv(csv_path)
        assert len(df) == 1
        assert df.iloc[0]["start"] == 5   # updated frame number

    def test_mismatched_annotations_raises(self, tmp_path):
        csv_path = tmp_path / "out.csv"
        with pytest.raises(ValueError):
            update_csv(csv_path, ["A1", "B2"], [0, 30], Path("video.mp4"))

    def test_multiple_clips(self, tmp_path):
        csv_path = tmp_path / "out.csv"
        update_csv(csv_path, ["A1", "B2"], [0, 30, 60, 90], Path("video.mp4"))
        df = pd.read_csv(csv_path)
        assert len(df) == 2


# ---------------------------------------------------------------------------
# update_json
# ---------------------------------------------------------------------------

class TestUpdateJson:
    # start/end inside clips are persisted as raw frame integers.

    def test_creates_new_file(self, tmp_path):
        json_path = tmp_path / "out.json"
        update_json(json_path, ["A1"], [0, 30], Path("video.mp4"))
        with json_path.open() as f:
            data = json.load(f)
        assert len(data) == 1
        assert data[0]["street"] == "A1"
        assert data[0]["clips"]["video.mp4"][0] == {"start": 0, "end": 30}  # frame ints

    def test_appends_clip_to_existing_street(self, tmp_path):
        json_path = tmp_path / "out.json"
        update_json(json_path, ["A1"], [0, 30], Path("video1.mp4"))
        update_json(json_path, ["A1"], [60, 90], Path("video2.mp4"))
        with json_path.open() as f:
            data = json.load(f)
        assert len(data) == 1
        assert "video1.mp4" in data[0]["clips"]
        assert "video2.mp4" in data[0]["clips"]

    def test_multiple_clips_same_file(self, tmp_path):
        json_path = tmp_path / "out.json"
        update_json(json_path, ["A1"], [0, 30], Path("video.mp4"))
        update_json(json_path, ["A1"], [60, 90], Path("video.mp4"))
        with json_path.open() as f:
            data = json.load(f)
        assert len(data[0]["clips"]["video.mp4"]) == 2

    def test_handles_corrupted_json(self, tmp_path):
        json_path = tmp_path / "out.json"
        json_path.write_text("not valid json")
        update_json(json_path, ["A1"], [0, 30], Path("video.mp4"))
        with json_path.open() as f:
            data = json.load(f)
        assert len(data) == 1

    def test_multiple_streets(self, tmp_path):
        json_path = tmp_path / "out.json"
        update_json(json_path, ["A1", "B2"], [0, 30, 60, 90], Path("video.mp4"))
        with json_path.open() as f:
            data = json.load(f)
        streets = {entry["street"] for entry in data}
        assert streets == {"A1", "B2"}


# ---------------------------------------------------------------------------
# handle_annotation_key
# ---------------------------------------------------------------------------

class TestHandleAnnotationKey:
    def _state(self):
        s = PlaybackState()
        s.waiting_for_annotation = True
        s.is_paused = True
        return s

    def test_no_key_does_nothing(self):
        state = self._state()
        handle_annotation_key(KEY_NONE, state)
        assert state.current_annotation == ""

    def test_types_letter(self):
        state = self._state()
        handle_annotation_key(ord("A"), state)
        assert state.current_annotation == "A"

    def test_digit_ignored_as_first_char(self):
        state = self._state()
        handle_annotation_key(ord("1"), state)
        assert state.current_annotation == ""

    def test_types_letter_then_digit(self):
        state = self._state()
        handle_annotation_key(ord("A"), state)
        handle_annotation_key(ord("1"), state)
        assert state.current_annotation == "A1"

    def test_letter_ignored_as_second_char(self):
        state = self._state()
        handle_annotation_key(ord("A"), state)
        handle_annotation_key(ord("B"), state)
        assert state.current_annotation == "A"

    def test_backspace_removes_last_char(self):
        state = self._state()
        state.current_annotation = "A"
        handle_annotation_key(KEY_BACKSPACE, state)
        assert state.current_annotation == ""

    def test_enter_with_valid_annotation_confirms(self):
        state = self._state()
        state.current_annotation = "A1"
        handle_annotation_key(KEY_ENTER, state)
        assert state.annotations == ["A1"]
        assert state.waiting_for_annotation is False
        assert state.is_paused is False
        assert state.current_annotation == ""

    def test_enter_with_invalid_annotation_does_nothing(self):
        state = self._state()
        state.current_annotation = "A"
        handle_annotation_key(KEY_ENTER, state)
        assert state.annotations == []
        assert state.waiting_for_annotation is True

    def test_enter_uppercases_annotation(self):
        state = self._state()
        state.current_annotation = "b3"
        handle_annotation_key(KEY_ENTER, state)
        assert state.annotations == ["B3"]

    def test_max_two_chars(self):
        state = self._state()
        state.current_annotation = "A1"
        handle_annotation_key(ord("2"), state)  # third char — should be ignored
        assert state.current_annotation == "A1"


# ---------------------------------------------------------------------------
# _handle_start_key  /  _handle_end_key
#
# fps is now stored on state.fps, not passed as an argument.
# ---------------------------------------------------------------------------

class TestHandleStartKey:
    def test_starts_recording(self):
        state = PlaybackState()
        state.fps = 1.0
        state.current_frame = 30
        _handle_start_key(state)
        assert state.is_recording is True
        assert state.timestamps == [30]   # raw frame number

    def test_closes_and_reopens_segment(self):
        state = PlaybackState()
        state.fps = 1.0
        state.current_frame = 60
        state.is_recording = True
        state.timestamps = [0]
        _handle_start_key(state)
        assert state.is_recording is False
        assert state.waiting_for_annotation is True
        assert state.is_paused is True
        # timestamps[1] and [2] are both the current frame number (60)
        assert state.timestamps[1] == 60
        assert state.timestamps[2] == 60


class TestHandleEndKey:
    def test_ends_recording(self):
        state = PlaybackState()
        state.fps = 1.0
        state.current_frame = 30
        state.is_recording = True
        state.timestamps = [0]
        _handle_end_key(state)
        assert state.is_recording is False
        assert state.timestamps == [0, 30]   # raw frame numbers
        assert state.waiting_for_annotation is True

    def test_no_effect_when_not_recording_and_no_timestamps(self):
        state = PlaybackState()
        state.fps = 1.0
        state.current_frame = 30
        _handle_end_key(state)
        assert state.timestamps == []

    def test_creates_zero_length_clip_when_not_recording(self):
        """When E is pressed while not recording, a new clip is opened from
        the last known end frame up to the current frame."""
        state = PlaybackState()
        state.fps = 1.0
        state.current_frame = 60
        state.is_recording = False
        state.timestamps = [0, 30]  # one complete clip already
        _handle_end_key(state)
        assert state.timestamps[-2] == 30  # re-uses last end frame as start
        assert state.timestamps[-1] == 60  # current frame as end


# ---------------------------------------------------------------------------
# handle_paused_key
#
# fps is now stored on state.fps; the function signature no longer accepts
# an fps argument. All call sites updated accordingly.
# ---------------------------------------------------------------------------

class TestHandlePausedKey:
    def _make_cap(self):
        cap = MagicMock()
        cap.set = MagicMock()
        cap.read = MagicMock(return_value=(True, MagicMock()))
        return cap

    def _state(self, fps=30.0):
        state = PlaybackState()
        state.fps = fps
        state.is_paused = True
        return state

    def test_q_returns_true(self):
        assert handle_paused_key(KEY_Q, self._state(), self._make_cap(), 0, 9000) is True

    def test_esc_returns_true(self):
        assert handle_paused_key(KEY_ESC, self._state(), self._make_cap(), 0, 9000) is True

    def test_p_resumes(self):
        state = self._state()
        handle_paused_key(KEY_P, state, self._make_cap(), 0, 9000)
        assert state.is_paused is False

    def test_space_resumes(self):
        state = self._state()
        handle_paused_key(KEY_SPACE, state, self._make_cap(), 0, 9000)
        assert state.is_paused is False

    def test_none_key_returns_false(self):
        result = handle_paused_key(KEY_NONE, self._state(), self._make_cap(), 0, 9000)
        assert result is False

    def test_back_clamps_to_start(self):
        state = self._state(fps=30.0)
        state.current_frame = 10
        cap = self._make_cap()
        handle_paused_key(KEY_B, state, cap, 0, 9000)
        # 10 - 5*30 = -140, clamped to start_frame=0
        args = cap.set.call_args[0]
        assert args[1] == 0

    def test_forward_clamps_to_end(self):
        state = self._state(fps=30.0)
        state.current_frame = 8990
        cap = self._make_cap()
        handle_paused_key(KEY_N, state, cap, 0, 9000)
        # 8990 + 5*30 = 9140, clamped to end_frame=9000
        args = cap.set.call_args[0]
        assert args[1] == 9000


# ---------------------------------------------------------------------------
# handle_playback_key
#
# fps is now stored on state.fps; the function signature no longer accepts
# an fps argument. All call sites updated accordingly.
# ---------------------------------------------------------------------------

class TestHandlePlaybackKey:
    def _make_cap(self):
        cap = MagicMock()
        cap.set = MagicMock()
        return cap

    def _state(self, fps=30.0):
        state = PlaybackState()
        state.fps = fps
        return state

    def test_q_returns_true(self):
        assert handle_playback_key(KEY_Q, self._state(), self._make_cap(), 0, 9000) is True

    def test_esc_returns_true(self):
        assert handle_playback_key(KEY_ESC, self._state(), self._make_cap(), 0, 9000) is True

    def test_p_pauses(self):
        state = self._state()
        handle_playback_key(KEY_P, state, self._make_cap(), 0, 9000)
        assert state.is_paused is True

    def test_s_starts_recording(self):
        state = self._state(fps=1.0)
        state.current_frame = 90
        handle_playback_key(KEY_S, state, self._make_cap(), 0, 9000)
        assert state.is_recording is True
        assert state.timestamps == [90]   # raw frame number

    def test_e_ends_recording(self):
        state = self._state(fps=1.0)
        state.current_frame = 60
        state.is_recording = True
        state.timestamps = [0]
        handle_playback_key(KEY_E, state, self._make_cap(), 0, 9000)
        assert state.is_recording is False
        assert state.timestamps == [0, 60]  # raw frame numbers

    def test_none_key_returns_false(self):
        result = handle_playback_key(KEY_NONE, self._state(), self._make_cap(), 0, 9000)
        assert result is False


# ---------------------------------------------------------------------------
# collect_missing_annotations
# ---------------------------------------------------------------------------

class TestCollectMissingAnnotations:
    def test_no_missing(self):
        state = PlaybackState()
        state.timestamps = [0, 30]
        state.annotations = ["A1"]
        collect_missing_annotations(state)
        assert state.annotations == ["A1"]

    def test_prompts_for_missing(self, monkeypatch):
        state = PlaybackState()
        state.timestamps = [0, 30]
        state.annotations = []
        monkeypatch.setattr("builtins.input", lambda _: "B2")
        collect_missing_annotations(state)
        assert state.annotations == ["B2"]

    def test_retries_on_invalid_input(self, monkeypatch):
        state = PlaybackState()
        state.timestamps = [0, 30]
        state.annotations = []
        responses = iter(["bad", "also_bad", "C3"])
        monkeypatch.setattr("builtins.input", lambda _: next(responses))
        collect_missing_annotations(state)
        assert state.annotations == ["C3"]


# ---------------------------------------------------------------------------
# finalize_timestamps
#
# fps is now stored on state.fps, not passed as an argument.
# Signature: finalize_timestamps(state, source_path, output_path, format)
# ---------------------------------------------------------------------------

class TestFinalizeTimestamps:
    def test_returns_empty_list_when_no_timestamps(self, tmp_path, capsys):
        state = PlaybackState()
        state.fps = 30.0
        result = finalize_timestamps(state, Path("v.mp4"), tmp_path / "out.json", "json")
        assert result == []
        captured = capsys.readouterr()
        assert "No timestamps" in captured.out

    def test_closes_open_segment(self, tmp_path, monkeypatch):
        state = PlaybackState()
        state.fps = 1.0
        state.timestamps = [0]         # one open segment (frame 0)
        state.current_frame = 90       # will be used to close it
        state.annotations = []
        monkeypatch.setattr("builtins.input", lambda _: "A1")
        finalize_timestamps(state, Path("v.mp4"), tmp_path / "out.json", "json")
        assert state.timestamps == [0, 90]  # closed with current_frame

    def test_saves_json(self, tmp_path, monkeypatch):
        state = PlaybackState()
        state.fps = 1.0
        state.timestamps = [0, 30]
        state.annotations = ["A1"]
        out = tmp_path / "out.json"
        finalize_timestamps(state, Path("v.mp4"), out, "json")
        assert out.exists()
        with out.open() as f:
            data = json.load(f)
        assert data[0]["street"] == "A1"

    def test_saves_csv(self, tmp_path):
        state = PlaybackState()
        state.fps = 1.0
        state.timestamps = [0, 30]
        state.annotations = ["B2"]
        out = tmp_path / "out.csv"
        finalize_timestamps(state, Path("v.mp4"), out, "csv")
        assert out.exists()
        df = pd.read_csv(out)
        assert df.iloc[0]["segment"] == "B2"

    def test_returns_formatted_string(self, tmp_path):
        state = PlaybackState()
        state.fps = 1.0
        state.timestamps = [0, 30]
        state.annotations = ["C3"]
        out = tmp_path / "out.csv"
        # fps=1.0: 30 frames → 30 seconds → "00:30"
        result = finalize_timestamps(state, Path("v.mp4"), out, "csv")
        assert result == "00:00-00:30"

    def test_returns_formatted_string_fps30(self, tmp_path):
        state = PlaybackState()
        state.fps = 30.0
        state.timestamps = [0, 900]    # 900 frames at 30fps → 30 seconds
        state.annotations = ["C3"]
        out = tmp_path / "out.csv"
        result = finalize_timestamps(state, Path("v.mp4"), out, "csv")
        assert result == "00:00-00:30"


# ---------------------------------------------------------------------------
# draw_overlay  (smoke tests — just checks it doesn't crash)
# ---------------------------------------------------------------------------

class TestDrawOverlay:
    def _blank_frame(self):
        import numpy as np
        return np.zeros((480, 640, 3), dtype="uint8")

    def test_no_state(self):
        state = PlaybackState()
        out = draw_overlay(self._blank_frame(), state)
        assert out.shape == (480, 640, 3) # type: ignore

    def test_with_timestamps_and_annotations(self):
        state = PlaybackState()
        state.fps = 30.0
        state.timestamps = [0, 30]
        state.annotations = ["A1"]
        out = draw_overlay(self._blank_frame(), state)
        assert out.shape == (480, 640, 3) # type: ignore

    def test_paused(self):
        state = PlaybackState()
        state.is_paused = True
        out = draw_overlay(self._blank_frame(), state)
        assert out.shape == (480, 640, 3) # type: ignore

    def test_waiting_for_annotation(self):
        state = PlaybackState()
        state.waiting_for_annotation = True
        state.current_annotation = "A"
        out = draw_overlay(self._blank_frame(), state)
        assert out.shape == (480, 640, 3) # type: ignore

    def test_does_not_mutate_original_frame(self):
        import numpy as np
        frame = np.zeros((480, 640, 3), dtype="uint8")
        original = frame.copy()
        state = PlaybackState()
        state.fps = 30.0
        state.timestamps = [0, 30]
        state.annotations = ["A1"]
        draw_overlay(frame, state)
        assert (frame == original).all()