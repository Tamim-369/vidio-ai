"""Tests for the background-music mixer.

The filter graph is asserted as a string (fast, no ffmpeg). The tests that
actually run ffmpeg are skipped when it is unavailable, so the suite still
passes on a bare box.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess

import pytest

from src.agents.soundtrack import music
from src.agents.soundtrack.music import MUSIC_PATH, MUSIC_SKIP_S

needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe not installed",
)


def _mean_volume(path: str) -> float:
    result = subprocess.run(
        ["ffmpeg", "-hide_banner", "-i", path, "-af", "volumedetect", "-f", "null", "-"],
        capture_output=True, text=True,
    )
    match = re.search(r"mean_volume:\s*(-?[\d.]+)", result.stderr)
    assert match, result.stderr[-400:]
    return float(match.group(1))


@pytest.fixture(scope="module")
def silent_clip(tmp_path_factory):
    """9:16 clip whose audio is digital silence."""
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not installed")
    path = tmp_path_factory.mktemp("media") / "silent.mp4"
    subprocess.run([
        "ffmpeg", "-y", "-v", "error",
        "-f", "lavfi", "-i", "testsrc=size=608x1080:rate=10:duration=9",
        "-f", "lavfi", "-i", "anullsrc=channel_layout=mono:sample_rate=24000",
        "-shortest", "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-ac", "1", "-ar", "24000", str(path),
    ], check=True)
    return str(path)


@pytest.fixture(scope="module")
def mono_clip(tmp_path_factory):
    """Clip matching real TTS output: mono 24 kHz narration."""
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not installed")
    path = tmp_path_factory.mktemp("media") / "mono.mp4"
    subprocess.run([
        "ffmpeg", "-y", "-v", "error",
        "-f", "lavfi", "-i", "testsrc=size=608x1080:rate=10:duration=9",
        "-f", "lavfi", "-i", "sine=frequency=300:sample_rate=24000:duration=9",
        "-shortest", "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-ac", "1", "-ar", "24000", str(path),
    ], check=True)
    return str(path)


# --- filter graph ------------------------------------------------------------

class TestBuildFilter:
    def test_mix_does_not_normalize_inputs(self):
        # normalize=0 is load-bearing: ffmpeg's default divides by the input
        # count, which would halve the narration as well as the music.
        assert "amix=inputs=2" in music.build_filter(10.0)
        assert "normalize=0" in music.build_filter(10.0)

    def test_mono_narration_is_centered_without_attenuation(self):
        # aformat's mono->stereo conversion costs 3 dB; pan does not.
        graph = music.build_filter(10.0, narr_channels=1)
        assert "pan=stereo|c0=c0|c1=c0" in graph

    def test_stereo_narration_skips_the_upmix(self):
        graph = music.build_filter(10.0, narr_channels=2)
        assert "pan=stereo" not in graph

    def test_both_branches_trimmed_to_exact_duration(self):
        graph = music.build_filter(9.5)
        assert graph.count("atrim=0:9.500") == 2
        assert "-t" not in graph  # length is enforced by atrim, not here

    def test_gain_is_applied_to_the_music_branch_only(self):
        graph = music.build_filter(10.0, gain_db=-20.0)
        assert graph.count("volume=-20.0dB") == 1
        # The gain belongs to the branch that terminates in [m], so it sits
        # after [n] and before [m] — never on the narration branch.
        assert graph.index("volume=-20.0dB") > graph.index("[n]")
        assert graph.index("volume=-20.0dB") < graph.index("[m]")

    def test_the_bed_is_loudness_normalised(self):
        # Without loudnorm the trim is relative to the track's own loudness,
        # so the same gain that sounds right on a hot master silences a quiet
        # one. This is the regression guard for an inaudible Oogway bed.
        graph = music.build_filter(10.0)
        assert "loudnorm=I=-24.0" in graph
        # Normalisation must come BEFORE the trim, or the trim is meaningless.
        assert graph.index("loudnorm") < graph.index("volume=")

    def test_normalisation_target_is_configurable(self):
        assert "loudnorm=I=-30.0" in music.build_filter(10.0, target_lufs=-30.0)

    def test_the_bed_target_is_audible_under_the_voice(self):
        # Chatterbox narration sits near -18 dB mean. The bed must land in the
        # clearly-audible-but-under range, or it vanishes in the mix.
        from src.agents.soundtrack.music import MUSIC_GAIN_DB, MUSIC_TARGET_LUFS

        assert -27 <= MUSIC_TARGET_LUFS <= -20, \
            f"bed target {MUSIC_TARGET_LUFS} LUFS is not audible-under-voice"
        assert MUSIC_GAIN_DB <= 0, "trim must not push the bed over the narration"

    @needs_ffmpeg
    def test_the_bed_is_actually_audible_in_the_mix(self, tmp_path):
        """End-to-end guard: mix the real track under silence and measure it.

        This is the check that would have caught the silent-Oogway bug. A filter
        graph can look correct while producing an inaudible result, so the only
        reliable assertion is the level that actually lands in the file.
        """
        import numpy as np

        # A real (silent) video: the mix maps 0:v, so an audio-only input
        # would fail before the level could be measured.
        silent = tmp_path / "silence.mp4"
        result = subprocess.run([
            "ffmpeg", "-hide_banner",
            "-f", "lavfi", "-i", "color=c=black:s=320x568:d=6:r=24",
            "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo", "-t", "6",
            "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac",
            str(silent), "-y",
        ], capture_output=True)
        assert result.returncode == 0, result.stderr[-300:]

        out = music.add_background_music(str(silent), out_path=str(tmp_path / "bed.mp4"))
        raw = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", out, "-f", "s16le",
             "-ac", "1", "-ar", "44100", "-"],
            capture_output=True, check=True).stdout
        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        rms_db = 20 * np.log10(np.sqrt((samples ** 2).mean()) + 1e-12)
        # Narration sits near -18 dB; the bed should be audible but under it.
        assert rms_db > -30, f"bed is inaudible at {rms_db:.1f} dB"
        assert rms_db < -18, f"bed would compete with narration at {rms_db:.1f} dB"

    def test_narration_branch_is_padded(self):
        # apad stops amix from cutting the voice off at the end of a short bed.
        assert "apad" in music.build_filter(10.0)

    def test_fades_do_not_exceed_the_clip(self):
        # A 5s fade-out on a 1s clip is clamped to what is left after the
        # fade-in, so the two can never overlap.
        graph = music.build_filter(1.0, fade_in_s=0.1, fade_out_s=5.0)
        assert "afade=t=out:st=0.100:d=0.900" in graph

    def test_fade_out_starts_before_the_end(self):
        graph = music.build_filter(20.0, fade_in_s=0.1, fade_out_s=1.5)
        assert "afade=t=out:st=18.500:d=1.500" in graph

    def test_bed_is_already_audible_on_the_first_frame(self):
        """The bed must establish the mood before the first word, not swell in."""
        assert music.MUSIC_FADE_IN_S <= 0.15, music.MUSIC_FADE_IN_S
        graph = music.build_filter(20.0)
        assert "afade=t=in:st=0:d=0.100" in graph

    def test_fade_in_is_shorter_than_fade_out(self):
        assert music.MUSIC_FADE_IN_S < music.MUSIC_FADE_OUT_S

    def test_output_is_labelled_for_mapping(self):
        assert music.build_filter(10.0).endswith("[aout]")


# --- availability ------------------------------------------------------------

class TestMusicAvailable:
    def test_missing_file_is_unavailable(self, tmp_path):
        assert music.music_available(str(tmp_path / "nope.mp3")) is False

    def test_empty_path_is_unavailable(self):
        assert music.music_available("") is False

    def test_shipped_track_is_available(self):
        # The Oogway bed ships with the project and must be usable as-is.
        assert os.path.isfile(MUSIC_PATH)
        assert music.music_available(MUSIC_PATH) is True

    def test_shipped_track_is_long_enough_after_the_skip(self):
        assert music.probe_duration(MUSIC_PATH) > MUSIC_SKIP_S + 5

    def test_the_dead_air_is_cut_from_the_file_not_skipped_at_runtime(self):
        # The first 5s of the track were removed from the file itself, so the
        # runtime skip must stay at 0 — otherwise every render silently eats
        # 5s of music on top of the trim.
        assert MUSIC_SKIP_S == 0, (
            "oogway.mp3 no longer has dead air at the head; a non-zero "
            "MUSIC_SKIP_S would cut into real music"
        )

    def test_the_track_starts_immediately(self):
        # The whole point of the trim: the bed must have real level in its
        # first second, otherwise videos open on silence again.
        result = music._run([
            "ffmpeg", "-hide_banner", "-i", MUSIC_PATH,
            "-af", "atrim=0:1,volumedetect", "-f", "null", "-",
        ])
        mean = next(
            (float(line.split("mean_volume:")[1].split("dB")[0])
             for line in (result.stderr or "").splitlines()
             if "mean_volume:" in line),
            None,
        )
        assert mean is not None, result.stderr[-400:]
        assert mean > -40, f"track still opens on near-silence ({mean:.1f} dB)"


# --- probing -----------------------------------------------------------------

@needs_ffmpeg
class TestProbe:
    def test_duration_of_real_clip(self, silent_clip):
        assert music.probe_duration(silent_clip) == pytest.approx(9.0, abs=0.1)

    def test_mono_clip_reports_one_channel(self, mono_clip):
        # Real TTS output is mono; the mixer must know that.
        assert music.probe_audio_channels(mono_clip) == 1

    def test_unreadable_file_raises(self, tmp_path):
        broken = tmp_path / "broken.mp4"
        broken.write_text("not a video")
        with pytest.raises(RuntimeError):
            music.probe_duration(str(broken))


# --- mixing (real ffmpeg) ----------------------------------------------------

@needs_ffmpeg
class TestAddBackgroundMusic:
    def test_music_is_present_in_output(self, silent_clip, tmp_path):
        out = music.add_background_music(
            silent_clip, out_path=str(tmp_path / "bed.mp4"))
        assert _mean_volume(out) > -70.0  # silence was -91 dB

    def test_gain_is_applied_exactly(self, silent_clip, tmp_path):
        loud = music.add_background_music(
            silent_clip, out_path=str(tmp_path / "loud.mp4"), gain_db=0.0)
        quiet = music.add_background_music(
            silent_clip, out_path=str(tmp_path / "quiet.mp4"), gain_db=-18.0)
        assert _mean_volume(loud) - _mean_volume(quiet) == pytest.approx(18.0, abs=0.5)

    def test_output_duration_matches_input_exactly(self, silent_clip, tmp_path):
        out = music.add_background_music(
            silent_clip, out_path=str(tmp_path / "exact.mp4"))
        assert music.probe_duration(out) == pytest.approx(
            music.probe_duration(silent_clip), abs=0.05)

    def test_narration_level_is_preserved(self, mono_clip, tmp_path):
        # The regression this guards: aformat's mono->stereo upmix cost 3 dB of
        # voice, so the mix was quieter than the narration it was meant to sit
        # under.
        out = music.add_background_music(
            mono_clip, out_path=str(tmp_path / "keep.mp4"), gain_db=-40.0)
        assert _mean_volume(out) == pytest.approx(_mean_volume(mono_clip), abs=0.3)

    def test_mix_does_not_clip(self, silent_clip, tmp_path):
        out = music.add_background_music(
            silent_clip, out_path=str(tmp_path / "lim.mp4"), gain_db=0.0)
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-i", out, "-af", "volumedetect", "-f", "null", "-"],
            capture_output=True, text=True)
        assert float(re.search(r"max_volume:\s*(-?[\d.]+)",
                               result.stderr).group(1)) <= 0.0

    def test_video_stream_is_copied_not_reencoded(self, silent_clip, tmp_path):
        out = music.add_background_music(
            silent_clip, out_path=str(tmp_path / "copy.mp4"))
        before = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=codec_name", "-of", "csv=p=0", silent_clip],
            capture_output=True, text=True).stdout.strip()
        after = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=codec_name", "-of", "csv=p=0", out],
            capture_output=True, text=True).stdout.strip()
        assert before == after

    def test_input_is_left_on_disk(self, silent_clip, tmp_path):
        music.add_background_music(silent_clip, out_path=str(tmp_path / "keep_in.mp4"))
        assert os.path.isfile(silent_clip)

    def test_default_output_sits_beside_the_input(self, silent_clip):
        out = music.add_background_music(silent_clip)
        try:
            assert out.endswith("_music.mp4")
            assert os.path.dirname(out) == os.path.dirname(silent_clip)
        finally:
            if os.path.isfile(out):
                os.remove(out)

    def test_missing_track_is_a_noop(self, silent_clip, tmp_path):
        before = music.probe_duration(silent_clip)
        out = music.add_background_music(
            silent_clip, out_path=str(tmp_path / "x.mp4"),
            music_path=str(tmp_path / "absent.mp3"))
        assert out == silent_clip
        assert music.probe_duration(silent_clip) == pytest.approx(before, abs=0.05)
