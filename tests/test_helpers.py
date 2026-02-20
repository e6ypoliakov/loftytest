import os
import tempfile
import shutil
import zipfile

from tasks.generation_tasks import _find_lora_train_tmp_dir
from api.main import _extract_audio_from_zip, AUDIO_EXTENSIONS
from core.config import Settings


class TestExtractAudioFromZip:
    def test_extracts_wav_files(self):
        tmp = tempfile.mkdtemp()
        try:
            zip_buf = self._make_zip(["a.wav", "b.wav", "c.mp3"])
            result = _extract_audio_from_zip(zip_buf, tmp)
            extensions = {os.path.splitext(f)[1].lower() for f in result}
            assert extensions <= AUDIO_EXTENSIONS
            assert len(result) == 3
        finally:
            shutil.rmtree(tmp)

    def test_ignores_non_audio(self):
        tmp = tempfile.mkdtemp()
        try:
            zip_buf = self._make_zip(["readme.txt", "song.wav", "image.jpg"])
            result = _extract_audio_from_zip(zip_buf, tmp)
            assert len(result) == 1
            assert result[0].endswith(".wav")
        finally:
            shutil.rmtree(tmp)

    def test_handles_nested_directories(self):
        tmp = tempfile.mkdtemp()
        try:
            zip_buf = self._make_zip(["dir1/song.flac", "dir2/beat.ogg"])
            result = _extract_audio_from_zip(zip_buf, tmp)
            assert len(result) == 2
        finally:
            shutil.rmtree(tmp)

    def test_all_supported_formats(self):
        tmp = tempfile.mkdtemp()
        try:
            files = ["a.mp3", "b.wav", "c.flac", "d.ogg", "e.opus"]
            zip_buf = self._make_zip(files)
            result = _extract_audio_from_zip(zip_buf, tmp)
            assert len(result) == 5
        finally:
            shutil.rmtree(tmp)

    def test_empty_zip(self):
        tmp = tempfile.mkdtemp()
        try:
            zip_buf = self._make_zip([])
            result = _extract_audio_from_zip(zip_buf, tmp)
            assert len(result) == 0
        finally:
            shutil.rmtree(tmp)

    def test_m4a_not_supported(self):
        tmp = tempfile.mkdtemp()
        try:
            zip_buf = self._make_zip(["song.m4a", "track.aac", "file.wma"])
            result = _extract_audio_from_zip(zip_buf, tmp)
            assert len(result) == 0
        finally:
            shutil.rmtree(tmp)

    def test_case_insensitive_extensions(self):
        tmp = tempfile.mkdtemp()
        try:
            zip_buf = self._make_zip(["SONG.WAV", "Track.MP3", "Beat.FLAC"])
            result = _extract_audio_from_zip(zip_buf, tmp)
            assert len(result) == 3
        finally:
            shutil.rmtree(tmp)

    def _make_zip(self, names):
        import io
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            for n in names:
                zf.writestr(n, b"\x00" * 50)
        return buf.getvalue()


class TestFindLoraTmpDir:
    def test_finds_direct_parent(self):
        paths = ["/tmp/lora_train_abc123/song.wav"]
        result = _find_lora_train_tmp_dir(paths)
        assert result == "/tmp/lora_train_abc123"

    def test_finds_grandparent(self):
        paths = ["/tmp/lora_train_abc123/subdir/song.wav"]
        result = _find_lora_train_tmp_dir(paths)
        assert result == "/tmp/lora_train_abc123"

    def test_returns_none_for_non_tmp(self):
        paths = ["/home/user/music/song.wav"]
        result = _find_lora_train_tmp_dir(paths)
        assert result is None

    def test_returns_none_for_empty(self):
        result = _find_lora_train_tmp_dir([])
        assert result is None

    def test_finds_first_match(self):
        paths = [
            "/home/user/song.wav",
            "/tmp/lora_train_xyz/track.wav",
        ]
        result = _find_lora_train_tmp_dir(paths)
        assert result == "/tmp/lora_train_xyz"


class TestSettings:
    def test_default_values(self):
        s = Settings()
        assert s.REDIS_URL == "redis://localhost:6379/0"
        assert s.MODEL_PATH == "acestep-v15-turbo"
        assert s.OUTPUT_DIR == "generated_audio"
        assert s.HF_TOKEN is None
        assert s.LORA_DIR == "lora_models"
        assert s.ACESTEP_INIT_LLM is False

    def test_env_override(self):
        os.environ["MODEL_PATH"] = "custom-model"
        try:
            s = Settings()
            assert s.MODEL_PATH == "custom-model"
        finally:
            del os.environ["MODEL_PATH"]

    def test_audio_extensions_constant(self):
        assert ".wav" in AUDIO_EXTENSIONS
        assert ".mp3" in AUDIO_EXTENSIONS
        assert ".flac" in AUDIO_EXTENSIONS
        assert ".ogg" in AUDIO_EXTENSIONS
        assert ".opus" in AUDIO_EXTENSIONS
        assert ".m4a" not in AUDIO_EXTENSIONS
        assert ".aac" not in AUDIO_EXTENSIONS
