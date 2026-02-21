import io
import zipfile
from unittest.mock import patch, MagicMock

from starlette.testclient import TestClient
from tests.conftest import app

client = TestClient(app)


def create_zip_with_files(file_names: list, content: bytes = b"\x00" * 100) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name in file_names:
            zf.writestr(name, content)
    return buf.getvalue()


class TestLoraTrainValidation:
    def test_reject_non_zip_file(self):
        resp = client.post(
            "/train/lora",
            data={"style_name": "test_style"},
            files={"audio_archive": ("audio.tar.gz", b"fake", "application/gzip")},
        )
        assert resp.status_code == 400
        assert "ZIP" in resp.json()["detail"]

    def test_reject_no_filename(self):
        resp = client.post(
            "/train/lora",
            data={"style_name": "test_style"},
            files={"audio_archive": ("", b"fake", "application/zip")},
        )
        assert resp.status_code in (400, 422)

    def test_reject_too_few_audio_files(self):
        zip_data = create_zip_with_files(["track1.wav", "track2.wav", "track3.wav"])
        resp = client.post(
            "/train/lora",
            data={"style_name": "test_style"},
            files={"audio_archive": ("audio.zip", zip_data, "application/zip")},
        )
        assert resp.status_code == 400
        assert "at least 5" in resp.json()["detail"]

    def test_reject_too_many_audio_files(self):
        files = [f"track{i}.wav" for i in range(15)]
        zip_data = create_zip_with_files(files)
        resp = client.post(
            "/train/lora",
            data={"style_name": "test_style"},
            files={"audio_archive": ("audio.zip", zip_data, "application/zip")},
        )
        assert resp.status_code == 400
        assert "Maximum 10" in resp.json()["detail"]

    @patch("api.main.celery_app")
    def test_accept_valid_zip(self, mock_celery):
        files = [f"track{i}.wav" for i in range(7)]
        zip_data = create_zip_with_files(files)
        with patch("tasks.generation_tasks.train_lora_task") as mock_train:
            mock_train.apply_async = MagicMock()
            resp = client.post(
                "/train/lora",
                data={"style_name": "my_lofi"},
                files={"audio_archive": ("tracks.zip", zip_data, "application/zip")},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "pending"
        assert data["style_name"] == "my_lofi"

    @patch("api.main.celery_app")
    def test_accept_5_files_minimum(self, mock_celery):
        files = [f"track{i}.mp3" for i in range(5)]
        zip_data = create_zip_with_files(files)
        with patch("tasks.generation_tasks.train_lora_task") as mock_train:
            mock_train.apply_async = MagicMock()
            resp = client.post(
                "/train/lora",
                data={"style_name": "edge_case"},
                files={"audio_archive": ("min.zip", zip_data, "application/zip")},
            )
        assert resp.status_code == 200

    @patch("api.main.celery_app")
    def test_accept_10_files_maximum(self, mock_celery):
        files = [f"track{i}.flac" for i in range(10)]
        zip_data = create_zip_with_files(files)
        with patch("tasks.generation_tasks.train_lora_task") as mock_train:
            mock_train.apply_async = MagicMock()
            resp = client.post(
                "/train/lora",
                data={"style_name": "edge_max"},
                files={"audio_archive": ("max.zip", zip_data, "application/zip")},
            )
        assert resp.status_code == 200

    def test_reject_invalid_zip_content(self):
        resp = client.post(
            "/train/lora",
            data={"style_name": "bad"},
            files={"audio_archive": ("bad.zip", b"not a zip at all", "application/zip")},
        )
        assert resp.status_code == 400
        assert "Invalid ZIP" in resp.json()["detail"]

    def test_ignore_non_audio_files_in_zip(self):
        files = ["readme.txt", "image.png", "track1.wav", "track2.wav"]
        zip_data = create_zip_with_files(files)
        resp = client.post(
            "/train/lora",
            data={"style_name": "mixed"},
            files={"audio_archive": ("mixed.zip", zip_data, "application/zip")},
        )
        assert resp.status_code == 400
        assert "at least 5" in resp.json()["detail"]

    @patch("api.main.celery_app")
    def test_all_supported_audio_formats(self, mock_celery):
        files = ["song.wav", "song.mp3", "song.flac", "song.ogg", "song.opus"]
        zip_data = create_zip_with_files(files)
        with patch("tasks.generation_tasks.train_lora_task") as mock_train:
            mock_train.apply_async = MagicMock()
            resp = client.post(
                "/train/lora",
                data={"style_name": "all_formats"},
                files={"audio_archive": ("formats.zip", zip_data, "application/zip")},
            )
        assert resp.status_code == 200

    def test_case_insensitive_zip_extension(self):
        zip_data = create_zip_with_files([f"t{i}.wav" for i in range(6)])
        with patch("tasks.generation_tasks.train_lora_task") as mock_train:
            mock_train.apply_async = MagicMock()
            resp = client.post(
                "/train/lora",
                data={"style_name": "upper"},
                files={"audio_archive": ("TRACKS.ZIP", zip_data, "application/zip")},
            )
        assert resp.status_code == 200

    def test_missing_style_name(self):
        resp = client.post(
            "/train/lora",
            files={"audio_archive": ("test.zip", b"data", "application/zip")},
        )
        assert resp.status_code == 422

    def test_nested_audio_in_zip(self):
        files = [f"subdir/track{i}.ogg" for i in range(6)]
        zip_data = create_zip_with_files(files)
        with patch("tasks.generation_tasks.train_lora_task") as mock_train:
            mock_train.apply_async = MagicMock()
            resp = client.post(
                "/train/lora",
                data={"style_name": "nested"},
                files={"audio_archive": ("nested.zip", zip_data, "application/zip")},
            )
        assert resp.status_code == 200


class TestGenerationRequestValidation:
    def test_default_values(self):
        from api.main import GenerationRequest
        req = GenerationRequest(prompt="test beat")
        assert req.duration == 120
        assert req.seed == -1
        assert req.num_steps == 8
        assert req.cfg_scale == 3.5
        assert req.batch_size == 1
        assert req.lyrics is None
        assert req.style is None
        assert req.task_type.value == "text2music"
        assert req.instrumental is False
        assert req.thinking is False
        assert req.audio_format.value == "wav"
        assert req.infer_method.value == "ode"
        assert req.use_adg is False
        assert req.cfg_interval_start == 0.0
        assert req.cfg_interval_end == 1.0
        assert req.shift == 1.0
        assert req.bpm is None
        assert req.keyscale is None
        assert req.timesignature is None
        assert req.vocal_language is None
        assert req.src_audio is None
        assert req.repainting_start is None
        assert req.repainting_end is None
        assert req.audio_cover_strength is None
        assert req.lm_temperature is None
        assert req.lm_top_p is None
        assert req.lm_top_k is None
        assert req.lm_max_tokens is None

    def test_boundary_duration_min(self):
        from api.main import GenerationRequest
        req = GenerationRequest(prompt="test", duration=10)
        assert req.duration == 10

    def test_boundary_duration_max(self):
        from api.main import GenerationRequest
        req = GenerationRequest(prompt="test", duration=600)
        assert req.duration == 600

    def test_boundary_batch_size_min(self):
        from api.main import GenerationRequest
        req = GenerationRequest(prompt="test", batch_size=1)
        assert req.batch_size == 1

    def test_boundary_batch_size_max(self):
        from api.main import GenerationRequest
        req = GenerationRequest(prompt="test", batch_size=8)
        assert req.batch_size == 8

    def test_task_type_cover(self):
        from api.main import GenerationRequest
        req = GenerationRequest(prompt="test", task_type="cover", src_audio="base64data", audio_cover_strength=0.7)
        assert req.task_type.value == "cover"
        assert req.src_audio == "base64data"
        assert req.audio_cover_strength == 0.7

    def test_task_type_repaint(self):
        from api.main import GenerationRequest
        req = GenerationRequest(prompt="test", task_type="repaint", src_audio="data", repainting_start=10.0, repainting_end=20.0)
        assert req.task_type.value == "repaint"
        assert req.repainting_start == 10.0
        assert req.repainting_end == 20.0

    def test_metadata_fields(self):
        from api.main import GenerationRequest
        req = GenerationRequest(prompt="test", bpm=128, keyscale="C major", timesignature="4/4", vocal_language="ru")
        assert req.bpm == 128
        assert req.keyscale == "C major"
        assert req.timesignature == "4/4"
        assert req.vocal_language == "ru"

    def test_bpm_boundaries(self):
        from api.main import GenerationRequest
        import pydantic
        req = GenerationRequest(prompt="test", bpm=40)
        assert req.bpm == 40
        req = GenerationRequest(prompt="test", bpm=300)
        assert req.bpm == 300
        try:
            GenerationRequest(prompt="test", bpm=10)
            assert False, "Should have raised"
        except pydantic.ValidationError:
            pass

    def test_diffusion_advanced_fields(self):
        from api.main import GenerationRequest
        req = GenerationRequest(
            prompt="test",
            use_adg=True,
            cfg_interval_start=0.2,
            cfg_interval_end=0.8,
            shift=1.5,
            infer_method="sde",
        )
        assert req.use_adg is True
        assert req.cfg_interval_start == 0.2
        assert req.cfg_interval_end == 0.8
        assert req.shift == 1.5
        assert req.infer_method.value == "sde"

    def test_lm_fields(self):
        from api.main import GenerationRequest
        req = GenerationRequest(
            prompt="test",
            thinking=True,
            lm_temperature=0.8,
            lm_top_p=0.9,
            lm_top_k=30,
            lm_max_tokens=1024,
        )
        assert req.thinking is True
        assert req.lm_temperature == 0.8
        assert req.lm_top_p == 0.9
        assert req.lm_top_k == 30
        assert req.lm_max_tokens == 1024

    def test_audio_format_options(self):
        from api.main import GenerationRequest
        for fmt in ("wav", "mp3", "flac"):
            req = GenerationRequest(prompt="test", audio_format=fmt)
            assert req.audio_format.value == fmt

    def test_instrumental_flag(self):
        from api.main import GenerationRequest
        req = GenerationRequest(prompt="test", instrumental=True)
        assert req.instrumental is True

    def test_seed_boundaries(self):
        from api.main import GenerationRequest
        import pydantic
        req = GenerationRequest(prompt="test", seed=-1)
        assert req.seed == -1
        req = GenerationRequest(prompt="test", seed=0)
        assert req.seed == 0
        req = GenerationRequest(prompt="test", seed=2147483647)
        assert req.seed == 2147483647
        try:
            GenerationRequest(prompt="test", seed=-2)
            assert False, "Should reject seed < -1"
        except pydantic.ValidationError:
            pass

    def test_shift_boundaries(self):
        from api.main import GenerationRequest
        import pydantic
        req = GenerationRequest(prompt="test", shift=0.1)
        assert req.shift == 0.1
        req = GenerationRequest(prompt="test", shift=10.0)
        assert req.shift == 10.0
        try:
            GenerationRequest(prompt="test", shift=0.0)
            assert False, "Should reject shift < 0.1"
        except pydantic.ValidationError:
            pass

    def test_cfg_scale_boundaries(self):
        from api.main import GenerationRequest
        import pydantic
        req = GenerationRequest(prompt="test", cfg_scale=0.0)
        assert req.cfg_scale == 0.0
        req = GenerationRequest(prompt="test", cfg_scale=15.0)
        assert req.cfg_scale == 15.0
        try:
            GenerationRequest(prompt="test", cfg_scale=16.0)
            assert False, "Should reject cfg_scale > 15"
        except pydantic.ValidationError:
            pass

    def test_num_steps_boundaries(self):
        from api.main import GenerationRequest
        import pydantic
        req = GenerationRequest(prompt="test", num_steps=1)
        assert req.num_steps == 1
        req = GenerationRequest(prompt="test", num_steps=100)
        assert req.num_steps == 100
        try:
            GenerationRequest(prompt="test", num_steps=0)
            assert False, "Should reject num_steps < 1"
        except pydantic.ValidationError:
            pass
        try:
            GenerationRequest(prompt="test", num_steps=101)
            assert False, "Should reject num_steps > 100"
        except pydantic.ValidationError:
            pass

    def test_lm_top_k_boundaries(self):
        from api.main import GenerationRequest
        import pydantic
        req = GenerationRequest(prompt="test", lm_top_k=1)
        assert req.lm_top_k == 1
        req = GenerationRequest(prompt="test", lm_top_k=500)
        assert req.lm_top_k == 500
        try:
            GenerationRequest(prompt="test", lm_top_k=501)
            assert False, "Should reject lm_top_k > 500"
        except pydantic.ValidationError:
            pass

    def test_repainting_end_accepts_minus_one(self):
        from api.main import GenerationRequest
        req = GenerationRequest(prompt="test", repainting_end=-1.0)
        assert req.repainting_end == -1.0

    def test_repainting_start_boundaries(self):
        from api.main import GenerationRequest
        import pydantic
        req = GenerationRequest(prompt="test", repainting_start=0.0)
        assert req.repainting_start == 0.0
        req = GenerationRequest(prompt="test", repainting_start=600.0)
        assert req.repainting_start == 600.0
        try:
            GenerationRequest(prompt="test", repainting_start=-1.0)
            assert False, "Should reject repainting_start < 0"
        except pydantic.ValidationError:
            pass

    def test_lm_max_tokens_boundaries(self):
        from api.main import GenerationRequest
        import pydantic
        req = GenerationRequest(prompt="test", lm_max_tokens=64)
        assert req.lm_max_tokens == 64
        req = GenerationRequest(prompt="test", lm_max_tokens=4096)
        assert req.lm_max_tokens == 4096
        try:
            GenerationRequest(prompt="test", lm_max_tokens=63)
            assert False, "Should reject lm_max_tokens < 64"
        except pydantic.ValidationError:
            pass
        try:
            GenerationRequest(prompt="test", lm_max_tokens=4097)
            assert False, "Should reject lm_max_tokens > 4096"
        except pydantic.ValidationError:
            pass
