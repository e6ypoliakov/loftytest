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
