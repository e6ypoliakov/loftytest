import os
from unittest.mock import patch, MagicMock

from starlette.testclient import TestClient
from tests.conftest import app

client = TestClient(app)


class TestRootEndpoint:
    def test_root_returns_service_info(self):
        resp = client.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert data["service"] == "ACE-Step Music Generation API"
        assert data["version"] == "1.0.0"
        assert "endpoints" in data
        assert data["endpoints"]["generate"] == "POST /generate"

    def test_root_cors_headers(self):
        resp = client.get("/", headers={"Origin": "http://example.com"})
        assert resp.headers.get("access-control-allow-origin") == "*"


class TestHealthEndpoint:
    def test_health_returns_structure(self):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert "redis_connected" in data
        assert "output_dir" in data
        assert "model_path" in data

    def test_health_redis_disconnected(self):
        resp = client.get("/health")
        data = resp.json()
        assert data["redis_connected"] is False


class TestGenerateEndpoint:
    @patch("api.main.celery_app")
    def test_generate_valid_request(self, mock_celery):
        with patch("tasks.generation_tasks.generate_track") as mock_gen:
            mock_gen.apply_async = MagicMock()
            resp = client.post("/generate", json={
                "prompt": "chill lofi beat",
                "duration": 60,
            })
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "pending"
        assert "task_id" in data

    def test_generate_missing_prompt(self):
        resp = client.post("/generate", json={})
        assert resp.status_code == 422

    def test_generate_invalid_duration_too_low(self):
        resp = client.post("/generate", json={"prompt": "test", "duration": 5})
        assert resp.status_code == 422

    def test_generate_invalid_duration_too_high(self):
        resp = client.post("/generate", json={"prompt": "test", "duration": 700})
        assert resp.status_code == 422

    def test_generate_invalid_batch_size(self):
        resp = client.post("/generate", json={"prompt": "test", "batch_size": 20})
        assert resp.status_code == 422

    @patch("api.main.celery_app")
    def test_generate_with_all_optional_fields(self, mock_celery):
        with patch("tasks.generation_tasks.generate_track") as mock_gen:
            mock_gen.apply_async = MagicMock()
            resp = client.post("/generate", json={
                "prompt": "epic orchestral",
                "duration": 180,
                "lyrics": "[verse]\nHello world\n[chorus]\nSing along",
                "style": "orchestral epic cinematic",
                "seed": 42,
                "num_steps": 16,
                "cfg_scale": 5.0,
                "batch_size": 3,
                "lora_id": "my_style",
            })
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "pending"


class TestStatusEndpoint:
    @patch("api.main.celery_app")
    def test_status_pending(self, mock_celery):
        with patch("celery.result.AsyncResult") as mock_result_cls:
            mock_result = MagicMock()
            mock_result.state = "PENDING"
            mock_result_cls.return_value = mock_result
            resp = client.get("/status/some-task-id")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "pending"
        assert data["task_id"] == "some-task-id"

    @patch("api.main.celery_app")
    def test_status_success_with_file(self, mock_celery):
        with patch("celery.result.AsyncResult") as mock_result_cls:
            mock_result = MagicMock()
            mock_result.state = "SUCCESS"
            mock_result.result = {
                "status": "success",
                "file_path": "/output/song_123.wav",
            }
            mock_result_cls.return_value = mock_result
            resp = client.get("/status/task-123")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"
        assert data["file_url"] == "/files/song_123.wav"

    @patch("api.main.celery_app")
    def test_status_failure(self, mock_celery):
        with patch("celery.result.AsyncResult") as mock_result_cls:
            mock_result = MagicMock()
            mock_result.state = "FAILURE"
            mock_result.result = Exception("GPU out of memory")
            mock_result_cls.return_value = mock_result
            resp = client.get("/status/task-fail")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "failed"
        assert "GPU out of memory" in data["error"]

    @patch("api.main.celery_app")
    def test_status_progress(self, mock_celery):
        with patch("celery.result.AsyncResult") as mock_result_cls:
            mock_result = MagicMock()
            mock_result.state = "PROGRESS"
            mock_result_cls.return_value = mock_result
            resp = client.get("/status/task-prog")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "processing"


class TestFilesEndpoint:
    def test_file_not_found(self):
        resp = client.get("/files/nonexistent.wav")
        assert resp.status_code == 404

    def test_path_traversal_blocked(self):
        resp = client.get("/files/..%2F..%2Fetc%2Fpasswd")
        assert resp.status_code in (403, 404)

    def test_dotdot_in_filename_blocked(self):
        resp = client.get("/files/..%2Fsecret.txt")
        assert resp.status_code in (403, 404)

    def test_serve_existing_file(self):
        from core.config import settings
        test_file = os.path.join(settings.OUTPUT_DIR, "test_serve.wav")
        os.makedirs(settings.OUTPUT_DIR, exist_ok=True)
        with open(test_file, "wb") as f:
            f.write(b"RIFF" + b"\x00" * 100)
        try:
            resp = client.get("/files/test_serve.wav")
            assert resp.status_code == 200
            assert "audio" in resp.headers.get("content-type", "")
            assert resp.headers.get("cache-control") == "no-cache"
        finally:
            os.remove(test_file)

    def test_media_type_mp3(self):
        from core.config import settings
        test_file = os.path.join(settings.OUTPUT_DIR, "test_serve.mp3")
        os.makedirs(settings.OUTPUT_DIR, exist_ok=True)
        with open(test_file, "wb") as f:
            f.write(b"\xff\xfb" + b"\x00" * 100)
        try:
            resp = client.get("/files/test_serve.mp3")
            assert resp.status_code == 200
            assert "audio/mpeg" in resp.headers.get("content-type", "")
        finally:
            os.remove(test_file)
