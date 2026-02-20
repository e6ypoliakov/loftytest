import os
import sys
from unittest.mock import MagicMock, patch

os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")

celery_mock = MagicMock()
celery_mock.conf = MagicMock()
celery_mock.conf.update = MagicMock()

with patch("core.celery_app.celery_app", celery_mock), \
     patch("core.celery_app.Celery", return_value=celery_mock):
    sys.modules.pop("core.celery_app", None)
    sys.modules.pop("api.main", None)

import importlib
import core.celery_app
core.celery_app.celery_app = celery_mock
importlib.reload(core.celery_app)

from api.main import app

import pytest
from starlette.testclient import TestClient


@pytest.fixture
def client():
    return TestClient(app)
