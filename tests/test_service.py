import io

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from src.service_fastapi import app


def test_classify_endpoint():
    client = TestClient(app)
    image = Image.fromarray((np.random.rand(64, 64, 3) * 255).astype("uint8"))
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)

    response = client.post("/classify", files={"file": ("test.png", buf, "image/png")})
    assert response.status_code == 200
    data = response.json()
    for key in ["label", "prob_defect", "prob_good", "confidence", "model_version"]:
        assert key in data
