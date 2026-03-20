import os
import random
from locust import HttpUser, task, between

# Find a real test image to use
TEST_IMAGE_PATH = None
test_dirs = [
    os.path.expanduser("~/MLOPS/data/test/Healthy"),
    os.path.expanduser("~/MLOPS/data/test/Miner"),
]
for d in test_dirs:
    if os.path.exists(d):
        files = [f for f in os.listdir(d) if f.endswith(".jpg")]
        if files:
            TEST_IMAGE_PATH = os.path.join(d, files[0])
            break

if TEST_IMAGE_PATH and os.path.exists(TEST_IMAGE_PATH):
    with open(TEST_IMAGE_PATH, "rb") as f:
        TEST_IMAGE_BYTES = f.read()
    print(f"Using test image: {TEST_IMAGE_PATH}")
else:
    TEST_IMAGE_BYTES = None
    print("No test image found")

class CoffeeGuardUser(HttpUser):
    wait_time = between(1, 3)

    @task(3)
    def predict(self):
        if TEST_IMAGE_BYTES:
            self.client.post(
                "/predict",
                files={"file": ("leaf.jpg", TEST_IMAGE_BYTES, "image/jpeg")},
            )

    @task(1)
    def health_check(self):
        self.client.get("/health")

    @task(1)
    def get_metrics(self):
        self.client.get("/metrics")
