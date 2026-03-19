from locust import HttpUser, task, between
import random
import os

# Run with: locust -f locustfile.py --host=http://localhost:8000
# Then open http://localhost:8089 to control the test

class CoffeeGuardUser(HttpUser):
    wait_time = between(1, 3)

    @task(3)
    def predict(self):
        folders = [
            "data/test/Healthy",
            "data/test/Rust",
            "data/test/Miner",
            "data/test/Cercospora",
            "data/test/Phoma",
        ]

        folder = random.choice(folders)

        if os.path.exists(folder):
            files = os.listdir(folder)
            if files:
                image_path = os.path.join(folder, files[0])
                with open(image_path, "rb") as f:
                    self.client.post(
                        "/predict",
                        files={"file": ("leaf.jpg", f, "image/jpeg")},
                    )
        else:
            self.client.post(
                "/predict",
                files={"file": ("leaf.jpg", b"dummy", "image/jpeg")},
            )

    @task(1)
    def health_check(self):
        self.client.get("/health")

    @task(1)
    def get_metrics(self):
        self.client.get("/metrics")
