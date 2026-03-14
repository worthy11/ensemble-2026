import os

import numpy as np
import requests
from dotenv import load_dotenv

load_dotenv()

ENDPOINT = "task4"

API_TOKEN = os.getenv("TEAM_TOKEN")
SERVER_URL = os.getenv("SERVER_URL")

NPZ_FILE="data/out/ecg_submission.npz"

def main():

    if not API_TOKEN:
        raise ValueError(
            "TEAM_TOKEN not provided. Define TEAM_TOKEN in .env"
        )

    if not SERVER_URL:
        raise ValueError(
            "SERVER_URL not defined. Define SERVER_URL in .env"
        )

    headers = {
        "X-API-Token": API_TOKEN
    }

    response = requests.post(
        f"{SERVER_URL}/{ENDPOINT}",
        files={"npz_file": open(NPZ_FILE, "rb")},
        headers=headers
    )

    try:
        data = response.json()
    except Exception:
        data = response.text

    print("response:", response.status_code, data)

if __name__ == "__main__":
    main()
