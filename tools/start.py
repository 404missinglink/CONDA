import os
import subprocess


def main() -> None:
    index_dir = os.environ.get("INDEX_DIR", os.path.join("artifacts", "index"))
    os.makedirs(index_dir, exist_ok=True)
    env = os.environ.copy()
    env.setdefault("CUSTOM_URL", "https://bsypq4hednykzclslp3aamimtm0ytpzd.lambda-url.eu-west-2.on.aws")
    cmd = [
        "streamlit",
        "run",
        "app/streamlit_app.py",
        "--server.headless=true",
        "--server.port=8501",
    ]
    subprocess.run(cmd, env=env, check=False)


if __name__ == "__main__":
    main()


