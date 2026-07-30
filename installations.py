import subprocess
import sys


def install_dependencies():
    print("Installing system packages...")

    subprocess.run(
        [
            "apt",
            "update",
            "-y"
        ],
        check=True
    )

    subprocess.run(
        [
            "apt",
            "install",
            "-y",
            "ffmpeg",
        ],
        check=True
    )

    print("Installing Python packages...")

    packages = [
        "google-api-core==2.25.1",
        "google-api-python-client==2.177.0",
        "google-auth==2.40.3",
        "google-auth-httplib2==0.2.0",
        "google-generativeai==0.8.5",
        "googleapis-common-protos==1.70.0",
        "python-dotenv",
        "flask",
        "validators",
        "ultralytics",
        "opencv-fixer==0.2.5",
        "pandas"
    ]

    subprocess.run(
        [sys.executable, "-m", "pip", "install"] + packages,
        check=True
    )

    print("Running OpenCV Fixer...")

    subprocess.run(
        [
            sys.executable,
            "-c",
            "from opencv_fixer import AutoFix; AutoFix()"
        ],
        check=True
    )

    subprocess.run(
        [sys.executable, "-m", "pip", "install", "numpy==1.23.4"],
        check=True
    )

    print("Installation completed successfully.")


install_dependencies()