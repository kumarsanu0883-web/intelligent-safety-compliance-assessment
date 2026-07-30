import cv2
import torch
import numpy as np
import pandas as pd
import os
from datetime import datetime
import threading
from ultralytics import YOLO
import time
from dotenv import load_dotenv

import smtplib
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart


# Global Variables
is_email_allowed = False  # When user checks the email checkbox, this variable will be set to True
send_next_email = True  # We have to wait for 10 minutes before sending another email

# detections_summary will be used to store the detections summary report
detections_summary = ''

email_sender = 'prajapatisanu431@gmail.com'
email_recipient = 'kumarsanu0883@gmail.com'

load_dotenv("credentials.env")

try:
    SENDER_PASSWORD = os.getenv("sender_password")
except (FileNotFoundError, KeyError):
    print("ERROR: Credentials not found. Please set up your secrets.env file.")


def violation_alert_generator(
    im0,
    subject="PPE Violation Detected",
    message_text="A PPE violation has been detected",
):
    """
    Send email with attached violation image,
    save the violation image locally,
    and prevent another email for 10 minutes.
    """

    global send_next_email
    global email_recipient

    send_next_email = False

    # Save violation image
    try:
        frames_dir = "static/violations"
        os.makedirs(frames_dir, exist_ok=True)

        timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )

        image_path = os.path.join(
            frames_dir,
            f"violation_{timestamp}.jpeg"
        )

        cv2.imwrite(image_path, im0)

        print(
            f"Violation image saved: "
            f"{image_path}"
        )

    except Exception as e:
        print(
            f"Error saving violation image: {e}"
        )

    print(
        f"Sending email alert to "
        f"{email_recipient}"
    )

    try:

        msg = MIMEMultipart()

        msg["From"] = email_sender
        msg["To"] = email_recipient
        msg["Subject"] = subject

        msg.attach(
            MIMEText(
                message_text,
                "plain"
            )
        )

        # Attach saved image
        with open(image_path, "rb") as f:

            image_part = MIMEImage(
                f.read(),
                _subtype="jpeg"
            )

        image_part.add_header(
            "Content-Disposition",
            "attachment",
            filename=os.path.basename(image_path)
        )

        msg.attach(image_part)

        server = smtplib.SMTP_SSL(
            "smtp.gmail.com",
            465
        )

        server.login(
            email_sender,
            SENDER_PASSWORD
        )

        server.sendmail(
            email_sender,
            email_recipient,
            msg.as_string()
        )

        server.close()

        print(
            f"E-mail sent successfully "
            f"to {email_recipient}!"
        )

    except Exception as e:

        print(
            f"An error occurred while "
            f"sending the email: {e}"
        )

    # Cooldown period (10 minutes)
    time.sleep(600)

    send_next_email = True

    print(
        "Email alerts enabled again."
    )


def video_detection(conf_, frames_buffer=[]):
    print("Starting PPE Detection ########################")

    global send_next_email
    global is_email_allowed
    global email_recipient
    global detections_summary

    detections_summary = ""

    torch.cuda.empty_cache()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)

    print("Loading Models ########################")

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    print(BASE_DIR)

    WEIGHTS_DIR = os.path.join(BASE_DIR, "weights")
    print(WEIGHTS_DIR)

    try:
        model = YOLO(
            os.path.join(WEIGHTS_DIR, "best.pt")
        ).to(device)

        print("Model loaded successfully")

    except Exception as e:
        print(f"Error loading model: {e}")
        return

    violation_count = 0

    try:
        while True:

            if len(frames_buffer) == 0:
                time.sleep(0.01)
                continue

            # Keep latest frame only
            while len(frames_buffer) > 1:
                frames_buffer.pop(0)

            img0 = frames_buffer.pop(0)

            if img0 is None:
                continue

            # YOLO inference
            results = model(
                img0,
                conf=conf_,
                verbose=False
            )

            frame_detections = []
            unsafe = False

            for box in results[0].boxes:

                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                confidence = float(box.conf[0])

                frame_detections.append(
                    f"{class_name} {confidence:.4f}"
                )

                # Any class other than safe is unsafe
                if class_name.lower() != "safe":
                    unsafe = True

            # Store report in detections_summary
            if frame_detections:

                timestamp = datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                )

                detections_summary += (
                    f"[{timestamp}] "
                    + ", ".join(frame_detections)
                    + "\n"
                )

            # Email logic (same concept as original code)
            if unsafe:

                violation_count += 1

                print(
                    f"Violation detected. "
                    f"Count = {violation_count}"
                )

                if (
                    violation_count >= 3
                    and is_email_allowed
                    and send_next_email
                ):

                    violation_count = 0

                    try:

                        t = threading.Thread(
                            target=violation_alert_generator,
                            args=(img0.copy(),)
                        )
                        
                        t.daemon = True
                        t.start()

                    except Exception as mail_error:
                        print(
                            f"Email sending failed: "
                            f"{mail_error}"
                        )

            else:
                violation_count = 0

            # Draw detections
            frame = results[0].plot()

            yield frame

    except Exception as e:
        print(f"Error in video_detection: {e}")