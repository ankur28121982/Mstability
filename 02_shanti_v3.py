import csv
import os
import time

import cv2
import numpy as np
import mediapipe as mp

import tkinter as tk
from tkinter import simpledialog, messagebox

# Try to import Streamlit (for web UI)
try:
    import streamlit as st
except ImportError:
    st = None

# Initialize MediaPipe Face Mesh for detecting facial landmarks
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# Blink detection threshold
BLINK_THRESHOLD = 0.25


def calculate_ear(eye):
    """
    Calculate Eye Aspect Ratio (EAR) for blink detection.
    """
    vertical1 = np.linalg.norm(
        np.array([eye[1][0], eye[1][1]]) - np.array([eye[5][0], eye[5][1]])
    )
    vertical2 = np.linalg.norm(
        np.array([eye[2][0], eye[2][1]]) - np.array([eye[4][0], eye[4][1]])
    )
    horizontal = np.linalg.norm(
        np.array([eye[0][0], eye[0][1]]) - np.array([eye[3][0], eye[3][1]])
    )

    # Avoid division by zero
    if horizontal == 0:
        return 0.0
    return (vertical1 + vertical2) / (2.0 * horizontal)


def calculate_pupil_dilation(eye):
    """
    Approximate pupil dilation using bounding box area of eye landmarks.
    """
    eye_points = np.array(eye)
    if len(eye_points) > 0:
        x, y, w, h = cv2.boundingRect(eye_points)
        return float(w * h)
    return 0.0


def get_user_input():
    """
    Original Tkinter-based input dialog (kept for compatibility / offline use).
    """
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    name = simpledialog.askstring("Input", "Enter your name:")
    phone = simpledialog.askstring("Input", "Enter your phone number:")
    root.destroy()
    return name, phone


def run_for_user(
    user_name,
    user_phone,
    total_time=60,
    show_popup=True,
    use_streamlit=False,
    st_frame_placeholder=None,
    st_status_placeholder=None,
    st_progress_bar=None,
):
    """
    Run calmness session for a single user.

    Args:
        user_name (str): participant name
        user_phone (str): participant phone
        total_time (int): session duration in seconds
        show_popup (bool): whether to show Tkinter popup at end
        use_streamlit (bool): if True, show video inside Streamlit instead of cv2 window
        st_frame_placeholder: streamlit empty() placeholder for frame
        st_status_placeholder: streamlit placeholder for text status
        st_progress_bar: streamlit progress bar object

    Returns:
        float or None: final calmness score (0–100) or None if not computed
    """
    user_name = (user_name or "").strip()
    user_phone = (user_phone or "").strip()

    # --- Basic validation ---
    if not user_name:
        raise ValueError("Name is required.")
    if not user_phone:
        raise ValueError("Phone number is required.")
    if not any(ch.isdigit() for ch in user_phone):
        raise ValueError("Phone number must contain at least one digit.")

    # --- Camera setup ---
    try:
        cap = cv2.VideoCapture(0)
    except Exception as e:
        raise RuntimeError(f"Could not access camera: {e}")

    if not cap.isOpened():
        cap.release()
        raise RuntimeError(
            "Camera is not available. Please check connection and permissions."
        )

    blink_count = 0
    frame_counter = 0
    pupil_dilations = []
    calmness_scores = []

    start_time = time.time()
    last_logged_second = -1  # to ensure we log once per second

    # Prepare CSV log file (per-second logs)
    log_file_path = "calmness_log.csv"
    log_exists = os.path.isfile(log_file_path)
    try:
        log_file = open(log_file_path, "a", newline="", encoding="utf-8")
    except Exception as e:
        cap.release()
        cv2.destroyAllWindows()
        raise RuntimeError(f"Could not open log file: {e}")

    with log_file:
        log_writer = csv.writer(log_file)
        if not log_exists:
            log_writer.writerow(
                [
                    "Name",
                    "Phone",
                    "Time (sec)",
                    "Blink Rate (blinks/min)",
                    "Pupil Dilation",
                    "Calmness Score",
                ]
            )

        # --- Main loop ---
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                cap.release()
                cv2.destroyAllWindows()
                raise RuntimeError("Failed to read from camera.")

            # Mirror the frame so participant sees themselves naturally
            frame = cv2.flip(frame, 1)

            height, width, _ = frame.shape

            # Convert to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)

            ear = None
            avg_pupil_dilation_value = 0.0

            if results.multi_face_landmarks:
                # Use first detected face
                face_landmarks = results.multi_face_landmarks[0]
                landmarks = [
                    (int(lm.x * width), int(lm.y * height))
                    for lm in face_landmarks.landmark
                ]

                # Indices for left and right eyes (MediaPipe landmark indices)
                left_eye_indices = [362, 385, 387, 263, 373, 380]
                right_eye_indices = [33, 160, 158, 133, 153, 144]

                left_eye = [landmarks[i] for i in left_eye_indices]
                right_eye = [landmarks[i] for i in right_eye_indices]

                # Calculate EAR for both eyes
                left_ear = calculate_ear(left_eye)
                right_ear = calculate_ear(right_eye)
                ear = (left_ear + right_ear) / 2.0

                # Blink detection
                if ear < BLINK_THRESHOLD:
                    frame_counter += 1
                else:
                    if frame_counter > 2:
                        blink_count += 1
                    frame_counter = 0

                # Pupil dilation estimate
                left_pupil_area = calculate_pupil_dilation(left_eye)
                right_pupil_area = calculate_pupil_dilation(right_eye)
                if left_pupil_area or right_pupil_area:
                    avg_pupil_dilation_value = (left_pupil_area + right_pupil_area) / 2.0

            elapsed_time = time.time() - start_time
            current_second = int(elapsed_time)

            # Log once per second
            if current_second != last_logged_second and elapsed_time > 0:
                last_logged_second = current_second

                blink_rate_per_minute = blink_count * (60.0 / elapsed_time)

                pupil_dilations.append(avg_pupil_dilation_value)

                # Contributions (tunable)
                blink_rate_contribution = min(blink_rate_per_minute / 10.0, 100.0)
                pupil_dilation_contribution = min(
                    (avg_pupil_dilation_value / 10000.0) * 100.0, 100.0
                )
                if ear is None:
                    ear_contribution = 0.0
                else:
                    ear_contribution = max(
                        0.0,
                        min((BLINK_THRESHOLD - ear) * 100.0 / BLINK_THRESHOLD, 100.0),
                    )

                calmness_score_value = max(
                    40.0,
                    min(
                        100.0,
                        100.0
                        - (
                            blink_rate_contribution
                            + pupil_dilation_contribution
                            + ear_contribution
                        ),
                    ),
                )

                calmness_scores.append(calmness_score_value)

                # Write row to CSV log
                log_writer.writerow(
                    [
                        user_name,
                        user_phone,
                        current_second,
                        round(blink_rate_per_minute, 2),
                        round(avg_pupil_dilation_value, 2),
                        round(calmness_score_value, 2),
                    ]
                )

            # --- On-screen overlays ---
            cv2.putText(
                frame,
                f"Blinks: {blink_count}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                frame,
                f"EAR: {ear:.2f}" if ear is not None else "EAR: N/A",
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                frame,
                f"Pupil Dilation: {avg_pupil_dilation_value:.2f}",
                (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

            stopwatch_time = int(elapsed_time)
            minutes_stopwatch = stopwatch_time // 60
            seconds_stopwatch = stopwatch_time % 60
            stopwatch_text = f"{minutes_stopwatch:02}:{seconds_stopwatch:02}"

            cv2.putText(
                frame,
                f"Stopwatch: {stopwatch_text}",
                (frame.shape[1] - 280, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
            )

            # --- Show frame either in Streamlit or in a normal cv2 window ---
            if use_streamlit and st_frame_placeholder is not None:
                # Show inside browser
                st_frame_placeholder.image(frame, channels="BGR")
                if st_status_placeholder is not None:
                    st_status_placeholder.write(
                        f"⏱ Time: {stopwatch_text} | Blinks: {blink_count}"
                    )
                if st_progress_bar is not None:
                    progress_value = min(elapsed_time / float(total_time), 1.0)
                    st_progress_bar.progress(progress_value)
            else:
                # Fallback to normal OpenCV window
                cv2.imshow("Blink Rate and Pupil Dilation", frame)

            # Stop conditions
            if elapsed_time >= total_time:
                break
            # Only check 'q' if using cv2 window
            if (not use_streamlit) and (cv2.waitKey(1) & 0xFF == ord("q")):
                break

    cap.release()
    cv2.destroyAllWindows()

    if not calmness_scores:
        # No face detected or no scores
        if show_popup:
            messagebox.showwarning(
                "Anadi Kaal Shanti Awastha",
                "Face not detected properly. Please try again.",
            )
        return None

    avg_calmness_score = sum(calmness_scores) / len(calmness_scores)
    final_calmness_score = max(avg_calmness_score, 40.0)

    # Save final score to separate CSV
    final_score_log_path = "final_calmness_scores.csv"
    score_log_exists = os.path.isfile(final_score_log_path)
    try:
        with open(final_score_log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not score_log_exists:
                writer.writerow(["Name", "Phone", "Final Calmness Score (%)"])
            writer.writerow(
                [
                    user_name,
                    user_phone,
                    round(final_calmness_score, 2),
                ]
            )
    except Exception as e:
        if show_popup:
            messagebox.showwarning(
                "Warning", f"Could not save final score to CSV: {e}"
            )

    if show_popup:
        messagebox.showinfo(
            "Anadi Kaal Shanti Awastha",
            f"Apki Shanti ki awastha: {final_calmness_score:.2f}%",
        )

    return final_calmness_score


def main_cli():
    """
    Optional: original CLI / Tkinter flow.
    """
    while True:
        name, phone = get_user_input()
        if not name or not phone:
            break
        try:
            run_for_user(name, phone, total_time=60, show_popup=True)
        except Exception as e:
            messagebox.showerror("Error", str(e))
        if not messagebox.askyesno(
            "Another User?", "Do you want to start for another user?"
        ):
            break


def main_streamlit():
    """
    Streamlit UI so that even a child can operate it easily,
    with live camera preview on the page.
    """
    if st is None:
        print(
            "Streamlit is not installed. "
            "Install it with: pip install streamlit"
        )
        return

    st.set_page_config(
        page_title="Calmness Meter",
        page_icon="🧘",
        layout="centered",
    )

    st.title("🧘 Anadi Kaal Shanti Awastha – Calmness Meter")

    st.markdown(
        """
        **How to use:**
        1. Type your **name**.
        2. Type your **phone number**.
        3. Click **Start Calmness Test**.
        4. Look at your face in the video box and sit quietly.
        5. When the timer finishes, your **calmness score** will appear below.
        """
    )

    with st.form("user_form"):
        name = st.text_input("Your Name")
        phone = st.text_input("Your Phone Number")
        duration = st.slider(
            "Test duration (seconds)",
            min_value=30,
            max_value=120,
            value=60,
            step=10,
            help="How long each person sits in front of the camera.",
        )
        submitted = st.form_submit_button("Start Calmness Test")

    # Placeholders for live video and status
    frame_placeholder = st.empty()
    status_placeholder = st.empty()
    progress_bar = st.progress(0.0)

    if submitted:
        # Basic validation in UI
        errors = []
        if not name.strip():
            errors.append("Please enter your name.")
        if not phone.strip():
            errors.append("Please enter your phone number.")
        elif not any(ch.isdigit() for ch in phone):
            errors.append("Phone number must contain at least one digit.")

        if errors:
            for e in errors:
                st.error(e)
            return

        try:
            status_placeholder.info(
                "Starting camera... If browser asks, please allow camera access."
            )
            score = run_for_user(
                name.strip(),
                phone.strip(),
                total_time=duration,
                show_popup=False,           # No Tk popups in web mode
                use_streamlit=True,         # Show video inside page
                st_frame_placeholder=frame_placeholder,
                st_status_placeholder=status_placeholder,
                st_progress_bar=progress_bar,
            )
        except Exception as e:
            status_placeholder.error(f"Something went wrong: {e}")
            return

        if score is None:
            status_placeholder.warning(
                "Face was not detected properly. "
                "Please check lighting and camera position and try again."
            )
        else:
            status_placeholder.success("Calmness test finished! ✅")
            st.metric("Your Calmness Score", f"{score:.1f} %")
            st.info("You can now invite the next participant to take the test.")


# If streamlit is available, try to start the web UI.
if st is not None:
    try:
        main_streamlit()
    except Exception as e:
        # Probably not running under 'streamlit run'
        if __name__ == "__main__":
            print(
                "Streamlit UI could not start ('%s').\n"
                "To use the web UI, run: streamlit run calmness_app.py" % e
            )
            try:
                main_cli()
            except Exception as inner_e:
                print(f"CLI mode also failed: {inner_e}")

# If streamlit is NOT installed and script is executed directly,
# fall back to the old Tkinter-based flow:
if __name__ == "__main__" and st is None:
    main_cli()
