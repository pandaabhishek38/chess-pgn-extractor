import cv2
import json


CONFIG_FILE = "config.json"


def calibrate_video(video_path):
    """Launch interactive corner calibration tool for a video."""

    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("Failed to load video.")
        return False

    corners = []
    temp_frame = frame.copy()

    def click_event(event, x, y, flags, param):
        nonlocal temp_frame, corners

        if event == cv2.EVENT_LBUTTONDOWN:
            corners.append([x, y])
            print(f"Corner {len(corners)}: ({x}, {y})")
            cv2.circle(temp_frame, (x, y), 8, (0, 255, 0), -1)
            cv2.imshow("Calibration", temp_frame)

    print("\nClick board corners in order:")
    print("  1. Top-left")
    print("  2. Top-right")
    print("  3. Bottom-right")
    print("  4. Bottom-left")
    print("\nPress Q or ESC to cancel.")

    cv2.imshow("Calibration", temp_frame)
    cv2.setMouseCallback("Calibration", click_event)

    while len(corners) < 4:
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            cv2.destroyAllWindows()
            return False

    cv2.destroyAllWindows()

    try:
        with open(CONFIG_FILE, "r") as f:
            config = json.load(f)
    except:
        config = {}

    config[video_path] = corners

    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=4)

    print("\nCalibration saved.")
    return True
