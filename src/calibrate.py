import cv2
import json
import sys


video_path = sys.argv[1]

cap = cv2.VideoCapture(video_path)

ret, frame = cap.read()

cap.release()

if not ret:

    print("Failed to load video")

    sys.exit()


corners = []


def click_event(event, x, y, flags, param):

    global frame

    if event == cv2.EVENT_LBUTTONDOWN:

        corners.append([x, y])

        print(
            f"Corner {len(corners)}: "
            f"({x}, {y})"
        )

        cv2.circle(
            frame,
            (x, y),
            8,
            (0, 255, 0),
            -1
        )

        cv2.imshow(
            "Calibration",
            frame
        )

        if len(corners) == 4:

            print("Reached 4 corners")

            try:

                with open(
                    "config.json",
                    "r"
                ) as f:

                    config = json.load(f)

            except:

                config = {}

            config[video_path] = corners

            with open(
                "config.json",
                "w"
            ) as f:

                json.dump(
                    config,
                    f,
                    indent=4
                )

            print("Calibration saved!")

            cv2.waitKey(500)

            cv2.destroyAllWindows()


print(
    "Click corners in order:\n"
    "1. Top-left\n"
    "2. Top-right\n"
    "3. Bottom-right\n"
    "4. Bottom-left"
)

cv2.imshow(
    "Calibration",
    frame
)

cv2.waitKey(1)

cv2.setMouseCallback(
    "Calibration",
    click_event
)

while True:

    if len(corners) == 4:
        break

    key = cv2.waitKey(1) & 0xFF

    if key == 27 or key == ord('q'):
        break


cv2.destroyAllWindows()
