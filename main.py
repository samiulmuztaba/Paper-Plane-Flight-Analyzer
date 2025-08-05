# TODO:
# [DONE] Detect the plane per every model of every flight video
# [DONE] Save the trajectory points to a CSV file for each video
# [IN PROGRESS] Get the distance, airtime and speed of the plane
# [NEXT ONE] Log the stabilty of the plane
# Find the avg trajectory points for each model
# Plot the trajectory points and average trajectory points on a graph, flight test video graphs should low in opacity and the average trajectory points should be more visible


"""Show the final report, show the comparision graphs:
# Distance vs. Angle
# Speed vs. Launch Config
# Stability vs. Mass
"""
# Rank the planes based on their performance

import cv2 as cv
import numpy as np
import os
from colorama import Fore, Style, init

init(autoreset=True)


# ==== Utility Functions =====
def draw_rectangle(img, x, y, w, h):
    glow_color = (0, 255, 255)
    thickness = 2
    length = 20

    # Main rectangle
    corners = [
        ((x, y), (x + length, y), (x, y + length)),
        ((x + w, y), (x + w - length, y), (x + w, y + length)),
        ((x, y + h), (x + length, y + h), (x, y + h - length)),
        ((x + w, y + h), (x + w - length, y + h), (x + w, y + h - length)),
    ]
    for pt1, pt2, pt3 in corners:
        cv.line(img, pt1, pt2, glow_color, thickness)
        cv.line(img, pt1, pt3, glow_color, thickness)

    # Cross
    cv.line(img, (x, y), (x + w, y + h), glow_color, thickness)
    cv.line(img, (x + w, y), (x, y + h), glow_color, thickness)
    center = (x + w // 2, y + h // 2)

    # Circle
    for i in range(6, 0, -1):
        alpha = 0.1 * i
        overlay = img.copy()
        cv.circle(overlay, center, i * 3, glow_color, -1)
        cv.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    cv.circle(img, center, 4, glow_color, -1)


def draw_grid(img, spacing=60, color=(0, 255, 255), thickness=1, alpha=0.1):
    overlay = img.copy()
    h, w = img.shape[:2]
    for x in range(0, w, spacing):
        cv.line(overlay, (x, 0), (x, h), color, thickness)
    for y in range(0, h, spacing):
        cv.line(overlay, (0, y), (w, y), color, thickness)
    cv.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


# ===== MAIN LOGIC STARTS =======
FLIGHTS_DIR = "flights/"


def get_flight_videos(base_dir):
    flight_data = {}  # { 'CD': [video1, video2], 'WG': [...] }

    for model in os.listdir(base_dir):
        model_path = os.path.join(base_dir, model)
        if os.path.isdir(model_path):
            videos = [
                os.path.join(model_path, f)
                for f in os.listdir(model_path)
                if f.endswith(".mp4")
            ]
            if videos:
                flight_data[model] = videos

    return flight_data


# ===== Enhanced CLI Output =====
flight_videos = get_flight_videos(FLIGHTS_DIR)
total_models = len(flight_videos)
print(f"{Fore.CYAN}{'='*50}")
print(f"{Fore.YELLOW}{Style.BRIGHT}Paper Plane Flight Analyzer CLI")
print(f"{Fore.CYAN}{'='*50}\n")

for idx, (model, videos) in enumerate(flight_videos.items(), 1):
    print(f"{Fore.CYAN}{'-'*40}")
    print(f"{Fore.YELLOW}{Style.BRIGHT}✈️  Model [{idx}/{total_models}]: {model}")
    print(f"{Fore.CYAN}{'-'*40}")
    total_videos = len(videos)
    for v_idx, video_path in enumerate(videos, 1):
        print(
            f"{Fore.GREEN}  ▶ Processing [{v_idx}/{total_videos}] {os.path.basename(video_path)}"
        )
        ccap = cv.VideoCapture(video_path)

        if not ccap.isOpened():
            print(f"{Fore.RED}    ⚠️ Can't Find Any Video!")
            continue

        distance = 0
        airtime = 0
        frame_count = 0
        trajectory_points = []
        while True:
            ret, frame = ccap.read()
            if not ret:
                break

            frame = cv.resize(frame, (960, 540))
            hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
            lower, upper = np.array([35, 50, 102]), np.array([179, 255, 255])
            mask = cv.inRange(hsv, lower, upper)

            kernel = np.ones((5, 5), np.uint8)
            mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
            mask = cv.morphologyEx(mask, cv.MORPH_DILATE, kernel)

            contours, _ = cv.findContours(
                mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
            )

            for contour in contours:
                area = cv.contourArea(contour)
                if area > 500:
                    x, y, w, h = cv.boundingRect(contour)
                    center = (x + w // 2, y + h // 2)
                    trajectory_points.append((frame_count, center[0], center[1]))
                    cv.putText(
                        frame,
                        "Plane",
                        (x, y - 10),
                        cv.FONT_HERSHEY_DUPLEX,
                        0.5,
                        (0, 255, 0),
                        2,
                    )
                    draw_rectangle(frame, x, y, w, h)
                    cv.polylines(
                        frame,
                        [np.array([pt[1:] for pt in trajectory_points])],
                        False,
                        (0, 255, 0),
                        2,
                    )
                    draw_grid(frame)
                    cv.putText(
                        frame,
                        f"{center}",
                        (frame.shape[1] - 90, frame.shape[0] - 10),
                        cv.FONT_HERSHEY_DUPLEX,
                        0.4,
                        (0, 255, 255),
                        1,
                    )

            cv.imshow("Flight Video", frame)
            frame_count += 1

            if cv.waitKey(1) & 0xFF == ord("q"):
                print(f"{Fore.YELLOW}    ⏹️ Interrupted by user. Skipping video.")
                break

        ccap.release()
        cv.destroyAllWindows()
        print(
            f"{Fore.BLUE}    📊 Finished Processing {os.path.basename(video_path)} with {frame_count} frames."
        )

        # Save the coordinates of the trajectory points to a csv file (once per video)
        output_dir = f"outputs/{model}"
        os.makedirs(output_dir, exist_ok=True)
        csv_path = os.path.join(
            output_dir, f"{os.path.basename(video_path)[0:-4]}_trajectory.csv"
        )

        with open(csv_path, "w") as f:
            f.write("Frame,X,Y\n")
            for frame_num, x, y in trajectory_points:
                f.write(f"{frame_num},{x},{y}\n")
                distance += np.sqrt(
                    (x - trajectory_points[0][1]) ** 2
                    + (y - trajectory_points[0][2]) ** 2
                )
                airtime += 1 / 30  # Assuming 30 FPS, adjust as necessary
        # Get the distance, airtime and speed of the plane and save it to the text file

        stats_path = os.path.join(
            output_dir, f"{os.path.basename(video_path)[0:-4]}_stats.txt"
        )
        with open(stats_path, "w") as f:
            f.write(f"Distance: {distance:.2f} pixels\n")
            f.write(f"Airtime: {airtime:.2f} seconds\n")
            if airtime > 0:
                speed = distance / airtime
                f.write(f"Speed: {speed:.2f} pixels/second\n")
            else:
                f.write("Speed: N/A (no airtime)\n")

        print(f"{Fore.YELLOW}    📏 Stats saved to text: {stats_path}")
        print(f"{Fore.GREEN}    📄 Saved trajectory points to CSV: {csv_path}")

    print(f"{Fore.MAGENTA}{Style.BRIGHT}✔️ Finished Reading Model: {model}\n")

print(f"{Fore.GREEN}{Style.BRIGHT}✅ All videos processed successfully.")
