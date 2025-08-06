# TODO:
# [DONE] Detect the plane per every model of every flight video
# [DONE] Save the trajectory points to a CSV file for each video
# [DONE] Get the distance, airtime and speed of the plane {Remember to set the ref_pixels and ref_m values for the distance calculation}
# [DONE] Take input from the user for the stabilty of the plane
# [DONE] Find the avg trajectory points for each model
# [DONE] Plot the trajectory points and average trajectory points on a graph, flight test video graphs should low in opacity and the average trajectory points should be more visible
# Show the final report, show the comparision graphs:
# Distance vs. Angle
# Speed vs. Model
# Stability vs. Mass

# Rank the planes based on their performance

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
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
print(f"{Fore.CYAN}{'='*50}")

for idx, (model, videos) in enumerate(flight_videos.items(), 1):
    print(f"\n{Fore.CYAN}{'='*50}")
    print(f"{Fore.YELLOW}{Style.BRIGHT}✈️  Model [{idx}/{total_models}]: {model}")
    print(f"{Fore.CYAN}{'='*50}")
    total_videos = len(videos)
    for v_idx, video_path in enumerate(videos, 1):
        percent_model = (v_idx / total_videos) * 100 if total_videos else 100
        print(f"{Fore.GREEN}▶ [{v_idx}/{total_videos}] {os.path.basename(video_path)} | {percent_model:.1f}% of model videos")
        ccap = cv.VideoCapture(video_path)

        if not ccap.isOpened():
            print(f"{Fore.RED}  ⚠️ Can't Find Any Video!")
            continue

        ref_pixels = 0
        ref_m = 0
        pm = ref_pixels / ref_m if ref_m != 0 else 1  # Avoid division by zero
        distance_px = 0
        distance = distance_px / pm if pm != 0 else 1  # Avoid division by zero
        airtime = 0
        frame_count = 0
        trajectory_points = []
        total_frames = int(ccap.get(cv.CAP_PROP_FRAME_COUNT))
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

            percent_video = ((frame_count + 1) / total_frames) * 100 if total_frames else 100
            print(f"{Fore.CYAN}    Frame {frame_count + 1}/{total_frames} ({percent_video:.1f}%)", end='\r')

            cv.imshow("Flight Video", frame)
            frame_count += 1

            if cv.waitKey(1) & 0xFF == ord("q"):
                print(f"{Fore.YELLOW}    ⏹️ Interrupted by user. Skipping video.")
                break

        print()  # For clean progress output
        ccap.release()
        cv.destroyAllWindows()
        print(
            f"{Fore.BLUE}    📊 Finished Processing {os.path.basename(video_path)} with {frame_count} frames."
        )

        print(f"{Fore.CYAN}{Style.BRIGHT}  📝 Enter stability score for {model} (1-10): ", end='')
        stabilty_score = input().strip()

        # Save the coordinates of the trajectory points to a csv file (once per video)
        output_dir = f"outputs/{model}"
        coordinates_dir = os.path.join(output_dir, 'coordinates')
        stats_dir = os.path.join(output_dir, 'stats')
        os.makedirs(coordinates_dir, exist_ok=True)
        os.makedirs(stats_dir, exist_ok=True)
        csv_path = os.path.join(coordinates_dir, f"{os.path.basename(video_path)[0:-4]}_trajectory.csv")

        with open(csv_path, "w") as f:
            f.write("Frame,X,Y\n")
            for frame_num, x, y in trajectory_points:
                f.write(f"{frame_num},{x},{y}\n")
                distance_px += np.sqrt(
                    (x - trajectory_points[0][1]) ** 2
                    + (y - trajectory_points[0][2]) ** 2
                )
                airtime += 1 / 30  # Assuming 30 FPS, adjust as necessary

        stats_path = os.path.join(stats_dir, f"{os.path.basename(video_path)[0:-4]}_stats.csv")
        with open(stats_path, "w") as f:
            f.write("Distance,Airtime,Speed,Stability\n")
            if airtime > 0:
                speed = distance_px / airtime
                f.write(f"{distance_px:.2f},{airtime:.2f},{speed:.2f},{stabilty_score}\n")
            else:
                f.write("0,0,0,0\n")

        print(f"{Fore.YELLOW}  📏 Stats saved: {stats_path}")
        print(f"{Fore.GREEN}  📄 Trajectory CSV saved: {csv_path}")

    # Calculate the average trajectory points for the model
    from collections import defaultdict

    frame_sums = defaultdict(lambda: [0, 0, 0])  # frame: [count, sum_x, sum_y]
    max_frame = 0

    for file in os.listdir(coordinates_dir):
        if file.endswith("_trajectory.csv"):
            file_path = os.path.join(coordinates_dir, file)
            with open(file_path, "r") as f:
                lines = f.readlines()[1:]
            for line in lines:
                frame, x, y = map(float, line.strip().split(","))
                frame = int(frame)
                frame_sums[frame][0] += 1
                frame_sums[frame][1] += x
                frame_sums[frame][2] += y
                if frame > max_frame:
                    max_frame = frame

    avg_trajectory = []
    for frame in range(max_frame + 1):
        count, sum_x, sum_y = frame_sums[frame]
        if count > 0:
            avg_x = sum_x / count
            avg_y = sum_y / count
            avg_trajectory.append((frame, avg_x, avg_y))

    # Save average trajectory CSV
    avg_csv_path = os.path.join(coordinates_dir, f"avg_trajectory.csv")
    with open(avg_csv_path, "w") as f:
        f.write("Frame,X,Y\n")
        for frame, x, y in avg_trajectory:
            f.write(f"{frame},{x},{y}\n")

    # Calculate stats for average trajectory
    avg_distance_px = 0
    avg_airtime = 0
    for i in range(1, len(avg_trajectory)):
        x0, y0 = avg_trajectory[i-1][1], avg_trajectory[i-1][2]
        x1, y1 = avg_trajectory[i][1], avg_trajectory[i][2]
        avg_distance_px += np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)
        avg_airtime += 1 / 30  # Assuming 30 FPS
    avg_speed = avg_distance_px / avg_airtime if avg_airtime > 0 else 0

    # Compute average stability score from all flights
    stability_scores = []
    for file in os.listdir(stats_dir):
        if file.endswith("_stats.csv"):
            file_path = os.path.join(stats_dir, file)
            with open(file_path, "r") as f:
                lines = f.readlines()[1:]
                for line in lines:
                    parts = line.strip().split(",")
                    if len(parts) == 4:
                        try:
                            stability_scores.append(float(parts[3]))
                        except Exception:
                            pass
    avg_stability = sum(stability_scores) / len(stability_scores) if stability_scores else 0

    avg_stats_path = os.path.join(stats_dir, "avg_stats.csv")
    with open(avg_stats_path, "w") as f:
        f.write("Distance,Airtime,Speed,Stability\n")
        f.write(f"{avg_distance_px:.2f},{avg_airtime:.2f},{avg_speed:.2f},{avg_stability:.2f}\n")

    print(f"{Fore.YELLOW}{Style.BRIGHT}  📏 Avg stats saved: {avg_stats_path}")
    print(f"{Fore.CYAN}{Style.BRIGHT}  📊 Avg trajectory points calculated for {model}")
    percent_all = (idx / total_models) * 100 if total_models else 100
    print(f"{Fore.MAGENTA}{Style.BRIGHT}✔️ Finished Model: {model} | {percent_all:.1f}% of all models")

    # Plot the trajectory points and average trajectory points on a graph
    plt.figure(figsize=(10, 6))
    plt.title(f"Trajectory Points for {model}", fontsize=16, color='navy')
    plt.xlabel("X Coordinate", fontsize=12)
    plt.ylabel("Y Coordinate", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.3)

    # Plot each flight's trajectory with low opacity
    first_flight = True
    for file in os.listdir(coordinates_dir):
        if file.endswith("_trajectory.csv") and file != "avg_trajectory.csv":
            file_path = os.path.join(coordinates_dir, file)
            data = np.loadtxt(file_path, delimiter=",", skiprows=1)
            if data.ndim == 1:  # Only one point
                data = data.reshape(1, -1)
            plt.plot(data[:,1], data[:,2], color='gray', alpha=0.3, linewidth=2, label='Flight' if first_flight else "")
            first_flight = False

    # Plot the average trajectory with high opacity and a distinct color
    avg_data = np.loadtxt(avg_csv_path, delimiter=",", skiprows=1)
    if avg_data.ndim == 1:  # Only one point
        avg_data = avg_data.reshape(1, -1)
    plt.plot(avg_data[:,1], avg_data[:,2], color='red', alpha=1.0, linewidth=3, label='Average Trajectory')

    plt.legend(fontsize=10)
    plt.gca().invert_yaxis()  # Optional: if your y-coordinates are image coordinates
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"trajectory_plot.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"{Fore.CYAN}{Style.BRIGHT}  📈 Trajectory plot saved: {plot_path}")

print(f"{Fore.GREEN}{Style.BRIGHT}✅ All videos processed successfully.")