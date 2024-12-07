"""CLI interface for vi_planner project.
"""

import argparse
import signal
import sys
import time

import cv2

from .model import VIPlanner


def signal_handler(sig, frame):
    """Handle Ctrl+C exit"""
    print("\nExiting...")
    cv2.destroyAllWindows()
    sys.exit(0)


def main():
    """
    The main function executes on commands:
    `python -m vi_planner` and `$ vi_planner `.
    """
    # Set up argument parser
    parser = argparse.ArgumentParser(description="VI-Planner demo")
    parser.add_argument("video_path", help="Path to input video file")
    parser.add_argument(
        "--headless",
        help="No display mode",
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "--save", help="Path to save output video", default=None
    )
    parser.add_argument(
        "--goal",
        nargs=3,
        type=float,
        default=[0, 0, 4],
        help="Goal coordinates (x,y,z)",
    )
    args = parser.parse_args()

    # Set up video capture
    cap = cv2.VideoCapture(args.video_path)

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Set up video writer if save path provided
    video_writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(
            args.save,
            fourcc,
            fps,
            (width, int(height * 1.5)),  # Account for visualization height
        )

    # Initialize planner
    planner = VIPlanner(
        fov_x=70,
        fov_y=70,
        height=height,
        width=width,
        offsets=(0, 1.5, 0),
        rotation=(0, 0, 0),
    )

    # Register signal handler for graceful exit
    signal.signal(signal.SIGINT, signal_handler)

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Run model
            start_time = time.time()
            # model_output = planner.run(frame, args.goal)
            model_output = planner.run(frame, [5, 0, 0])

            # Visualize results
            vis_frame = planner.visualize(frame, model_output)

            end_time = time.time()
            duration = end_time - start_time
            model_fps = 1 / duration
            print("FPS:", model_fps)

            skip_frames = int(duration * fps)
            for _ in range(skip_frames):
                ret, frame = cap.read()

            if not args.headless:
                # Display results
                cv2.imshow("VI-Planner", vis_frame)

                # Break if 'q' pressed
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            # Save frame if video writer exists
            if video_writer is not None:
                video_writer.write(vis_frame)

    finally:
        # Clean up
        if video_writer is not None:
            video_writer.release()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
