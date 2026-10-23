import argparse
import glob
import os
import sys

import cv2

from pipeline import LanePipeline


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(BASE_DIR, "..")


def process_images(paths, out_dir, debug):
    os.makedirs(out_dir, exist_ok=True)
    for path in paths:
        img = cv2.imread(path)
        if img is None:
            print("Ne mogu da ucitam:", path)
            continue
        pipeline = LanePipeline(smooth=False, debug_view=debug)
        result = pipeline.process(img)
        out_path = os.path.join(out_dir, os.path.basename(path))
        cv2.imwrite(out_path, result)
        print("Sacuvano:", out_path)


def process_video(path, out_dir, debug, show):
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print("Ne mogu da otvorim video:", path)
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    name = os.path.splitext(os.path.basename(path))[0] + "_output.mp4"
    out_path = os.path.join(out_dir, name)
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    pipeline = LanePipeline(smooth=True, debug_view=debug)
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        result = pipeline.process(frame)
        writer.write(result)
        frame_idx += 1
        if show:
            cv2.imshow("Detekcija trake", result)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        if frame_idx % 50 == 0:
            print("{}: {}/{}".format(os.path.basename(path), frame_idx, total))

    cap.release()
    writer.release()
    if show:
        cv2.destroyAllWindows()
    print("Sacuvano:", out_path)


def main():
    parser = argparse.ArgumentParser(description="Detekcija saobracajne trake")
    parser.add_argument("--images", nargs="*", help="putanje do slika")
    parser.add_argument("--videos", nargs="*", help="putanje do video snimaka")
    parser.add_argument("--all-images", action="store_true", help="obradi sve slike iz test_images")
    parser.add_argument("--all-videos", action="store_true", help="obradi sve snimke iz test_videos")
    parser.add_argument("--debug", action="store_true", help="prikazi medjukorake u uglu slike")
    parser.add_argument("--show", action="store_true", help="prikazi video tokom obrade")
    args = parser.parse_args()

    images = list(args.images or [])
    videos = list(args.videos or [])
    if args.all_images:
        images += sorted(glob.glob(os.path.join(ROOT_DIR, "test_images", "*.jpg")))
    if args.all_videos:
        videos += sorted(glob.glob(os.path.join(ROOT_DIR, "test_videos", "*.mp4")))

    if not images and not videos:
        parser.print_help()
        sys.exit(1)

    if images:
        process_images(images, os.path.join(BASE_DIR, "output_images"), args.debug)
    for video in videos:
        process_video(video, os.path.join(BASE_DIR, "output_videos"), args.debug, args.show)


if __name__ == "__main__":
    main()
