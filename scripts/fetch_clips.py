"""Download reference clips listed in a CSV and extract their poses.

CSV columns: url,discipline,start,end,notes   (start/end like 0:42 / 1:05)

Only use footage you have the right to use for this purpose (your own, your
club/coach's with permission, or openly licensed). Keep downloads local and
don't redistribute them. Requires `pip install yt-dlp` and ffmpeg.
"""
import argparse
import csv
import subprocess
from pathlib import Path

from skibidi.pose import extract


def main():
    p = argparse.ArgumentParser()
    p.add_argument("csv")
    p.add_argument("--out", default="data/reference")
    a = p.parse_args()
    for row in csv.DictReader(open(a.csv)):
        d = Path(a.out) / row["discipline"]
        d.mkdir(parents=True, exist_ok=True)
        stem = f"{abs(hash(row['url'] + row.get('start', ''))) % 10**8:08d}"
        video = d / f"{stem}.mp4"
        if not video.exists():
            cmd = ["yt-dlp", "-f", "mp4[height<=1080]/best[ext=mp4]", "-o", str(video), row["url"]]
            if row.get("start") and row.get("end"):
                cmd += ["--download-sections", f"*{row['start']}-{row['end']}"]
            subprocess.run(cmd, check=True)
        pose = video.with_suffix(".pose.npz")
        if not pose.exists():
            extract(video).save(pose)
            print("pose ->", pose)


if __name__ == "__main__":
    main()
