"""Command line entry point.

    skibidi extract VIDEO [-o pose.npz]
    skibidi build-ref --discipline carving -o references/carving_pros.json poses/*.npz
    skibidi analyze VIDEO --discipline carving [--ref references/carving_pros.json] [--out runs/x]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .analysis import DISCIPLINES, analyze_sequence, build_reference, build_report_data, load_reference
from .pose import DEFAULT_MODEL, PoseSequence, extract


def _get_pose(video: Path, model: str, stride: int, cache_dir: Path) -> PoseSequence:
    if video.suffix == ".npz":
        return PoseSequence.load(video)
    cache = cache_dir / f"{video.stem}.pose.npz"
    if cache.exists():
        return PoseSequence.load(cache)
    seq = extract(video, model, stride)
    seq.save(cache)
    return seq


def cmd_extract(a):
    out = Path(a.output or Path(a.video).with_suffix(".pose.npz"))
    seq = extract(a.video, a.model, a.stride)
    seq.save(out)
    print(f"Saved {seq.n_frames} frames -> {out}")


def cmd_build_ref(a):
    files = []
    for p in map(Path, a.inputs):
        files += sorted(p.glob("*.npz")) if p.is_dir() else [p]
    ref = build_reference(files, a.discipline, a.name, require_frontal=not a.any_view)
    Path(a.output).write_text(json.dumps(ref, indent=2))
    print(f"Reference from {len(ref['clips'])} clips ({len(ref['skipped'])} skipped) -> {a.output}")


def cmd_analyze(a):
    from .coach import write_report

    video = Path(a.video)
    out = Path(a.out or f"runs/{video.stem}")
    out.mkdir(parents=True, exist_ok=True)
    seq = _get_pose(video, a.model, a.stride, out)
    res = analyze_sequence(seq)
    ref = load_reference(a.discipline, a.ref)
    data = build_report_data(res, ref, a.discipline, a.level)
    (out / "analysis.json").write_text(json.dumps(data, indent=2, default=float))
    report = write_report(data, a.language)
    (out / "report.md").write_text(report)
    if not a.no_render:
        from .render import overlay_video, timeline_plot
        timeline_plot(res["metrics"], res["turns"], seq.fps, out / "timeline.png")
        if video.suffix != ".npz":
            overlay_video(video, seq, res["metrics"], res["turns"], out / "overlay.mp4", a.stride)
    print(report)
    print(f"\nOutputs in {out}/")


def main(argv=None):
    p = argparse.ArgumentParser(prog="skibidi", description="Ski technique video analysis")
    sub = p.add_subparsers(required=True)

    e = sub.add_parser("extract", help="extract skeleton keypoints from a video")
    e.add_argument("video")
    e.add_argument("-o", "--output")
    e.set_defaults(func=cmd_extract)

    r = sub.add_parser("build-ref", help="build a reference profile from pro/coach pose files")
    r.add_argument("inputs", nargs="+", help=".npz pose files or directories")
    r.add_argument("--discipline", choices=DISCIPLINES, required=True)
    r.add_argument("--name", default="")
    r.add_argument("--any-view", action="store_true", help="keep side-on clips too")
    r.add_argument("-o", "--output", required=True)
    r.set_defaults(func=cmd_build_ref)

    an = sub.add_parser("analyze", help="analyse your video and write a coaching report")
    an.add_argument("video", help="video file or a .npz pose file")
    an.add_argument("--discipline", choices=DISCIPLINES, default="carving")
    an.add_argument("--ref", help="reference JSON (default: built-in expert prior)")
    an.add_argument("--level", help="e.g. 'intermediate', 'advanced', 'instructor candidate'")
    an.add_argument("--language", default="English")
    an.add_argument("--out")
    an.add_argument("--no-render", action="store_true")
    an.set_defaults(func=cmd_analyze)

    for sp in (e, an):
        sp.add_argument("--model", default=DEFAULT_MODEL, help="ultralytics pose weights")
        sp.add_argument("--stride", type=int, default=1, help="process every Nth frame")

    a = p.parse_args(argv)
    a.func(a)


if __name__ == "__main__":
    main()
