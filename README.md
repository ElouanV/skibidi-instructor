# skibidi-instructor

Ski technique analysis from video. Film yourself, get a coaching report that is easy to read and still technically serious.

```
your video ──► pose extraction ──► biomechanics ──► turn segmentation ──► compare with experts ──► coaching report
              (YOLO-pose, 17 kp)   (angles, ratios)   (edge changes, apex)   (pose-derived reference)   (Claude + drills)
pro / coach clips ──► pose extraction ──► ... ──► reference distributions (p10–p90 per metric)
```

## Quick start

```bash
pip install -e ".[dev]"           # add ".[scrape]" for yt-dlp
export ANTHROPIC_API_KEY=...       # optional: without it you get a deterministic template report

skibidi analyze my_run.mp4 --discipline carving --level intermediate --language French
# -> runs/my_run/{report.md, analysis.json, overlay.mp4, timeline.png}
```

Build your own expert reference from clips (see `scripts/clips_example.csv`):

```bash
python scripts/fetch_clips.py my_clips.csv            # download + extract poses -> data/reference/<discipline>/
skibidi build-ref data/reference/carving --discipline carving -o references/carving_pros.json
skibidi analyze my_run.mp4 --ref references/carving_pros.json
```

Until you build one, `references/*_prior.json` hold **hand-set approximate expert ranges**. Treat them as placeholders.

## What is measured

Per frame (`skibidi/features.py`), then summarised per turn at the apex:

| Metric | Plane | Detects |
|---|---|---|
| Inclination (ankles→shoulders vs vertical) | lateral | edge angle, low edging |
| Angulation (leg lean − upper-body lean) | lateral | banking |
| Inside shoulder drop (trunk side-bend) | lateral | dropping the inside shoulder |
| Stance width, knee/ankle gap ratio | lateral | narrow stance, A-frame |
| Knee & hip flexion | sagittal | stiff/tall stance, folding at the waist |
| Flexion–extension range | sagittal | static "park and ride" |
| Hand height & spread | both | dropped/narrow hands |
| Left vs right turn differences | lateral | weaker side |

Faults are rule-based (`skibidi/analysis.py`) and grounded in a small knowledge base of explanations and drills (`skibidi/knowledge.py`). Claude then writes the report from this data only: TL;DR, strengths, top 3 priorities with numbers, biomechanics and drills, a technical table, and the limits of the analysis.

## Design choices worth knowing

- **Camera angle matters a lot.** 2D angles are projections. The tool estimates the view (frontal, oblique or side) and gives each fault a confidence. Lateral metrics need footage filmed along the fall line (from below or behind). Flexion needs side-on footage. **Best input: one clip from below plus one from the side, 6–8 turns each, the skier filling at least 1/3 of the frame, 60 fps if possible.**
- **Compare like with like.** A pose-derived reference only makes sense if the clips are filmed the same way as yours. `build-ref` drops side-on clips by default. Curating good clips matters more than collecting many.
- **Turn sides are in image space.** Filmed from below, image-left is the skier's right.
- **The LLM doesn't look at the video.** It explains measured numbers, so its claims stay traceable. Every report also includes `analysis.json`.

## About scraping YouTube

Downloading YouTube videos goes against YouTube's Terms of Service, and the footage belongs to its creators. Safer sources: your own footage, your club or coach's footage (with permission), openly licensed clips, or asking creators. `scripts/fetch_clips.py` is a helper for clips you're allowed to use. Keep them local and don't redistribute them.

## Roadmap

1. **3D lifting** (e.g. MotionBERT / a 3D pose model) to remove most camera-angle dependence. This is the biggest accuracy gain.
2. **Ski / pole detection** (fine-tuned keypoints on ski tips and tails) for true edge angle, ski separation and fore-aft balance.
3. **Park / tricks module**: detect takeoff, air and landing from the trajectory, then measure rotation (shoulder/hip yaw), pop timing, grab and landing absorption. The `park` discipline currently only checks stance metrics.
4. **Freeride specifics**: terrain-aware (pitch estimation), bump absorption timing.
5. **Learned scoring**: once you have labelled clips (coach ratings), train a small model on per-turn feature sequences instead of hand-set thresholds.
6. A web UI (upload → report with synced overlay video and timeline).

## Development

```bash
pytest -q        # synthetic-skier tests, no video or model needed
```
