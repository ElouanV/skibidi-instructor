"""Turns analysis data into a readable coaching report, using Claude when available."""

from __future__ import annotations

import json
import os

MODEL = os.environ.get("SKIBIDI_MODEL", "claude-opus-5-5")

SYSTEM_PROMPT = """You are an experienced, certified ski instructor and biomechanics coach writing a video-analysis \
report for one skier. You receive measurements extracted from their video by a pose-estimation pipeline, a \
comparison against a reference built from expert skiers, rule-based fault detections, and a glossary.

Write the report in Markdown with these sections:
1. **TL;DR** - 2-3 sentences, encouraging and honest.
2. **What you already do well** - grounded in metrics within or above the reference.
3. **Top priorities** (max 3, most impactful first). For each: what we saw (cite the numbers), why it matters in \
plain language, the technical explanation (biomechanics, edge/pressure/rotary/balance), and 2 drills with a short \
"how to know it's working" cue.
4. **Technical appendix** - a table of metric, your value, reference range, verdict.
5. **Limits of this analysis** - repeat the caveats briefly and how to film better next time.

Rules: only make claims supported by the data. Weigh each fault by its confidence; if confidence is below 0.5, \
present it as "worth checking" rather than a finding. Remember turn sides are in image space (when filmed from \
below, image-left is the skier's right). Adapt vocabulary to the skier level if given. Be concise, no filler."""


def write_report(data: dict, language: str = "English") -> str:
    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        return template_report(data)
    import anthropic

    client = anthropic.Anthropic()
    payload = json.dumps(data, indent=1, default=float)
    with client.beta.messages.stream(
        model=MODEL,
        max_tokens=16000,
        system=SYSTEM_PROMPT,
        thinking={"type": "adaptive"},
        output_config={"effort": "high"},
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        messages=[{"role": "user", "content": f"Write the report in {language}.\n\n<analysis>\n{payload}\n</analysis>"}],
    ) as stream:
        msg = stream.get_final_message()
    if msg.stop_reason == "refusal":
        return template_report(data)
    text = "".join(b.text for b in msg.content if b.type == "text")
    return text or template_report(data)


def template_report(data: dict) -> str:
    """Deterministic report used offline / without an API key."""
    L = [f"# Ski analysis — {data['discipline']}", ""]
    L += [f"Turns analysed: **{data['n_turns']}** · camera view: **{data['camera_view']['view']}** · "
          f"reference: {data['reference']['name']}", ""]
    good = [c for c in data["comparison"] if c["status"] == "within_reference"]
    if good:
        L += ["## What you already do well", ""]
        L += [f"- **{c['label']}**: {c['value']}{c['unit']} (expert range {c['reference_range_p10_p90'][0]}–"
              f"{c['reference_range_p10_p90'][1]})" for c in good[:4]]
        L.append("")
    L += ["## Top priorities", ""]
    if not data["faults"]:
        L += ["No clear faults detected against the reference. Film more turns or steeper terrain to dig deeper.", ""]
    for i, f in enumerate(data["faults"][:3], 1):
        tag = "" if f["confidence"] >= 0.5 else " *(worth checking — low confidence for this camera angle)*"
        L += [f"### {i}. {f['title']}{tag}", "", f"**What we saw:** {f['evidence']}.", "",
              f"**Why it matters:** {f['why']}", "", "**Drills:**"]
        L += [f"- {d}" for d in f["drills"]]
        L.append("")
    L += ["## Technical appendix", "", "| Metric | You | Expert range (p10–p90) | Verdict |", "|---|---|---|---|"]
    for c in data["comparison"]:
        lo, hi = c["reference_range_p10_p90"]
        L.append(f"| {c['label']} | {c['value']}{c['unit']} | {lo}–{hi} | {c['status'].replace('_', ' ')} |")
    L += ["", "## Limits of this analysis", ""] + [f"- {c}" for c in data["caveats"]]
    return "\n".join(L) + "\n"
