"""Ski technique knowledge base used to ground the analysis and the coaching report.

Each metric has a plain-language explanation and a technical one. Each fault has
the biomechanical cause, why it matters, and drills. This is what the LLM coach
is allowed to build on, so keep it accurate and conservative.
"""

METRICS = {
    "peak_inclination": {
        "label": "Peak inclination",
        "unit": "°",
        "plain": "How far your whole body tips into the turn at its sharpest point.",
        "technical": "Angle between the ankle-midpoint→shoulder-midpoint line and vertical at turn apex. "
                     "Drives edge angle together with angulation; must match speed and turn radius.",
    },
    "apex_angulation": {
        "label": "Angulation",
        "unit": "°",
        "plain": "How much your legs tip in while your upper body stays more upright, like a banana shape.",
        "technical": "Lower-body lean minus upper-body lean at apex. Hip/knee angulation lets you raise edge "
                     "angle while keeping the centre of mass balanced over the outside ski.",
    },
    "apex_knee_flex": {
        "label": "Knee flexion at apex",
        "unit": "°",
        "plain": "How bent your knees are in the middle of the turn.",
        "technical": "180° minus hip-knee-ankle angle, averaged over both legs. Too little = stiff, poor "
                     "absorption; too much while hips stay back = back seat.",
    },
    "apex_hip_flex": {
        "label": "Hip flexion at apex",
        "unit": "°",
        "plain": "How much you fold at the waist.",
        "technical": "180° minus shoulder-hip-knee angle. Excessive hip flexion with low knee/ankle flexion is "
                     "the classic 'bending at the waist' compensation for being back.",
    },
    "apex_shoulder_drop": {
        "label": "Inside shoulder drop",
        "unit": "°",
        "plain": "Whether your inside shoulder dips towards the snow (positive = dipping).",
        "technical": "Shoulder-line tilt relative to horizontal, signed by turn direction. A dropped inside "
                     "shoulder shifts weight to the inside ski and kills angulation.",
    },
    "apex_stance_width": {
        "label": "Stance width",
        "unit": "× torso",
        "plain": "How far apart your feet are, relative to your body size.",
        "technical": "Horizontal ankle gap / torso length at apex. Carving needs a functional (hip-width) "
                     "stance so the inside leg can flex and both skis can tip.",
    },
    "apex_knee_ankle_ratio": {
        "label": "Knee/ankle gap ratio",
        "unit": "",
        "plain": "Whether your knees come closer together than your feet (an 'A' shape).",
        "technical": "Horizontal knee gap / ankle gap at apex. Values well below 1 indicate an A-frame: the "
                     "inside leg is not tipping as much as the outside leg.",
    },
    "extension_range": {
        "label": "Flexion–extension range",
        "unit": "",
        "plain": "How much you move up and down through each turn instead of staying frozen.",
        "technical": "Range of hip-to-ankle distance / leg length within a turn. Active skiers flex and extend "
                     "to manage pressure; a tiny range means a static, 'park and ride' position.",
    },
    "mean_hand_height": {
        "label": "Hand height",
        "unit": "× torso",
        "plain": "Where your hands are — ideally in front and above your hips.",
        "technical": "Vertical wrist-midpoint height above hip midpoint / torso length. Low, dropped hands "
                     "usually go with a rotated or back-leaning upper body.",
    },
    "mean_hand_spread": {
        "label": "Hand spread",
        "unit": "× shoulders",
        "plain": "How wide your hands are, like holding a tray.",
        "technical": "Horizontal wrist gap / shoulder width. A stable, wide hand frame helps balance and pole "
                     "plant timing.",
    },
    "duration_s": {
        "label": "Turn duration",
        "unit": "s",
        "plain": "How long each turn lasts.",
        "technical": "Time between edge changes. Context for rhythm and turn shape, not a fault on its own.",
    },
}

FAULTS = {
    "banking": {
        "title": "Banking (leaning in with the whole body)",
        "why": "Edge grip comes from the body tipping in as one block instead of angulating, so balance "
               "shifts to the inside ski and the skis skid or wash out at higher edge angles.",
        "drills": ["Hands on knees / outside hand on outside knee through the turn",
                   "Pole across the shoulders held parallel to the slope",
                   "Javelin turns (lift inside ski, keep it crossed over the outside ski tip)"],
    },
    "static_stance": {
        "title": "Static stance (park and ride)",
        "why": "Without flexion and extension you can't manage pressure through the turn, edge changes "
               "get abrupt, and bumps/variable snow throw you off balance.",
        "drills": ["Exaggerated up-down: tall at transition, low at apex, count it out loud",
                   "Touch the outside boot top at each apex",
                   "Retraction turns: pull feet up under you through the transition"],
    },
    "too_tall": {
        "title": "Too tall / stiff legs",
        "why": "Straight legs can't absorb terrain or build edge angle; usually comes with hips over the tails.",
        "drills": ["Shin pressure on the boot tongue the whole run",
                   "Skiing with hands touching the knees on gentle terrain",
                   "Hop turns to find an athletic, flexed stance"],
    },
    "inside_shoulder_drop": {
        "title": "Dropping the inside shoulder",
        "why": "Moves mass to the inside ski and removes angulation; you'll feel the outside ski lose grip.",
        "drills": ["Keep shoulders level with the slope: imagine a tray of drinks on them",
                   "Drag the outside pole tip lightly through the turn",
                   "Outside-ski-only turns on easy terrain"],
    },
    "a_frame": {
        "title": "A-frame (inside knee collapses)",
        "why": "The inside leg isn't tipping and flexing as much as the outside leg, so skis run at different "
               "edge angles and carving becomes inconsistent.",
        "drills": ["Tip the inside ski onto its little-toe edge first to start the turn",
                   "Pull the inside foot back and flex the inside leg more",
                   "Railroad-track turns on a gentle slope: two clean lines in the snow"],
    },
    "narrow_stance": {
        "title": "Stance too narrow for carving",
        "why": "Feet glued together limit how far you can tip the skis and make balance on edge harder.",
        "drills": ["Railroad-track turns with hip-width stance",
                   "Ski with a fist-width gap between boots and check your tracks"],
    },
    "hands_low": {
        "title": "Hands dropped / behind",
        "why": "Low hands pull the shoulders back and up, often putting weight on the tails.",
        "drills": ["Hold a pole horizontally in front of you, like a tray",
                   "Keep both hands in your peripheral vision the whole run"],
    },
    "hands_narrow": {
        "title": "Hands too close to the body",
        "why": "A narrow frame reduces lateral balance and makes pole plants late.",
        "drills": ["Ski with hands wide, as if carrying a big tray", "Pole-plant rhythm drill on a gentle slope"],
    },
    "low_edge_angle": {
        "title": "Low edge angles",
        "why": "Without tipping the skis further, they can't bend and carve; turns stay skidded.",
        "drills": ["Garlands on edge: traverse and roll the knees uphill to engage edges",
                   "Thousand-step turns", "Progressively faster railroad tracks"],
    },
    "side_asymmetry": {
        "title": "One side is weaker",
        "why": "Turns to one side show clearly less lean or angulation; this is common and very trainable.",
        "drills": ["Repeat your best drill on the weaker side only (traverses + garlands)",
                   "Film both directions and compare frame by frame"],
    },
}
