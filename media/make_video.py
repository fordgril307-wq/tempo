"""Assemble the 2-3 minute demo video from scenes + narration + burned-in captions."""
import subprocess
from pathlib import Path

R = Path(__file__).resolve().parent
D = R.parent / "docs"
S = R / "scenes"
V = R / "private" / "voice" / "take3.mp3"
OUT = R / "Tempo_demo.mp4"
W = R / "build"; W.mkdir(exist_ok=True)
LEAD, TAIL = 0.6, 2.4
END = 129.96

# (abs start in narration, source, kind, src offset)
SCENES = [
    (0.0,    S / "intro.png", "img", 0),
    (10.77,  S / "problem.png", "img", 0),
    (24.82,  R / "kitchen_twin_demo.mp4", "vid", 6),
    (43.46,  D / "fig_camera_view.png", "img", 0),
    (52.94,  D / "fig_signals.png", "img", 0),
    (64.11,  S / "board.mp4", "vid", 2),
    (74.47,  S / "line.mp4", "vid", 12.5),
    (84.93,  D / "fig_arch.png", "img", 0),
    (95.75,  D / "fig_results.png", "img", 0),
    (114.43, D / "fig_real.png", "img", 0),
    (123.93, S / "close.png", "img", 0),
]
CAPS = [
    (0.0, "Hi, I'm Hyesoo Kim. I work in manufacturing operations in Korea."),
    (4.86, "I built Tempo — a vision agent that makes every dish on a table come out together."),
    (10.77, "Kitchens time everything after the water boils."),
    (13.73, "But nothing tells them when it will boil."),
    (16.67, "So one dish waits and goes cold while another is still heating —"),
    (20.70, "and pots get left alone until they boil over."),
    (24.82, "This is the whole kitchen, from above."),
    (27.57, "Orders come in at the bottom."),
    (29.64, "The prep station knows what to cut, and how much."),
    (33.34, "Above every burner there is a camera, and a ring of lights."),
    (38.19, "The cook moves between stations, and every screen runs on the same clock."),
    (43.46, "OpenCV 5 finds the pot, and measures shimmer, new bubbles and turbulence."),
    (49.74, "A heating model turns that into a countdown."),
    (52.94, "Here, the recipe says 330 seconds."),
    (57.12, "At the first shimmer, Tempo says 257."),
    (61.18, "The pot boils at 250."),
    (64.11, "When the camera changes an estimate, the plan changes — and the board says why."),
    (69.99, "Short dishes start later, so the whole ticket is ready together."),
    (74.47, "When starch foam is about to overflow, Tempo lowers the heat."),
    (81.81, "That is the only thing it does on its own."),
    (84.93, "It all runs on AWS Graviton: OpenCV, the physics model and the agent in one container,\\Nwith S3, SNS and CloudWatch, deployed from one CloudFormation template."),
    (95.75, "In a simulated kitchen over 120 services, cooked food waited 66% less in quiet service,\\Nand 38% less in a rush."),
    (106.39, "Boil-overs nearly disappeared."),
    (109.35, "To be honest: in a rush, the table takes about 14% longer."),
    (114.43, "Our camera numbers come from simulated footage, and on real clips we found two failures —\\Nhands and ladles — which are next on our list."),
    (123.93, "Tempo watches pots, never people."),
    (127.13, "Next: real kitchens. Thank you."),
]


def ts(t):
    h, m = int(t // 3600), int(t % 3600 // 60)
    return f"{h}:{m:02d}:{t % 60:05.2f}"


def run(*a):
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", *a], check=True)


total = LEAD + END + TAIL
parts = []
for i, (t0, src, kind, off) in enumerate(SCENES):
    t1 = SCENES[i + 1][0] + LEAD if i + 1 < len(SCENES) else total
    a = 0 if i == 0 else t0 + LEAD
    dur = t1 - a
    out = W / f"p{i:02d}.mp4"
    fit = ("scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2:color=0x18232a,"
           "setsar=1,fps=30,format=yuv420p")
    if kind == "img":
        # slow push-in, 3% over the scene
        n = int(dur * 30)
        run("-loop", "1", "-i", str(src), "-t", f"{dur:.3f}", "-vf",
            f"scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2:color=0x18232a,setsar=1,"
            f"scale=3840:2160,zoompan=z='1+0.03*on/{n}':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d=1:s=1920x1080:fps=30,format=yuv420p",
            "-c:v", "libx264", "-crf", "18", "-preset", "medium", str(out))
    else:
        run("-ss", str(off), "-i", str(src), "-t", f"{dur:.3f}", "-vf", fit, "-an", "-c:v", "libx264", "-crf", "18",
            "-preset", "medium", str(out))
    parts.append(out)

(W / "list.txt").write_text("".join(f"file '{p}'\n" for p in parts))
run("-f", "concat", "-safe", "0", "-i", str(W / "list.txt"), "-c", "copy", str(W / "video.mp4"))

ass = ["[Script Info]", "ScriptType: v4.00+", "PlayResX: 1920", "PlayResY: 1080", "",
       "[V4+ Styles]",
       "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
       "Style: Default,Inter,40,&H00FFFFFF,&H00FFFFFF,&H00000000,&HA0101418,0,0,0,0,100,100,0,0,3,14,0,2,120,120,46,1",
       "", "[Events]", "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text"]
for i, (t, txt) in enumerate(CAPS):
    e = CAPS[i + 1][0] if i + 1 < len(CAPS) else END
    ass.append(f"Dialogue: 0,{ts(t + LEAD)},{ts(e + LEAD - 0.05)},Default,,0,0,0,,{txt}")
(W / "caps.ass").write_text("\n".join(ass))
run("-i", str(W / "video.mp4"), "-itsoffset", str(LEAD), "-i", str(V),
    "-vf", f"ass={W / 'caps.ass'}", "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-crf", "20", "-preset", "medium",
    "-c:a", "aac", "-b:a", "160k", "-af", "loudnorm=I=-16:TP=-1.5,apad", "-t", f"{total:.2f}", "-movflags", "+faststart",
    str(OUT))
print(OUT, total)
