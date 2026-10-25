# Tempo — every dish on the ticket, ready together

Tempo is a vision agent for kitchens that boil things. A camera above each
burner reads the pot surface (still → shimmer → simmer → rolling boil, and
starch foam). A physics model turns those visual stages into a time-to-boil
estimate, and an order planner uses it to schedule every dish of a ticket so
they finish together. Station screens tell each cook what to do now: what to
cut and how much, which burner to load, when to add noodles, when to plate.
When foam is about to overflow, Tempo lowers the heat by itself (the only
action it takes without a human, because it is safe and reversible).

```
orders ──► planner ◄── boil ETA ◄── physics model ◄── stage events ◄── OpenCV 5 ◄── camera
              │                                                              ▲
              ▼                                                              │
   station screens (prep / burners / pass)          foam forecast ──► heat down (auto) + alert
```

## What runs where

| Part | File | OpenCV 5 / AWS |
|---|---|---|
| Pot localisation, surface activity, foam | `tempo/vision.py` | `HoughCircles`, `calcOpticalFlowFarneback`, `connectedComponentsWithStats`, morphology, HSV |
| Heating physics, boil ETA, burner learning | `tempo/physics.py` | — |
| Kitchen simulator + policies | `tempo/kitchen.py` | — |
| Station-screen replay | `tempo/replay.py`, `app/static/index.html` | — |
| Web service | `app/main.py` | S3 (results), SNS (alerts), CloudWatch Logs |
| Deployment | `deploy/` | CloudFormation, EC2 Graviton (arm64) |

## Run locally

```bash
pip install -r requirements.txt
uvicorn app.main:app --port 8000      # open http://localhost:8000
python -m pytest -q tests
```

## Reproduce the evaluation

```bash
python -m eval.kitchen_eval 120                                # 120 services x 4 policies x 2 load levels
python -m eval.make_real eval 0 8                              # 8 photo-realistic simulated pots with ground truth
python -m eval.vision_eval data/real_eval --realistic          # perception + boil prediction, leave-one-out calibration
python -m eval.make_real dish pasta                            # board clips (also kimchi_stew, egg_soup; demo for ramen)
python -m eval.real_check                                      # by-eye sanity check on real stock clips (see below)
```

## Deploy to AWS (from CloudShell)

```bash
unzip -o tempo.zip -d tempo && cd tempo && bash deploy/deploy.sh you@example.com   # first time
rm -rf tempo && unzip -o tempo.zip -d tempo && cd tempo && bash deploy/update.sh   # ship new code, same URL
bash deploy/teardown.sh                                                            # after judging
```

## Real stock clips used for the sanity check

Free-licence clips, not redistributed here; place them in `data/stock/` as p1..p10 to rerun `eval.real_check`:
Pixabay 360622, 10794-226624914, 248797, 248803, 252226, 15976-268112384 (p1–p6);
Pexels 7261519 (p8), 14554560 (p10). Judged by eye; no thresholds were tuned on them.

## Honest limits

- Perception accuracy numbers come from simulated footage (labelled in every frame). Real stock clips
  were used only for a by-eye check that found two failure modes: hand/ladle motion reads as boiling,
  and absolute thresholds depend on camera distance. Stage thresholds must be calibrated per camera.
- The kitchen results come from a simulator whose assumptions (one cook,
  four burners, burner power off by -25..+10 %, cooks notice an unannounced
  boil after 4–25 s) are stated in `tempo/kitchen.py`.
- Under heavy load, planning for "ready together" lengthens order-to-table
  time; the trade-off is reported, not hidden.
- Tempo never films or tracks people; it only looks at pots.
