# Seat Quality Vision Demo

A lightweight computer-vision demo for evaluating car-seat quality using a simple image-analysis pipeline for seam continuity, fabric texture, and defect detection.

This project creates a short demo video that simulates a manufacturing quality inspection workflow for automotive seats.

## Features
- Seam alignment / stitching inspection
- Fabric texture quality analysis
- Defect detection (wrinkles, tears, irregularities)
- Final quality verdict: PASS / REVIEW / FAIL
- Animated dashboard overlay similar to a production inspection UI

## Installation

1. Clone the repository.
2. Create a virtual environment if needed.
3. Install requirements:

```bash
pip install -r requirements.txt
```

## Run the demo

```bash
python demo.py
```

This produces a video at:

```text
output/seat_quality_demo.mp4
```

## Demo concept
The system is designed to be easy to extend:
- Replace synthetic seat frames with real seat photographs.
- Use the analysis logic as a template for a real inspection pipeline.
- Add camera calibration, ROI masking, and model-based defect detection for production use.

## Notes
This is a proof-of-concept demo, not a deployment-ready industrial inspection system. It is intended to visualize a seat-quality inspection workflow and help with pitch/demo presentations.
