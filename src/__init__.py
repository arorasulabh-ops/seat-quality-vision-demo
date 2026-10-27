from __future__ import annotations

import cv2
import numpy as np
from dataclasses import dataclass
from pathlib import Path


@dataclass
class QualityResult:
    seam_score: float
    fabric_score: float
    defect_score: float
    overall_score: float
    verdict: str
    details: dict


def create_synthetic_seat(size: tuple[int, int], seam_quality: float, fabric_quality: float, defect_level: float) -> np.ndarray:
    height, width = size
    image = np.zeros((height, width, 3), dtype=np.uint8)

    yy, xx = np.mgrid[:height, :width]
    directional = 0.6 * np.sin(xx / 12.0) + 0.5 * np.sin(yy / 16.0)
    weave = 0.45 * np.sin(xx / 6.0) * np.sin(yy / 6.0)
    base = 120 + 18 * directional + 22 * weave
    base = np.clip(base, 60, 200).astype(np.uint8)
    image[:, :, 0] = base
    image[:, :, 1] = base - 8
    image[:, :, 2] = base - 15

    # Simulate seat shape
    seat_mask = np.zeros((height, width), dtype=np.uint8)
    cv2.ellipse(seat_mask, (width // 2, height // 2 + 20), (width // 3 + 10, height // 2 - 20), 0, 0, 360, 255, -1)
    image = cv2.bitwise_and(image, image, mask=cv2.cvtColor(seat_mask[:, :, None], cv2.COLOR_GRAY2BGR))

    # Add cloth texture variations and seam lines
    seam_mask = np.zeros((height, width), dtype=np.uint8)
    seam_center = width // 2 + 14
    cv2.line(seam_mask, (seam_center - 70, height // 2 - 30), (seam_center + 90, height // 2 + 40), 255, 3)
    cv2.line(seam_mask, (seam_center - 120, height // 2 + 80), (seam_center + 60, height // 2 + 150), 255, 2)
    seam_mask = cv2.GaussianBlur(seam_mask, (5, 5), 0)
    seam_overlay = np.zeros_like(image)
    seam_overlay[..., 0] = 40
    seam_overlay[..., 1] = 55
    seam_overlay[..., 2] = 70
    image = cv2.addWeighted(image, 1.0, seam_overlay, 0.35, 0)
    image = cv2.bitwise_and(image, image, mask=cv2.cvtColor(seat_mask, cv2.COLOR_GRAY2BGR))

    if seam_quality < 0.55:
        # uneven seam structure to mimic poor stitching
        for index in range(12):
            x0 = int(width / 2 - 100 + index * 12)
            y0 = int(height / 2 - 20 + index * 5)
            x1 = int(width / 2 + 100 + index * 8)
            y1 = int(height / 2 + 70 + index * 4)
            cv2.line(image, (x0, y0), (x1, y1), (30, 30, 30), 2)

    # Fabric quality variation
    if fabric_quality < 0.55:
        texture_noise = np.random.default_rng(42).normal(0, 18, size=(height, width, 3)).astype(np.int16)
        image = np.clip(image.astype(np.int16) + texture_noise, 0, 255).astype(np.uint8)

    # Defects
    defect_mask = np.zeros((height, width), dtype=np.uint8)
    for _ in range(int(defect_level * 25)):
        cx = int(np.random.uniform(80, width - 80))
        cy = int(np.random.uniform(80, height - 80))
        radius = int(np.random.uniform(10, 35))
        cv2.circle(defect_mask, (cx, cy), radius, 255, -1)
    if defect_level > 0.18:
        for _ in range(4):
            x0 = int(np.random.uniform(50, width - 50))
            y0 = int(np.random.uniform(80, height - 70))
            x1 = int(np.random.uniform(50, width - 50))
            y1 = int(np.random.uniform(80, height - 70))
            cv2.line(defect_mask, (x0, y0), (x1, y1), 255, 2)
    defect_mask = cv2.GaussianBlur(defect_mask, (9, 9), 0)
    image = cv2.addWeighted(image, 1.0, cv2.cvtColor(defect_mask, cv2.COLOR_GRAY2BGR), -0.2, 0)
    image = cv2.bitwise_and(image, image, mask=cv2.cvtColor(seat_mask, cv2.COLOR_GRAY2BGR))

    # Add seat border and a light reflection to look realistic
    border = np.zeros_like(image)
    cv2.ellipse(border, (width // 2, height // 2 + 20), (width // 3 + 25, height // 2 - 5), 0, 0, 360, (200, 200, 200), 8)
    image = cv2.addWeighted(image, 1.0, border, 0.12, 0)
    return image


def analyze_seat_image(image: np.ndarray) -> QualityResult:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Fabric / texture score: high local variance indicates richer fabric surface
    laplacian = cv2.Laplacian(blur, cv2.CV_64F)
    fabric_score = float(np.clip(np.mean(np.abs(laplacian)) / 18.0, 0.0, 1.0))

    # Seam quality: estimate the continuity and contrast of seam lines.
    edges = cv2.Canny(blur, 70, 180)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=25, minLineLength=40, maxLineGap=20)
    seam_score = 0.5
    if lines is not None:
        detected = len(lines)
        seam_score = float(np.clip(1.0 - abs(detected - 14) / 18.0, 0.0, 1.0))
    seam_score = float(np.clip(seam_score * 0.8 + fabric_score * 0.2, 0.0, 1.0))

    # Defect score: count irregular dark/bright areas; lower is better.
    diff = cv2.absdiff(gray, cv2.GaussianBlur(gray, (21, 21), 0))
    _, thresholded = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    defect_area = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 40:
            defect_area += int(area)
    defect_density = defect_area / (gray.shape[0] * gray.shape[1])
    defect_score = float(np.clip(1.0 - defect_density * 8.0, 0.0, 1.0))

    overall = seam_score * 0.42 + fabric_score * 0.33 + defect_score * 0.25
    if overall >= 0.75:
        verdict = "PASS"
    elif overall >= 0.57:
        verdict = "REVIEW"
    else:
        verdict = "FAIL"

    return QualityResult(
        seam_score=round(seam_score, 3),
        fabric_score=round(fabric_score, 3),
        defect_score=round(defect_score, 3),
        overall_score=round(overall, 3),
        verdict=verdict,
        details={
            "defect_area": defect_area,
            "contours": len(contours),
            "detected_lines": 0 if lines is None else len(lines),
        },
    )


def draw_analysis_overlay(frame: np.ndarray, result: QualityResult, view_title: str = "Seat Quality Inspection") -> np.ndarray:
    overlay = frame.copy()
    height, width = frame.shape[:2]
    cv2.rectangle(overlay, (20, 20), (width - 20, height - 20), (35, 35, 35), 2)

    # Top title bar
    cv2.rectangle(overlay, (25, 25), (width - 25, 80), (18, 18, 18), -1)
    cv2.putText(overlay, view_title, (40, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    # Metric cards
    metrics = [
        ("Seam", result.seam_score),
        ("Fabric", result.fabric_score),
        ("Defects", result.defect_score),
        ("Overall", result.overall_score),
    ]
    start_x = 30
    start_y = 110
    card_w = 220
    card_h = 70
    for idx, (label, value) in enumerate(metrics):
        x = start_x + idx * (card_w + 18)
        y = start_y
        cv2.rectangle(overlay, (x, y), (x + card_w, y + card_h), (60, 60, 60), 2)
        cv2.putText(overlay, label, (x + 18, y + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        cv2.putText(overlay, f"{value:.2f}", (x + 18, y + 55), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (112, 255, 120), 2)

    # Verdict badge
    verdict_color = (0, 200, 0) if result.verdict == "PASS" else (0, 165, 255) if result.verdict == "REVIEW" else (0, 0, 255)
    cv2.rectangle(overlay, (width - 250, 110), (width - 30, 180), verdict_color, -1)
    cv2.putText(overlay, result.verdict, (width - 220, 157), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

    # Heatmap-like panel on right side
    panel_x = width - 235
    panel_y = 220
    panel_w = 180
    panel_h = 180
    cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (70, 70, 70), 2)
    for row in range(10):
        for col in range(8):
            intensity = int(120 + 110 * ((row + col) % 2))
            color = (intensity, 120, 150)
            cv2.rectangle(
                overlay,
                (panel_x + 10 + col * 20, panel_y + 10 + row * 16),
                (panel_x + 24 + col * 20, panel_y + 18 + row * 16),
                color,
                -1,
            )
    cv2.putText(overlay, "Seat integrity", (panel_x + 15, panel_y + 205), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)

    # Lower summary bar
    summary = f"Inspection summary: seam={result.seam_score:.2f}, fabric={result.fabric_score:.2f}, defects={result.defect_score:.2f}, overall={result.overall_score:.2f}"
    cv2.rectangle(overlay, (25, height - 90), (width - 25, height - 25), (30, 30, 30), -1)
    cv2.putText(overlay, summary, (40, height - 48), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (235, 235, 235), 2)
    return overlay


def generate_demo_video(output_path: str | Path) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    fps = 15
    duration = 6
    total_frames = fps * duration

    width, height = 1280, 720
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for {output}")

    sample_cases = [
        (0.84, 0.81, 0.16),
        (0.71, 0.69, 0.28),
        (0.44, 0.52, 0.42),
    ]

    for frame_idx in range(total_frames):
        # Create a dynamic camera pan effect
        case_index = frame_idx % len(sample_cases)
        seam_quality, fabric_quality, defect_level = sample_cases[case_index]
        frame = create_synthetic_seat((height, width), seam_quality, fabric_quality, defect_level)

        # Animate slight camera motion and lighting variations
        pan = int((frame_idx % 60) * 1.2)
        translated = np.zeros_like(frame)
        translated[:, max(0, pan):min(width, width + pan), :] = frame[:, :max(0, width - pan), :]
        translated[:, :max(0, pan), :] = frame[:, width - max(0, pan):, :]
        frame = translated

        brightness = 100 + int(25 * np.sin(frame_idx / 8.0))
        frame = cv2.convertScaleAbs(frame, alpha=1.0, beta=brightness)

        result = analyze_seat_image(frame)
        final_frame = draw_analysis_overlay(frame, result, "Seat Inspection Demo")
        writer.write(final_frame)

    writer.release()


if __name__ == "__main__":
    generate_demo_video("output/seat_quality_demo.mp4")
