from pathlib import Path

from src.seat_quality_demo import generate_demo_video


def main() -> None:
    output_path = Path("output") / "seat_quality_demo.mp4"
    output_path.parent.mkdir(exist_ok=True)
    generate_demo_video(output_path)
    print(f"Demo video saved to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
