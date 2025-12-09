"""Convert SVG files to PNG using cairosvg"""
import cairosvg
from pathlib import Path

# Lokasi file
assets_dir = Path(__file__).parent / "Assets"

# Convert START.svg to PNG
start_svg = assets_dir / "START.svg"
start_png = assets_dir / "START.png"

if start_svg.exists():
    cairosvg.svg2png(
        url=str(start_svg),
        write_to=str(start_png),
        output_width=200,
        output_height=60,
    )
    print(f"[OK] Converted {start_svg.name} to PNG")
else:
    print(f"[ERROR] {start_svg} not found")

# Convert Rectangle.svg to PNG (background button area)
rect_svg = assets_dir / "Rectangle 1.svg"
rect_png = assets_dir / "Rectangle.png"

if rect_svg.exists():
    cairosvg.svg2png(
        url=str(rect_svg),
        write_to=str(rect_png),
        output_width=389,
        output_height=114,
    )
    print(f"[OK] Converted {rect_svg.name} to PNG")
else:
    print(f"[ERROR] {rect_svg} not found")

print("\nDone!")
