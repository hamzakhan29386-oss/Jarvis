"""Generate the JARVIS Windows icon used by tray and PyInstaller."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    assets = root / "assets"
    assets.mkdir(exist_ok=True)

    image = Image.new("RGBA", (256, 256), (5, 12, 18, 255))
    draw = ImageDraw.Draw(image)
    draw.ellipse((28, 28, 228, 228), outline=(0, 220, 255, 255), width=12)
    draw.ellipse((72, 72, 184, 184), outline=(0, 120, 160, 255), width=5)
    draw.ellipse((104, 104, 152, 152), fill=(0, 230, 255, 255))
    draw.line((128, 12, 128, 52), fill=(0, 220, 255, 180), width=4)
    draw.line((128, 204, 128, 244), fill=(0, 220, 255, 180), width=4)
    draw.line((12, 128, 52, 128), fill=(0, 220, 255, 180), width=4)
    draw.line((204, 128, 244, 128), fill=(0, 220, 255, 180), width=4)

    icon_path = assets / "jarvis.ico"
    image.save(icon_path, sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
    print(f"Generated {icon_path}")


if __name__ == "__main__":
    main()

