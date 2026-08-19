"""Render overlay marks onto the photographed score sheet."""

from __future__ import annotations

from pathlib import Path

from doppelt.cli.overlay import (
    DicePanelState,
    OverlayMark,
    ShownDie,
    board_image_path,
    load_board_layout,
)

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError as exc:  # pragma: no cover - exercised when Pillow is missing
    Image = None  # type: ignore[assignment]
    ImageDraw = None  # type: ignore[assignment]
    ImageFont = None  # type: ignore[assignment]
    _PIL_ERROR = exc
else:
    _PIL_ERROR = None

DIE_FILL = {
    "white": (244, 244, 246),
    "yellow": (240, 196, 36),
    "blue": (42, 108, 186),
    "green": (62, 158, 58),
    "pink": (214, 92, 148),
    "silver": (154, 160, 168),
}

DIE_PIP = {
    "white": (40, 40, 44),
    "yellow": (40, 32, 12),
    "blue": (245, 248, 255),
    "green": (245, 252, 245),
    "pink": (255, 245, 250),
    "silver": (32, 32, 36),
}

# Unit offsets from die center for faces 1–6.
_PIPS: dict[int, tuple[tuple[float, float], ...]] = {
    1: ((0.0, 0.0),),
    2: ((-0.28, -0.28), (0.28, 0.28)),
    3: ((-0.28, -0.28), (0.0, 0.0), (0.28, 0.28)),
    4: ((-0.28, -0.28), (0.28, -0.28), (-0.28, 0.28), (0.28, 0.28)),
    5: ((-0.28, -0.28), (0.28, -0.28), (0.0, 0.0), (-0.28, 0.28), (0.28, 0.28)),
    6: (
        (-0.28, -0.30),
        (0.28, -0.30),
        (-0.28, 0.0),
        (0.28, 0.0),
        (-0.28, 0.30),
        (0.28, 0.30),
    ),
}


def require_pillow() -> None:
    if _PIL_ERROR is not None:
        raise RuntimeError(
            "Pillow is required for visualize. Install with: pip install pillow"
        ) from _PIL_ERROR


def _font(size: int):
    candidates = (
        Path(r"C:\Windows\Fonts\arialbd.ttf"),
        Path(r"C:\Windows\Fonts\arial.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
        Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf"),
    )
    for path in candidates:
        if path.is_file():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def load_board_image() -> Image.Image:
    require_pillow()
    path = board_image_path()
    if not path.is_file():
        raise FileNotFoundError(f"score sheet image missing: {path}")
    return Image.open(path).convert("RGBA")


def _draw_die(draw: ImageDraw.ImageDraw, cx: float, cy: float, size: float, die: ShownDie) -> None:
    half = size / 2
    box = (cx - half, cy - half, cx + half, cy + half)
    if die.color == "empty":
        draw.rounded_rectangle(
            box,
            radius=size * 0.16,
            fill=(28, 28, 32),
            outline=(90, 90, 96),
            width=2,
        )
        return
    fill = DIE_FILL.get(die.color, (200, 200, 200))
    outline = (255, 214, 80) if die.highlight else (18, 18, 20)
    width = 4 if die.highlight else 2
    draw.rounded_rectangle(box, radius=size * 0.16, fill=fill, outline=outline, width=width)
    if die.face is None:
        font = _font(max(14, int(size * 0.42)))
        text = "?"
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        pip = DIE_PIP.get(die.color, (40, 40, 40))
        draw.text((cx - tw / 2 - bbox[0], cy - th / 2 - bbox[1]), text, fill=pip, font=font)
        return
    pip_color = DIE_PIP.get(die.color, (30, 30, 30))
    radius = size * 0.08
    for dx, dy in _PIPS.get(die.face, ()):
        px, py = cx + dx * size, cy + dy * size
        draw.ellipse((px - radius, py - radius, px + radius, py + radius), fill=pip_color)


def _draw_dice_group(
    draw: ImageDraw.ImageDraw,
    *,
    origin_x: float,
    origin_y: float,
    title: str,
    dice: tuple[ShownDie, ...],
    spec: dict,
    fonts: tuple,
) -> None:
    label_font, empty_font = fonts
    label = spec["label"]
    draw.text((origin_x, origin_y), title, fill=tuple(label), font=label_font)
    size = spec["die_size"]
    gap = spec["gap"]
    columns = spec["columns"]
    start_y = origin_y + 34
    if not dice:
        draw.text(
            (origin_x, start_y),
            "(empty)",
            fill=(160, 160, 164),
            font=empty_font,
        )
        return
    for index, die in enumerate(dice):
        col = index % columns
        row = index // columns
        cx = origin_x + size / 2 + col * (size + gap)
        cy = start_y + size / 2 + row * (size + gap)
        _draw_die(draw, cx, cy, size, die)
        if die.badge:
            bbox = draw.textbbox((0, 0), die.badge, font=empty_font)
            tw = bbox[2] - bbox[0]
            draw.text(
                (cx - tw / 2 - bbox[0], cy + size / 2 + 2),
                die.badge,
                fill=tuple(label),
                font=empty_font,
            )


def _canvas_with_dice_panel(board: Image.Image, dice: DicePanelState) -> Image.Image:
    layout = load_board_layout()
    spec = layout["dice_panel"]
    panel_w = int(spec["width"])
    canvas = Image.new("RGBA", (board.width + panel_w, board.height), tuple(spec["background"]) + (255,))
    canvas.paste(board, (0, 0))
    draw = ImageDraw.Draw(canvas)
    fonts = (_font(20), _font(14))
    origin_x = board.width + spec["padding"]
    _draw_dice_group(
        draw,
        origin_x=origin_x,
        origin_y=spec["rolled_y"],
        title="Rolled",
        dice=dice.rolled,
        spec=spec,
        fonts=fonts,
    )
    _draw_dice_group(
        draw,
        origin_x=origin_x,
        origin_y=spec["picked_y"],
        title="Picked",
        dice=dice.picked,
        spec=spec,
        fonts=fonts,
    )
    _draw_dice_group(
        draw,
        origin_x=origin_x,
        origin_y=spec["platter_y"],
        title="Silver platter",
        dice=dice.platter,
        spec=spec,
        fonts=fonts,
    )
    return canvas


def render_overlay(
    marks: list[OverlayMark],
    *,
    base: Image.Image | None = None,
    dice: DicePanelState | None = None,
) -> Image.Image:
    require_pillow()
    layout = load_board_layout()
    ink = tuple(layout["ink"])
    highlight = tuple(layout["highlight"])
    image = (base or load_board_image()).copy()
    draw = ImageDraw.Draw(image)
    font = _font(18)
    small = _font(14)

    for mark in marks:
        cx, cy, r = mark.x, mark.y, mark.radius
        color = highlight if mark.highlight else ink
        width = 4 if mark.highlight else 3
        if mark.kind == "circle":
            draw.ellipse((cx - r, cy - r, cx + r, cy + r), outline=color, width=width)
        elif mark.kind == "cross":
            inset = r * 0.55
            draw.line((cx - inset, cy - inset, cx + inset, cy + inset), fill=color, width=width)
            draw.line((cx - inset, cy + inset, cx + inset, cy - inset), fill=color, width=width)
        elif mark.kind == "text" and mark.text is not None:
            used = small if len(mark.text) > 2 else font
            bbox = draw.textbbox((0, 0), mark.text, font=used)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            draw.text(
                (cx - tw / 2 - bbox[0], cy - th / 2 - bbox[1]),
                mark.text,
                fill=color,
                font=used,
            )
    if dice is not None:
        image = _canvas_with_dice_panel(image, dice)
    return image.convert("RGB")


def save_overlay(
    marks: list[OverlayMark],
    path: Path,
    *,
    base: Image.Image | None = None,
    dice: DicePanelState | None = None,
) -> None:
    render_overlay(marks, base=base, dice=dice).save(path)
