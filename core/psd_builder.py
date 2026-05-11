from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from fontTools.ttLib import TTFont
from PIL import Image

from photoshopapi import (
    GroupLayer_8bit,
    ImageLayer_8bit,
    LayeredFile_8bit,
    TextLayer_8bit,
)
from photoshopapi.enum import ChannelID, ColorMode, Justification


# ── PSD-specific font size recalculation ─────────────────────────────────────
# Uses fontTools raw metrics so the chosen size works in Photoshop's renderer
# regardless of differences from Skia/HarfBuzz (used for the PNG path).


def _get_tt_font(font_path: str) -> Optional[TTFont]:
    try:
        return TTFont(font_path, fontNumber=0)
    except Exception:
        return None


def _line_height_units(tt: TTFont) -> float:
    try:
        os2 = tt["OS/2"]
        return float(os2.sTypoAscender) + float(abs(os2.sTypoDescender)) + float(os2.sTypoLineGap)
    except Exception:
        try:
            hhea = tt["hhea"]
            return float(hhea.ascender) + float(abs(hhea.descender)) + float(hhea.lineGap)
        except Exception:
            return float(tt["head"].unitsPerEm) * 1.2


def _upem(tt: TTFont) -> float:
    return float(tt["head"].unitsPerEm)


def _glyph_advance(tt: TTFont, char: str, upem: float) -> float:
    try:
        cmap = tt.getBestCmap()
        if cmap is None:
            return upem * 0.5
        glyph_name = cmap.get(ord(char))
        if glyph_name is None:
            return upem * 0.5
        glyph_id = tt.getGlyphID(glyph_name)
        advance, _ = tt["hmtx"][glyph_id]
        return float(advance)
    except Exception:
        return upem * 0.5


def _text_advance(tt: TTFont, text: str, upem: float) -> float:
    total = 0.0
    for ch in text:
        total += _glyph_advance(tt, ch, upem)
    return total


def _split_psd_words(text: str) -> List[str]:
    """
    Split text into 'words' for PSD word-wrap measurement.
    Latin scripts: split on whitespace (space-delimited words).
    CJK: split each character individually since CJK wraps per-character.
    Mixed: handle both.
    """
    tokens: List[str] = []
    for segment in text.split():
        if not segment:
            continue
        # If any char in the segment is CJK, split into individual chars
        has_cjk = any(
            "\u4e00" <= c <= "\u9fff"
            or "\u3040" <= c <= "\u30ff"
            or "\uac00" <= c <= "\ud7af"
            or "\u3000" <= c <= "\u303f"
            for c in segment
        )
        if has_cjk:
            for ch in segment:
                tokens.append(ch)
        else:
            tokens.append(segment)
    return tokens if tokens else [text]


def _word_wrap_metrics(
    tt: TTFont,
    text: str,
    font_size: float,
    box_w: float,
    box_h: float,
    upem_val: float,
    line_h_units: float,
) -> Tuple[bool, int, float, float]:
    """Return (fits, num_lines, max_line_advance, total_height) for given font_size."""
    scale = font_size / upem_val
    avail_w = max(1.0, box_w - 6.0)

    line_adv = line_h_units * scale

    words = _split_psd_words(text)
    if not words:
        return True, 1, 0.0, line_adv

    lines = []
    cur_words: List[str] = []
    cur_adv = 0.0

    for w in words:
        w_adv = _text_advance(tt, w, upem_val) * scale
        gap = _glyph_advance(tt, " ", upem_val) * scale if cur_words else 0.0

        if cur_adv + gap + w_adv <= avail_w:
            if cur_words:
                cur_adv += gap
            cur_words.append(w)
            cur_adv += w_adv
        else:
            if cur_words:
                lines.append((cur_words, cur_adv))
            cur_words = [w]
            cur_adv = w_adv

    if cur_words:
        lines.append((cur_words, cur_adv))

    num_lines = len(lines)
    max_w = max((a for _, a in lines), default=0.0)
    total_h = num_lines * line_adv

    return total_h <= box_h, num_lines, max_w, total_h


def _recalculate_psd_layout(
    text: str,
    font_path: str,
    safe_box_w: float,
    safe_box_h: float,
    min_size: float = 4.0,
    max_size: float = 72.0,
) -> Optional[Dict]:
    """
    Use fontTools to find the largest font size where the entire text fits
    inside the safe box.  Returns {font_size, block_height, max_line_width}
    or None when the font can't be read.
    """
    tt = _get_tt_font(font_path)
    if tt is None:
        return None
    try:
        upem_val = _upem(tt)
        lh_units = _line_height_units(tt)
    except Exception:
        return None

    # Binary search for largest fitting font size
    lo, hi = min_size, max_size
    best_size = min_size

    for _ in range(25):  # ~0.25pt precision
        if lo > hi:
            break
        mid = (lo + hi) / 2.0
        fits, n_lines, mw, th = _word_wrap_metrics(
            tt, text, mid, safe_box_w, safe_box_h, upem_val, lh_units
        )
        if fits:
            best_size = mid
            best_lines = n_lines
            best_width = mw
            best_height = th
            lo = mid + 0.25
        else:
            hi = mid - 0.25

    # Final measurement at best_size
    _, n_lines, mw, th = _word_wrap_metrics(
        tt, text, best_size, safe_box_w, safe_box_h, upem_val, lh_units
    )

    return {
        "font_size": best_size,
        "block_height": th,
        "max_line_width": mw,
        "num_lines": n_lines,
    }


# ── PSD building ─────────────────────────────────────────────────────────────


def _get_font_postscript_name(font_path: str) -> str:
    try:
        tt = TTFont(font_path, fontNumber=0)
        name = tt["name"].getDebugName(6)
        tt.close()
        if name:
            return name
    except Exception:
        pass
    return Path(font_path).stem


def _pil_to_rgb_layer_data(img):
    if img.mode == "RGBA":
        background = Image.new("RGB", img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[3])
        img = background
    elif img.mode != "RGB":
        img = img.convert("RGB")
    img_array = np.array(img)
    return {
        ChannelID.red: np.ascontiguousarray(img_array[:, :, 0]),
        ChannelID.green: np.ascontiguousarray(img_array[:, :, 1]),
        ChannelID.blue: np.ascontiguousarray(img_array[:, :, 2]),
    }, img.size


def _to_argb_floats(rgb: Tuple[int, int, int]) -> list:
    return [1.0, rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0]


def _apply_text_style(
    layer: TextLayer_8bit,
    font_name: str,
    font_size: float,
    text_color: Tuple[int, int, int],
    has_bold: bool = False,
    has_italic: bool = False,
) -> None:
    try:
        editor = layer.style_all()
    except Exception:
        return
    if editor is None:
        return
    try:
        editor.set_font(font_name)
        editor.set_font_size(font_size)
        editor.set_fill_color(_to_argb_floats(text_color))
        editor.set_auto_leading(False)
        editor.set_leading(font_size * 1.3)
        editor.set_bold(has_bold)
        editor.set_italic(has_italic)
    except Exception:
        pass


def _apply_style_ranges(
    layer: TextLayer_8bit,
    style_segments: List[Tuple[int, int, str]],
    font_variant_paths: Dict[str, str],
    font_size: float,
    text_color: Tuple[int, int, int],
) -> None:
    if not style_segments:
        return

    _STYLE_FLAGS = {
        "bold":       (True,  False),
        "italic":     (False, True),
        "bold_italic": (True,  True),
    }
    _ps_name_cache: Dict[str, str] = {}

    for start, end, style in style_segments:
        if start >= end:
            continue
        try:
            rng = layer.style_range(start, end)
            if rng is None or not rng.valid:
                continue

            variant_path = font_variant_paths.get(style)
            if variant_path:
                if variant_path not in _ps_name_cache:
                    _ps_name_cache[variant_path] = _get_font_postscript_name(variant_path)
                ps_name = _ps_name_cache[variant_path]
                rng.set_font(ps_name)
            else:
                is_bold, is_italic = _STYLE_FLAGS.get(style, (False, False))
                rng.set_bold(is_bold)
                rng.set_italic(is_italic)

            rng.set_font_size(font_size)
            rng.set_fill_color(_to_argb_floats(text_color))
            rng.set_auto_leading(False)
            rng.set_leading(font_size * 1.3)
        except Exception:
            pass


def build_psd(cleaned_image, original_image, layers, output_path, verbose=False):
    bg_data, (w, h) = _pil_to_rgb_layer_data(cleaned_image)
    orig_data, _ = _pil_to_rgb_layer_data(original_image)

    psd = LayeredFile_8bit(ColorMode.rgb, w, h)

    text_group = GroupLayer_8bit(
        layer_name="Text Layers",
        is_collapsed=False,
    )
    text_group.fill = 1.0

    for i, layer_info in enumerate(layers):
        text = layer_info["translation"]
        if not text:
            continue

        font_path = layer_info["font_path"]
        font_name = layer_info.get("font_name") or _get_font_postscript_name(font_path)

        safe_box = layer_info["safe_box"]
        sx, sy, sw, sh = safe_box

        # ── Recalculate font size for PSD ────────────────────────────────────
        psd_layout = _recalculate_psd_layout(text, font_path, sw, sh)
        if psd_layout is not None:
            font_size = psd_layout["font_size"]
            block_height = psd_layout["block_height"]
            max_line_width = psd_layout["max_line_width"]
        else:
            # Fallback to Skia-computed values
            font_size = layer_info["font_size"]
            block_height = layer_info["block_height"]
            max_line_width = layer_info["max_line_width"]

        # Vertically center text block within safe_box
        y_offset = sy + max(0.0, (sh - block_height) / 2.0)

        text_color_rgb = layer_info.get("text_color_rgb", (0, 0, 0))
        r, g, b = text_color_rgb
        fill_color = [1.0, g / 255.0, b / 255.0, r / 255.0]

        text_layer = TextLayer_8bit(
            layer_name=f"Bubble {i + 1}",
            text=text,
            font=font_name,
            font_size=float(font_size),
            fill_color=fill_color,
            position_x=float(sx),
            position_y=float(y_offset),
            box_width=float(sw),
            box_height=float(sh),
        )
        text_layer.fill = 1.0
        text_layer.paragraph_all().set_justification(Justification.Center)

        _apply_text_style(
            text_layer,
            font_name,
            float(font_size),
            text_color_rgb,
            has_bold=layer_info.get("psd_text_has_bold", False),
            has_italic=layer_info.get("psd_text_has_italic", False),
        )

        _apply_style_ranges(
            text_layer,
            style_segments=layer_info.get("psd_style_segments", []),
            font_variant_paths=layer_info.get("psd_font_variant_paths", {}),
            font_size=float(font_size),
            text_color=text_color_rgb,
        )

        text_group.add_layer(psd, text_layer)

    psd.add_layer(text_group)

    bg_layer = ImageLayer_8bit(
        image_data=bg_data,
        layer_name="Background",
        width=w,
        height=h,
        pos_x=w / 2.0,
        pos_y=h / 2.0,
    )
    bg_layer.fill = 1.0
    psd.add_layer(bg_layer)

    original_layer = ImageLayer_8bit(
        image_data=orig_data,
        layer_name="Original",
        width=w,
        height=h,
        pos_x=w / 2.0,
        pos_y=h / 2.0,
    )
    original_layer.fill = 1.0
    psd.add_layer(original_layer)

    psd.write(str(output_path), force_overwrite=True)
