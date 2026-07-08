"""Fast OpenCV text-region hints before expensive OCR."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

from .config import VisionConfig
from .frames import KeyFrame


@dataclass(frozen=True)
class OcrRegion:
    x: int
    y: int
    width: int
    height: int
    text_density: float
    edge_density: float


@dataclass(frozen=True)
class OcrFrameCandidate:
    frame: KeyFrame
    source_frame: KeyFrame
    regions: tuple[OcrRegion, ...]
    selected_for_ocr: bool
    reason: str


def prepare_ocr_candidates(frames: list[KeyFrame], config: VisionConfig) -> list[OcrFrameCandidate]:
    if not config.ocr_roi_enabled:
        return [
            OcrFrameCandidate(frame=frame, source_frame=frame, regions=(), selected_for_ocr=True, reason="roi_disabled")
            for frame in frames
        ]

    candidates: list[OcrFrameCandidate] = []
    for frame in frames:
        candidates.extend(_prepare_frame_regions(frame, config))
    return candidates


def _prepare_frame_regions(frame: KeyFrame, config: VisionConfig) -> list[OcrFrameCandidate]:
    try:
        import cv2
    except ImportError:
        return [OcrFrameCandidate(frame, frame, (), True, "opencv_missing")]

    image = cv2.imread(frame.image_path)
    if image is None:
        return [OcrFrameCandidate(frame, frame, (), True, "image_read_failed")]

    if config.ocr_roi_window_first:
        window = detect_foreground_window_region(image, config)
        if window is not None:
            return _crop_candidates(frame, image, (window,), "foreground_window")

    regions = detect_text_regions(image, config)
    if not regions:
        return [OcrFrameCandidate(frame, frame, (), False, "no_text_region")]

    return _crop_candidates(frame, image, tuple(regions), "text_region")


def _crop_candidates(
    frame: KeyFrame,
    image,
    regions: tuple[OcrRegion, ...],
    reason: str,
) -> list[OcrFrameCandidate]:
    import cv2

    crop_dir = Path(tempfile.mkdtemp(prefix="dld_ocr_roi_"))
    candidates: list[OcrFrameCandidate] = []
    for index, region in enumerate(regions):
        crop = image[region.y : region.y + region.height, region.x : region.x + region.width]
        if crop.size == 0:
            continue
        crop_path = crop_dir / f"{frame.frame_id}_roi_{index}_{frame.timestamp_ms}.jpg"
        cv2.imwrite(str(crop_path), crop)
        roi_frame = KeyFrame(
            frame_id=f"{frame.frame_id}_roi_{index}",
            timestamp_ms=frame.timestamp_ms,
            image_path=str(crop_path),
            score=frame.score,
            reason=f"{frame.reason}:roi",
            window_id=frame.window_id,
        )
        candidates.append(OcrFrameCandidate(roi_frame, frame, (region,), True, reason))
    return candidates or [OcrFrameCandidate(frame, frame, tuple(regions), True, "roi_crop_failed")]


def detect_foreground_window_region(image, config: VisionConfig) -> OcrRegion | None:
    import cv2

    height, width = image.shape[:2]
    if height <= 0 or width <= 0:
        return None

    scale = min(1.0, 900.0 / max(width, height))
    small = cv2.resize(image, (max(1, int(width * scale)), max(1, int(height * scale)))) if scale < 1.0 else image
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    light_panel = cv2.inRange(saturation, 0, 80) & cv2.inRange(value, 145, 255)
    dark_panel = cv2.inRange(saturation, 0, 100) & cv2.inRange(value, 0, 95)
    strict_light_panel = cv2.inRange(saturation, 0, 45) & cv2.inRange(value, 170, 255)
    mask = light_panel | dark_panel
    mask[int(mask.shape[0] * 0.88) :, :] = 0
    strict_light_panel[int(strict_light_panel.shape[0] * 0.88) :, :] = 0
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (35, 25)), iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)), iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    frame_area = small.shape[0] * small.shape[1]
    best: tuple[float, tuple[int, int, int, int]] | None = None
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        ratio = area / max(frame_area, 1)
        if ratio < 0.12:
            continue
        if ratio > 0.86:
            projected = _window_region_from_projection(strict_light_panel, width, height, scale, config)
            if projected is not None:
                return projected
            continue
        if w < small.shape[1] * 0.25 or h < small.shape[0] * 0.25:
            continue
        if y > small.shape[0] * 0.55:
            continue
        fill = cv2.countNonZero(mask[y : y + h, x : x + w]) / max(area, 1)
        if fill < 0.35:
            continue
        score = ratio * fill
        if best is None or score > best[0]:
            best = (score, (x, y, w, h))

    if best is None:
        return _window_region_from_projection(strict_light_panel, width, height, scale, config)

    inv_scale = 1.0 / scale
    x, y, w, h = best[1]
    pad = min(config.ocr_roi_padding, 8)
    ox = max(0, int((x - pad) * inv_scale))
    oy = max(0, int((y - pad) * inv_scale))
    ox2 = min(width, int((x + w + pad) * inv_scale))
    oy2 = min(height, int((y + h + pad) * inv_scale))
    return OcrRegion(
        x=ox,
        y=oy,
        width=max(1, ox2 - ox),
        height=max(1, oy2 - oy),
        text_density=0.0,
        edge_density=0.0,
    )


def _window_region_from_projection(mask, width: int, height: int, scale: float, config: VisionConfig) -> OcrRegion | None:
    col_density = (mask > 0).mean(axis=0)
    row_density = (mask > 0).mean(axis=1)
    x_range = _dominant_projection_range(col_density, threshold=0.35, min_length=max(30, mask.shape[1] // 5))
    y_range = _dominant_projection_range(row_density, threshold=0.35, min_length=max(30, mask.shape[0] // 5))
    if x_range is None or y_range is None:
        return None

    x1, x2 = x_range
    y1, y2 = y_range
    small_area = mask.shape[0] * mask.shape[1]
    ratio = ((x2 - x1) * (y2 - y1)) / max(small_area, 1)
    if ratio < 0.10 or ratio > 0.86:
        return None

    inv_scale = 1.0 / scale
    pad = min(config.ocr_roi_padding, 8)
    ox = max(0, int((x1 - pad) * inv_scale))
    oy = max(0, int((y1 - pad) * inv_scale))
    ox2 = min(width, int((x2 + pad) * inv_scale))
    oy2 = min(height, int((y2 + pad) * inv_scale))
    return OcrRegion(
        x=ox,
        y=oy,
        width=max(1, ox2 - ox),
        height=max(1, oy2 - oy),
        text_density=0.0,
        edge_density=0.0,
    )


def _dominant_projection_range(values, *, threshold: float, min_length: int) -> tuple[int, int] | None:
    best: tuple[int, int] | None = None
    start: int | None = None
    for index, value in enumerate(values):
        if value > threshold and start is None:
            start = index
        is_end = value <= threshold or index == len(values) - 1
        if is_end and start is not None:
            end = index if value <= threshold else index + 1
            if end - start >= min_length and (best is None or end - start > best[1] - best[0]):
                best = (start, end)
            start = None
    return best


def detect_text_regions(image, config: VisionConfig) -> list[OcrRegion]:
    import cv2

    height, width = image.shape[:2]
    if height <= 0 or width <= 0:
        return []

    scale = min(1.0, 900.0 / max(width, height))
    small = cv2.resize(image, (max(1, int(width * scale)), max(1, int(height * scale)))) if scale < 1.0 else image
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    grad_x = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3)
    grad_x = cv2.convertScaleAbs(grad_x)
    _, binary = cv2.threshold(grad_x, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 3))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    closed = cv2.dilate(closed, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3)), iterations=1)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    regions: list[OcrRegion] = []
    inv_scale = 1.0 / scale
    small_area = gray.shape[0] * gray.shape[1]
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w < 24 or h < 8:
            continue
        area = w * h
        if area < 250:
            continue
        aspect = w / float(max(h, 1))
        if aspect < 1.4 and w < 120:
            continue
        crop = binary[y : y + h, x : x + w]
        text_density = float(cv2.countNonZero(crop) / max(area, 1))
        if text_density < config.ocr_roi_min_text_density:
            continue
        if area / max(small_area, 1) > 0.18 and text_density < 0.12:
            continue
        edge_density = float(cv2.countNonZero(crop) / max(gray.size, 1))
        pad = config.ocr_roi_padding
        ox = max(0, int((x - pad) * inv_scale))
        oy = max(0, int((y - pad) * inv_scale))
        ox2 = min(width, int((x + w + pad) * inv_scale))
        oy2 = min(height, int((y + h + pad) * inv_scale))
        if oy > height * 0.82 and (oy2 - oy) < height * 0.2:
            continue
        regions.append(
            OcrRegion(
                x=ox,
                y=oy,
                width=max(1, ox2 - ox),
                height=max(1, oy2 - oy),
                text_density=round(text_density, 6),
                edge_density=round(edge_density, 6),
            )
        )

    merged = _merge_regions(regions)
    return sorted(merged, key=lambda item: item.width * item.height, reverse=True)[: config.ocr_roi_max_regions]


def _merge_regions(regions: list[OcrRegion]) -> list[OcrRegion]:
    merged: list[OcrRegion] = []
    for region in sorted(regions, key=lambda item: (item.y, item.x)):
        match_index = next((index for index, item in enumerate(merged) if _overlaps_or_near(item, region)), -1)
        if match_index < 0:
            merged.append(region)
            continue
        merged[match_index] = _union_region(merged[match_index], region)
    return merged


def _overlaps_or_near(left: OcrRegion, right: OcrRegion) -> bool:
    left_right = left.x + left.width
    right_right = right.x + right.width
    left_bottom = left.y + left.height
    right_bottom = right.y + right.height
    horizontal_gap = max(right.x - left_right, left.x - right_right, 0)
    vertical_gap = max(right.y - left_bottom, left.y - right_bottom, 0)
    return horizontal_gap <= 16 and vertical_gap <= 10


def _union_region(left: OcrRegion, right: OcrRegion) -> OcrRegion:
    x1 = min(left.x, right.x)
    y1 = min(left.y, right.y)
    x2 = max(left.x + left.width, right.x + right.width)
    y2 = max(left.y + left.height, right.y + right.height)
    return OcrRegion(
        x=x1,
        y=y1,
        width=x2 - x1,
        height=y2 - y1,
        text_density=max(left.text_density, right.text_density),
        edge_density=max(left.edge_density, right.edge_density),
    )
