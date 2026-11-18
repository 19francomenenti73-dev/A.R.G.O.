
#!/usr/bin/env python3
"""A.R.G.O. V2 - Radar RainViewer, tracking osservativo 2D."""

import io
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import requests
from PIL import Image
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT / "data"
PUBLIC.mkdir(parents=True, exist_ok=True)

API = "https://api.rainviewer.com/public/weather-maps.json"
ZOOM = 6
TILE_SIZE = 256
COLOR = 2
OPTIONS = "0_0"

# Italia, Alpi e Mediterraneo centrale
LAT_MIN, LAT_MAX = 34.0, 48.5
LON_MIN, LON_MAX = 5.0, 20.0

MIN_PIXELS = 12
MAX_CELLS = 100
MAX_SPEED = 130.0
MIN_MINUTES = 4
MAX_MINUTES = 30
MAX_FRAME_AGE = 45
HISTORY_HOURS = 6

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "ARGO-NowcastingItalia/2.0"
})


def utc(ts):
    return datetime.fromtimestamp(
        int(ts), timezone.utc
    ).isoformat().replace("+00:00", "Z")


def mercator_x(lon, z):
    return (lon + 180.0) / 360.0 * (2 ** z)


def mercator_y(lat, z):
    lat = max(-85.0511, min(85.0511, lat))
    r = math.radians(lat)
    return (
        1 - math.asinh(math.tan(r)) / math.pi
    ) / 2 * (2 ** z)


def pixel_latlon(px, py, x0, y0):
    n = 2 ** ZOOM
    tx = x0 + px / TILE_SIZE
    ty = y0 + py / TILE_SIZE
    lon = tx / n * 360 - 180
    lat = math.degrees(
        math.atan(math.sinh(math.pi * (1 - 2 * ty / n)))
    )
    return float(lat), float(lon)


def distance_km(a, b):
    r = 6371.0
    p1, p2 = map(math.radians, (a["lat"], b["lat"]))
    dp = p2 - p1
    dl = math.radians(b["lon"] - a["lon"])
    h = (
        math.sin(dp / 2) ** 2
        + math.cos(p1) * math.cos(p2)
        * math.sin(dl / 2) ** 2
    )
    return 2 * r * math.asin(min(1, math.sqrt(h)))


def direction(a, b):
    p1 = math.radians(a["lat"])
    p2 = math.radians(b["lat"])
    dl = math.radians(b["lon"] - a["lon"])
    y = math.sin(dl) * math.cos(p2)
    x = (
        math.cos(p1) * math.sin(p2)
        - math.sin(p1) * math.cos(p2) * math.cos(dl)
    )
    return (math.degrees(math.atan2(y, x)) + 360) % 360


def get_frames():
    response = SESSION.get(API, timeout=25)
    response.raise_for_status()
    payload = response.json()

    host = payload["host"].rstrip("/")
    if not host.startswith("https://"):
        raise RuntimeError("Host RainViewer non HTTPS")

    frames = sorted(
        payload["radar"]["past"],
        key=lambda frame: frame["time"]
    )
    if len(frames) < 2:
        raise RuntimeError("Meno di due frame radar")

    first, second = frames[-2:]
    minutes = (
        second["time"] - first["time"]
    ) / 60

    if not MIN_MINUTES <= minutes <= MAX_MINUTES:
        raise RuntimeError(
            f"Intervallo scansioni non valido: {minutes}"
        )

    now = datetime.now(timezone.utc).timestamp()
    age = (now - second["time"]) / 60
    if age < -5 or age > MAX_FRAME_AGE:
        raise RuntimeError(
            f"Frame radar obsoleto: {age:.1f} minuti"
        )

    return host, first, second, minutes


def tile_bounds():
    x0 = math.floor(mercator_x(LON_MIN, ZOOM))
    x1 = math.floor(mercator_x(LON_MAX, ZOOM))
    y0 = math.floor(mercator_y(LAT_MAX, ZOOM))
    y1 = math.floor(mercator_y(LAT_MIN, ZOOM))
    return x0, x1, y0, y1


def get_tile(host, frame, x, y):
    url = (
        f"{host}{frame['path']}/{TILE_SIZE}/"
        f"{ZOOM}/{x}/{y}/{COLOR}/{OPTIONS}.png"
    )
    response = SESSION.get(url, timeout=20)
    response.raise_for_status()
    image = Image.open(io.BytesIO(response.content))
    image.load()
    if image.size != (TILE_SIZE, TILE_SIZE):
        raise RuntimeError("Dimensioni tile non valide")
    return np.asarray(image.convert("RGBA"))


def build_mosaic(host, frame):
    x0, x1, y0, y1 = tile_bounds()
    width = (x1 - x0 + 1) * TILE_SIZE
    height = (y1 - y0 + 1) * TILE_SIZE
    mosaic = np.zeros(
        (height, width, 4), dtype=np.uint8
    )

    failures = 0
    for y in range(y0, y1 + 1):
        for x in range(x0, x1 + 1):
            try:
                tile = get_tile(host, frame, x, y)
                py = (y - y0) * TILE_SIZE
                px = (x - x0) * TILE_SIZE
                mosaic[
                    py:py + TILE_SIZE,
                    px:px + TILE_SIZE
                ] = tile
            except (
                requests.RequestException,
                OSError, ValueError
            ) as exc:
                failures += 1
                print(f"Tile {x}/{y}: {exc}")

    if failures:
        raise RuntimeError(
            f"Mosaico incompleto: {failures} tile mancanti"
        )

    return mosaic, x0, y0


def detect(mosaic, x0, y0):
    rgb = mosaic[:, :, :3]
    alpha = mosaic[:, :, 3]

    # Segmentazione cromatica, NON conversione in dBZ.
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    mask = (
        (alpha >= 100)
        & (saturation >= 65)
        & (value >= 70)
    ).astype(np.uint8)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN, kernel
    )
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, kernel
    )

    count, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )
    )

    cells = []
    for label in range(1, count):
        area = int(
            stats[label, cv2.CC_STAT_AREA]
        )
        if area < MIN_PIXELS:
            continue

        px, py = centroids[label]
        lat, lon = pixel_latlon(px, py, x0, y0)

        if not (
            LAT_MIN <= lat <= LAT_MAX
            and LON_MIN <= lon <= LON_MAX
        ):
            continue

        component = labels == label
        intensity = float(
            np.mean(value[component])
        )

        cells.append({
            "lat": round(lat, 5),
            "lon": round(lon, 5),
            "area_px": area,
            "intensity_index": round(
                intensity / 255, 3
            ),
            "max_dbz": None
        })

    cells.sort(
        key=lambda c: c["area_px"],
        reverse=True
    )
    return cells[:MAX_CELLS]


def associate(previous, current, minutes):
    if not previous or not current:
        return []

    costs = np.full(
        (len(previous), len(current)),
        1e8, dtype=float
    )
    max_distance = MAX_SPEED * minutes / 60

    for i, old in enumerate(previous):
        for j, new in enumerate(current):
            km = distance_km(old, new)
            if km > max_distance:
                continue

            ratio = (
                max(old["area_px"], new["area_px"])
                / max(
                    1,
                    min(
                        old["area_px"],
                        new["area_px"]
                    )
                )
            )
            if ratio > 3.0:
                continue

            intensity_delta = abs(
                old["intensity_index"]
                - new["intensity_index"]
            )
            costs[i, j] = (
                km / max_distance
                + 0.25 * abs(math.log(ratio))
                + 0.4 * intensity_delta
            )

    rows, cols = linear_sum_assignment(costs)
    matches = []

    for i, j in zip(rows, cols):
        if costs[i, j] >= 1e8:
            continue

        # Escludi associazioni ambigue.
        row_candidates = sorted(
            v for v in costs[i] if v < 1e8
        )
        col_candidates = sorted(
            v for v in costs[:, j] if v < 1e8
        )
        if (
            len(row_candidates) > 1
            and row_candidates[1]
            - row_candidates[0] < 0.12
        ):
            continue
        if (
            len(col_candidates) > 1
            and col_candidates[1]
            - col_candidates[0] < 0.12
        ):
            continue

        matches.append((int(i), int(j)))

    return matches


def load_history():
    path = PUBLIC / "argo_history.json"
    if not path.exists():
        return {"tracks": {}}
    try:
        data = json.loads(
            path.read_text(encoding="utf-8")
        )
        if data.get("source_id") != "RAINVIEWER":
            return {"tracks": {}}
        return data
    except (ValueError, OSError):
        return {"tracks": {}}


def save(name, data):
    path = PUBLIC / name
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(
            data, ensure_ascii=False, indent=2
        ),
        encoding="utf-8"
    )
    temporary.replace(path)


def main():
    host, older, newer, minutes = get_frames()
    mosaic_old, x0, y0 = build_mosaic(
        host, older
    )
    mosaic_new, _, _ = build_mosaic(
        host, newer
    )

    old_cells = detect(mosaic_old, x0, y0)
    cells = detect(mosaic_new, x0, y0)
    matches = associate(
        old_cells, cells, minutes
    )

    history = load_history()
    tracks = history.get("tracks", {})
    now_ts = int(newer["time"])
    cutoff = now_ts - HISTORY_HOURS * 3600

    tracks = {
        key: track
        for key, track in tracks.items()
        if track.get("observations")
        and datetime.fromisoformat(
            track["observations"][-1]["timestamp"]
            .replace("Z", "+00:00")
        ).timestamp() >= cutoff
    }

    used = set()
    for i, j in matches:
        old = old_cells[i]
        new = cells[j]
        km = distance_km(old, new)
        speed = km * 60 / minutes
        bearing = direction(old, new)

        # Riprende una traccia soltanto se il punto
        # precedente coincide con il frame T1.
        track_id = None
        for key, track in tracks.items():
            obs = track["observations"][-1]
            if obs["timestamp"] != utc(
                older["time"]
            ):
                continue
            if distance_km(obs, old) < 2:
                track_id = key
                break

        if track_id is None:
            track_id = (
                f"ARGO-{older['time']}-{i}"
            )
            tracks[track_id] = {
                "observations": [{
                    "lat": old["lat"],
                    "lon": old["lon"],
                    "timestamp": utc(older["time"])
                }]
            }

        observation = {
            "lat": new["lat"],
            "lon": new["lon"],
            "timestamp": utc(newer["time"])
        }

        if track_id in used:
            continue
        used.add(track_id)

        obs = tracks[track_id]["observations"]
        if obs[-1]["timestamp"] != observation["timestamp"]:
            obs.append(observation)

        new["track_id"] = track_id
        new["track"] = tracks[track_id]
        new["vectors"] = {}
        new["motion"] = {
            "speed_kmh": round(speed, 2),
            "direction_deg": round(bearing, 1),
            "distance_km": round(km, 2),
            "interval_min": minutes,
            "type": "observed"
        }

    for index, cell in enumerate(cells):
        cell["id"] = f"CELL-{index + 1}"
        cell.setdefault("track_id", None)
        cell.setdefault("track", {
            "observations": []
        })
        cell.setdefault("vectors", {})
        cell.setdefault("motion", None)

    source = {
        "provider": "RainViewer",
        "attribution": "Weather data by RainViewer",
        "url": "https://www.rainviewer.com/",
        "product": "Radar composite PNG",
        "method": "Color segmentation, not raw dBZ"
    }

    payload = {
        "schema_version": 3,
        "timestamp": utc(newer["time"]),
        "generated_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "source": source,
        "radar_tiles": {
            "host": host,
            "path": newer["path"],
            "zoom_max": 7,
            "color": COLOR,
            "options": OPTIONS
        },
        "cell_count": len(cells),
        "observed_vectors": len(used),
        "cells": cells
    }

    save("argo.json", payload)
    save("argo_history.json", {
        "schema_version": 3,
        "source_id": "RAINVIEWER",
        "tracks": tracks
    })
    save("argo_quality.json", {
        "timestamp": utc(newer["time"]),
        "previous_timestamp": utc(older["time"]),
        "interval_minutes": minutes,
        "previous_cells": len(old_cells),
        "current_cells": len(cells),
        "matched_cells": len(used),
        "unmatched_cells": len(cells) - len(used),
        "method": "two-frame-observed",
        "quality_status": "experimental"
    })

    print(
        f"ARGO OK: {len(cells)} centroidi, "
        f"{len(used)} vettori osservati"
    )


if __name__ == "__main__":
    main()
    
