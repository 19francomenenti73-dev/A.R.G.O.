
#!/usr/bin/env python3
"""A.R.G.O. V2.3 - tracking radar osservativo 2D."""
import io
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import requests
from PIL import Image
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DATA.mkdir(parents=True, exist_ok=True)

API = "https://api.rainviewer.com/public/weather-maps.json"
ZOOM, TILE, COLOR, OPTIONS = 6, 256, 2, "0_0"
LAT_MIN, LAT_MAX = 34.0, 48.5
LON_MIN, LON_MAX = 5.0, 20.0
MIN_PIXELS = 25
MAX_CELLS = 100
MAX_SPEED = 130.0
MIN_DT, MAX_DT = 4, 30
MAX_AGE = 45
HISTORY_HOURS = 6
RECONNECT_KM = 3.0
MIN_VECTOR_KM = 0.5
MAX_TURN = 105.0

SESSION = requests.Session()
SESSION.headers["User-Agent"] = "ARGO-NowcastingItalia/2.3"


def stamp(ts):
    return datetime.fromtimestamp(
        int(ts), timezone.utc
    ).isoformat().replace("+00:00", "Z")


def epoch(value):
    return datetime.fromisoformat(
        value.replace("Z", "+00:00")
    ).timestamp()


def mx(lon):
    return (lon + 180) / 360 * 2 ** ZOOM


def my(lat):
    lat = max(-85.0511, min(85.0511, lat))
    r = math.radians(lat)
    return (1 - math.asinh(math.tan(r)) / math.pi) / 2 * 2 ** ZOOM


def geo(px, py, x0, y0):
    n = 2 ** ZOOM
    x = x0 + px / TILE
    y = y0 + py / TILE
    lat = math.degrees(
        math.atan(math.sinh(math.pi * (1 - 2 * y / n)))
    )
    return lat, x / n * 360 - 180


def km(a, b):
    p1 = math.radians(a["lat"])
    p2 = math.radians(b["lat"])
    dp = p2 - p1
    dl = math.radians(b["lon"] - a["lon"])
    h = (
        math.sin(dp / 2) ** 2
        + math.cos(p1) * math.cos(p2)
        * math.sin(dl / 2) ** 2
    )
    return 12742 * math.asin(min(1, math.sqrt(h)))


def bearing(a, b):
    p1 = math.radians(a["lat"])
    p2 = math.radians(b["lat"])
    dl = math.radians(b["lon"] - a["lon"])
    y = math.sin(dl) * math.cos(p2)
    x = (
        math.cos(p1) * math.sin(p2)
        - math.sin(p1) * math.cos(p2) * math.cos(dl)
    )
    return (math.degrees(math.atan2(y, x)) + 360) % 360


def angle_difference(a, b):
    return abs((a - b + 180) % 360 - 180)


def frames():
    response = SESSION.get(API, timeout=25)
    response.raise_for_status()
    obj = response.json()
    host = obj["host"].rstrip("/")
    if not host.startswith("https://"):
        raise RuntimeError("Host RainViewer non HTTPS")

    past = sorted(
        obj["radar"]["past"], key=lambda f: f["time"]
    )
    if len(past) < 2:
        raise RuntimeError("Servono due scansioni")

    a, b = past[-2:]
    dt = (b["time"] - a["time"]) / 60
    age = (
        datetime.now(timezone.utc).timestamp()
        - b["time"]
    ) / 60

    if not MIN_DT <= dt <= MAX_DT:
        raise RuntimeError("Intervallo radar anomalo")
    if not -5 <= age <= MAX_AGE:
        raise RuntimeError("Scansione radar obsoleta")
    return host, a, b, dt


def bounds():
    return (
        math.floor(mx(LON_MIN)),
        math.floor(mx(LON_MAX)),
        math.floor(my(LAT_MAX)),
        math.floor(my(LAT_MIN))
    )


def mosaic(host, frame):
    x0, x1, y0, y1 = bounds()
    image = np.zeros(
        ((y1 - y0 + 1) * TILE,
         (x1 - x0 + 1) * TILE, 4),
        dtype=np.uint8
    )

    for y in range(y0, y1 + 1):
        for x in range(x0, x1 + 1):
            url = (
                f"{host}{frame['path']}/{TILE}/"
                f"{ZOOM}/{x}/{y}/{COLOR}/{OPTIONS}.png"
            )
            response = SESSION.get(url, timeout=20)
            response.raise_for_status()
            with Image.open(io.BytesIO(response.content)) as im:
                if im.size != (TILE, TILE):
                    raise RuntimeError("Tile non valido")
                tile = np.asarray(im.convert("RGBA"))

            py = (y - y0) * TILE
            px = (x - x0) * TILE
            image[py:py + TILE, px:px + TILE] = tile

    return image, x0, y0


def detect(image, x0, y0):
    hsv = cv2.cvtColor(
        image[:, :, :3], cv2.COLOR_RGB2HSV
    )
    sat, val = hsv[:, :, 1], hsv[:, :, 2]
    mask = (
        (image[:, :, 3] >= 100)
        & (sat >= 65)
        & (val >= 70)
    ).astype(np.uint8)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN, kernel
    )
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, kernel
    )

    count, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )
    )
    cells = []

    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < MIN_PIXELS:
            continue

        yy, xx = np.where(labels == label)
        weights = np.maximum(
            1.0, val[yy, xx].astype(float)
        )
        px = float(np.average(xx, weights=weights))
        py = float(np.average(yy, weights=weights))
        lat, lon = geo(px, py, x0, y0)

        if not (
            LAT_MIN <= lat <= LAT_MAX
            and LON_MIN <= lon <= LON_MAX
        ):
            continue

        cells.append({
            "lat": round(lat, 5),
            "lon": round(lon, 5),
            "area_px": area,
            "intensity_index": round(
                float(weights.mean() / 255), 3
            ),
            "max_dbz": None
        })

    cells.sort(
        key=lambda c: c["area_px"], reverse=True
    )
    return cells[:MAX_CELLS]


def associate(old, new, dt):
    if not old or not new:
        return []

    maxdist = MAX_SPEED * dt / 60
    cost = np.full(
        (len(old), len(new)), 1e8
    )

    for i, a in enumerate(old):
        for j, b in enumerate(new):
            distance = km(a, b)
            ratio = (
                max(a["area_px"], b["area_px"])
                / max(1, min(a["area_px"], b["area_px"]))
            )
            if distance > maxdist or ratio > 3:
                continue

            cost[i, j] = (
                distance / maxdist
                + 0.25 * abs(math.log(ratio))
                + 0.4 * abs(
                    a["intensity_index"]
                    - b["intensity_index"]
                )
            )

    rows, cols = linear_sum_assignment(cost)
    result = []

    for i, j in zip(rows, cols):
        if cost[i, j] >= 1e8:
            continue

        row = sorted(v for v in cost[i] if v < 1e8)
        col = sorted(v for v in cost[:, j] if v < 1e8)

        if len(row) > 1 and row[1] - row[0] < 0.12:
            continue
        if len(col) > 1 and col[1] - col[0] < 0.12:
            continue

        result.append((int(i), int(j)))

    return result


def load_history():
    path = DATA / "argo_history.json"
    if not path.exists():
        return {"tracks": {}, "next_id": 1}

    try:
        obj = json.loads(
            path.read_text(encoding="utf-8")
        )
        next_id = max(1, int(obj.get("next_id", 1)))

        for key in obj.get("tracks", {}):
            if key.startswith("ARGO-") and key[5:].isdigit():
                next_id = max(
                    next_id, int(key[5:]) + 1
                )

        # Non importare collegamenti V2.2 sospetti.
        if obj.get("schema_version") != 5:
            return {"tracks": {}, "next_id": next_id}

        if obj.get("source_id") != "RAINVIEWER":
            return {"tracks": {}, "next_id": next_id}

        return {
            "tracks": obj.get("tracks", {}),
            "next_id": next_id
        }
    except (OSError, ValueError, TypeError, KeyError):
        return {"tracks": {}, "next_id": 1}


def write(name, obj):
    path = DATA / name
    temp = path.with_suffix(".tmp")
    temp.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    temp.replace(path)


def main():
    host, older, newer, dt = frames()
    t0, t1 = stamp(older["time"]), stamp(newer["time"])
    history = load_history()
    tracks = history["tracks"]
    next_id = history["next_id"]

    # Non rielaborare una scansione già pubblicata.
    current_path = DATA / "argo.json"
    if current_path.exists():
        try:
            current = json.loads(
                current_path.read_text(encoding="utf-8")
            )
            if (
                current.get("schema_version") == 5
                and current.get("timestamp") == t1
            ):
                print("ARGO: scansione già elaborata")
                return
        except (OSError, ValueError):
            pass

    old_image, x0, y0 = mosaic(host, older)
    new_image, _, _ = mosaic(host, newer)
    old = detect(old_image, x0, y0)
    cells = detect(new_image, x0, y0)
    pairs = associate(old, cells, dt)

    cutoff = newer["time"] - HISTORY_HOURS * 3600
    tracks = {
        key: track
        for key, track in tracks.items()
        if track.get("observations")
        and epoch(
            track["observations"][-1]["timestamp"]
        ) >= cutoff
    }

    def allocate():
        nonlocal next_id
        key = f"ARGO-{next_id:06d}"
        next_id += 1
        return key

    # Associazione ESCLUSIVA fra il frame T0
    # e l'ultima osservazione della traccia.
    candidates = []
    for i, old_cell in enumerate(old):
        for key, track in tracks.items():
            observations = track["observations"]
            last = observations[-1]

            if last["timestamp"] != t0:
                continue

            d = km(old_cell, last)
            if d <= RECONNECT_KM:
                candidates.append((d, i, key))

    candidates.sort()
    old_to_track = {}
    occupied_tracks = set()

    for distance, i, key in candidates:
        if i in old_to_track or key in occupied_tracks:
            continue

        # Se più tracce sono quasi equivalenti,
        # non forzare la continuità.
        alternatives = [
            item for item in candidates
            if item[1] == i and item[2] != key
        ]
        if alternatives and (
            alternatives[0][0] - distance < 1.0
        ):
            continue

        old_to_track[i] = key
        occupied_tracks.add(key)

    matched = set()
    observed_count = 0
    rejected_turns = 0

    for i, j in pairs:
        a, b = old[i], cells[j]
        distance = km(a, b)
        speed = distance * 60 / dt

        if distance < MIN_VECTOR_KM:
            continue

        key = old_to_track.get(i)
        previous = (
            tracks[key]["observations"]
            if key is not None else []
        )

        if len(previous) >= 2:
            previous_bearing = bearing(
                previous[-2], previous[-1]
            )
            new_bearing = bearing(a, b)

            if angle_difference(
                previous_bearing, new_bearing
            ) > MAX_TURN:
                key = None
                rejected_turns += 1

        if key is None:
            key = allocate()
            tracks[key] = {
                "observations": [{
                    "lat": a["lat"],
                    "lon": a["lon"],
                    "timestamp": t0
                }]
            }

        observations = tracks[key]["observations"]
        if observations[-1]["timestamp"] != t0:
            continue

        observations.append({
            "lat": b["lat"],
            "lon": b["lon"],
            "timestamp": t1
        })

        b["id"] = key
        b["track_id"] = key
        b["motion"] = {
            "type": "observed",
            "speed_kmh": round(speed, 2),
            "direction_deg": round(bearing(a, b), 1),
            "distance_km": round(distance, 2),
            "interval_min": dt
        }
        matched.add(j)
        observed_count += 1

    for j, cell in enumerate(cells):
        if j not in matched:
            key = allocate()
            tracks[key] = {
                "observations": [{
                    "lat": cell["lat"],
                    "lon": cell["lon"],
                    "timestamp": t1
                }]
            }
            cell["id"] = key
            cell["track_id"] = key
            cell["motion"] = None

        cell["vectors"] = {}

    payload = {
        "schema_version": 5,
        "timestamp": t1,
        "generated_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "source": {
            "provider": "RainViewer",
            "attribution": "Weather data by RainViewer",
            "url": "https://www.rainviewer.com/",
            "method": "Segmentazione cromatica 2D"
        },
        "radar_tiles": {
            "host": host,
            "path": newer["path"],
            "zoom_max": 7,
            "color": COLOR,
            "options": OPTIONS
        },
        "cell_count": len(cells),
        "observed_vectors": observed_count,
        "cells": cells
    }

    write("argo_history.json", {
        "schema_version": 5,
        "source_id": "RAINVIEWER",
        "next_id": next_id,
        "tracks": tracks
    })
    write("argo.json", payload)
    write("argo_quality.json", {
        "timestamp": t1,
        "previous_timestamp": t0,
        "interval_minutes": dt,
        "previous_cells": len(old),
        "current_cells": len(cells),
        "matched_cells": observed_count,
        "rejected_turns": rejected_turns,
        "quality_status": "experimental"
    })

    print(
        f"ARGO V2.3: {len(cells)} centroidi, "
        f"{observed_count} vettori osservati"
    )


if __name__ == "__main__":
    main()
    
