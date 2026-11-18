
#!/usr/bin/env python3
"""A.R.G.O. V1 - Tracking radar osservativo 2D."""

from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy.optimize import linear_sum_assignment
from pyproj import CRS, Transformer

import radar_dpc

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
PUBLIC = ROOT / "docs" / "data"

THRESHOLD = 25.0
MIN_PIXELS = 8
MAX_CELLS = 40
MAX_GAP_MIN = 70
MAX_SPEED_KMH = 90.0
MAX_TRACK_POINTS = 24
MAX_TURN_DEG = 100.0
MAX_DBZ_CHANGE = 20.0
REJECT = 1e9


def timestamp(dt):
    return dt.astimezone(timezone.utc).isoformat().replace(
        "+00:00", "Z"
    )


def parse_time(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(data, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8"
    )
    temporary.replace(path)


def load_history():
    path = DATA / "iride_history.json"
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
        if result.get("schema_version") == 2:
            if isinstance(result.get("tracks"), dict):
                return result
    except (OSError, ValueError, TypeError):
        pass
    return {"schema_version": 2, "tracks": {}}


def distance_km(a, b):
    r = math.pi / 180.0
    p1, p2 = a["lat"] * r, b["lat"] * r
    dp = (b["lat"] - a["lat"]) * r
    dl = (b["lon"] - a["lon"]) * r
    q = (
        math.sin(dp / 2) ** 2
        + math.cos(p1) * math.cos(p2)
        * math.sin(dl / 2) ** 2
    )
    return 12742.0176 * math.asin(
        min(1.0, math.sqrt(max(0.0, q)))
    )


def bearing(a, b):
    r = math.pi / 180.0
    p1, p2 = a["lat"] * r, b["lat"] * r
    dl = (b["lon"] - a["lon"]) * r
    y = math.sin(dl) * math.cos(p2)
    x = (
        math.cos(p1) * math.sin(p2)
        - math.sin(p1) * math.cos(p2) * math.cos(dl)
    )
    return (math.degrees(math.atan2(y, x)) + 360) % 360


def angle_difference(a, b):
    return abs((a - b + 180) % 360 - 180)


def segment(a, b):
    try:
        minutes = (
            parse_time(b["timestamp"])
            - parse_time(a["timestamp"])
        ).total_seconds() / 60
        if not 5 <= minutes <= MAX_GAP_MIN:
            return None

        km = distance_km(a, b)
        speed = km * 60 / minutes

        if speed > MAX_SPEED_KMH:
            return None

        return {
            "start": [a["lat"], a["lon"]],
            "end": [b["lat"], b["lon"]],
            "start_time": a["timestamp"],
            "end_time": b["timestamp"],
            "distance_km": round(km, 3),
            "speed_kmh": round(speed, 2),
            "bearing_deg": round(bearing(a, b), 2),
            "interval_minutes": round(minutes, 2),
            "quality": "candidate_observed"
        }
    except (KeyError, ValueError, TypeError):
        return None


def detect_cells(radar, grid, inverse, origin):
    mask = np.isfinite(radar) & (radar >= THRESHOLD)
    labels, _ = ndimage.label(
        mask, structure=np.ones((3, 3), dtype=np.uint8)
    )
    cells = []
    ox, oy = origin
    sx = float(grid["xscale_m"])
    sy = float(grid["yscale_m"])

    for index, slices in enumerate(
        ndimage.find_objects(labels), start=1
    ):
        if slices is None:
            continue
        ys, xs = slices
        component = labels[slices] == index
        area = int(component.sum())
        if area < MIN_PIXELS:
            continue

        yy, xx = np.nonzero(component)
        values = radar[slices][yy, xx]
        weights = np.maximum(values - THRESHOLD + 1, 1)

        cx = xs.start + float(np.average(xx, weights=weights))
        cy = ys.start + float(np.average(yy, weights=weights))

        gx = ox + (cx + 0.5) * sx
        gy = oy + (grid["ysize"] - 0.5 - cy) * sy
        lon, lat = inverse.transform(gx, gy)

        if not (
            np.isfinite(lat) and np.isfinite(lon)
            and -90 <= lat <= 90 and -180 <= lon <= 180
        ):
            continue

        cells.append({
            "id": f"ARGO-{index:05d}",
            "lat": round(float(lat), 5),
            "lon": round(float(lon), 5),
            "max_dbz": round(float(values.max()), 1),
            "mean_dbz": round(float(values.mean()), 1),
            "area_pixels": area,
            "approx_area_km2": round(area * sx * sy / 1e6, 2)
        })

    cells.sort(
        key=lambda c: (c["max_dbz"], c["area_pixels"]),
        reverse=True
    )
    return cells[:MAX_CELLS]


def assign_tracks(cells, scan_time, history):
    tracks = history.get("tracks", {})
    active = {}

    for track_id, track in tracks.items():
        obs = track.get("observations", [])
        if not obs:
            continue
        try:
            age = (
                scan_time - parse_time(obs[-1]["timestamp"])
            ).total_seconds() / 60
            if 0 <= age <= MAX_GAP_MIN:
                active[track_id] = track
        except (ValueError, TypeError, KeyError):
            continue

    ids = list(active)
    costs = np.full((len(ids), len(cells)), REJECT)
    current_time = timestamp(scan_time)

    for i, track_id in enumerate(ids):
        obs = active[track_id]["observations"]
        last = obs[-1]
        age = (
            scan_time - parse_time(last["timestamp"])
        ).total_seconds() / 60

        for j, cell in enumerate(cells):
            if age == 0:
                continue

            candidate = {
                **cell,
                "timestamp": current_time
            }
            movement = segment(last, candidate)
            if movement is None:
                continue

            dbz_change = abs(
                cell["max_dbz"] - last["max_dbz"]
            )
            if dbz_change > MAX_DBZ_CHANGE:
                continue

            turn_penalty = 0.0
            if len(obs) >= 2:
                previous = segment(obs[-2], last)
                if previous:
                    turn = angle_difference(
                        previous["bearing_deg"],
                        movement["bearing_deg"]
                    )
                    if turn > MAX_TURN_DEG:
                        continue
                    turn_penalty = turn * 0.1

            costs[i, j] = (
                movement["distance_km"]
                + dbz_change * 0.5
                + turn_penalty
            )

    assignments = {}
    if len(ids) and len(cells):
        rows, columns = linear_sum_assignment(costs)
        for i, j in zip(rows, columns):
            if costs[i, j] < REJECT:
                assignments[int(j)] = ids[int(i)]

    updated = {}
    for j, cell in enumerate(cells):
        track_id = assignments.get(j)

        if track_id is None:
            track_id = f"ARGO-{scan_time:%Y%m%dT%H%M}-{j:03d}"
            suffix = 1
            base = track_id
            while track_id in tracks or track_id in updated:
                track_id = f"{base}-{suffix:02d}"
                suffix += 1
            obs = []
        else:
            obs = list(active[track_id]["observations"])

        point = {
            "timestamp": current_time,
            "lat": cell["lat"],
            "lon": cell["lon"],
            "max_dbz": cell["max_dbz"]
        }

        if obs and obs[-1]["timestamp"] == current_time:
            obs[-1] = point
        else:
            obs.append(point)

        obs = obs[-MAX_TRACK_POINTS:]
        updated[track_id] = {"observations": obs}

        valid_segments = []
        for k in range(1, len(obs)):
            s = segment(obs[k - 1], obs[k])
            if s is not None:
                valid_segments.append(s)

        latest = (
            valid_segments[-1]
            if len(obs) >= 2 and valid_segments
            and valid_segments[-1]["end_time"] == current_time
            else None
        )

        real_vector = (
            [latest["start"], latest["end"]]
            if latest else None
        )

        cell["track_id"] = track_id
        cell["track"] = {
            "track_id": track_id,
            "observations": obs,
            "real_vector": real_vector,
            "speed_kmh": (
                latest["speed_kmh"] if latest else None
            ),
            "observed_interval_minutes": (
                latest["interval_minutes"] if latest else None
            ),
            "motion_quality": (
                "candidate_observed"
                if latest else "insufficient_history"
            ),
            "segments": valid_segments,
            "forecast_status": "disabled_pending_validation"
        }
        cell["vectors"] = {
            "real": real_vector,
            "predictive_30m": None,
            "predictive_60m": None,
            "predictive_180m": None,
            "forecast_status": "disabled_pending_validation"
        }

    history["tracks"] = updated
    history["source_id"] = "DPC_VMI"
    return cells, history


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    radar, grid, inverse, origin, scan_time, source = (
        radar_dpc.acquire(DATA / "raw")
    )

    if scan_time.tzinfo is None:
        raise ValueError("Timestamp radar privo di fuso orario")

    age = (
        datetime.now(timezone.utc) - scan_time
    ).total_seconds() / 60
    if not -5 <= age <= 90:
        raise ValueError(
            f"Scansione radar obsoleta: {age:.1f} minuti"
        )

    cells = detect_cells(radar, grid, inverse, origin)
    history = load_history()

    if history.get("source_id", "DPC_VMI") != "DPC_VMI":
        history = {"schema_version": 2, "tracks": {}}

    cells, history = assign_tracks(cells, scan_time, history)

    payload = {
        "schema_version": 6,
        "product": "A.R.G.O. - Radar Cell Tracking",
        "status": "ok",
        "timestamp": timestamp(scan_time),
        "generated_at": timestamp(datetime.now(timezone.utc)),
        "source": source,
        "grid": grid,
        "processing": {
            "threshold_dbz": THRESHOLD,
            "tracking": "two-observation candidate tracking",
            "forecast": "disabled_pending_validation",
            "vertical_data": "not_used"
        },
        "cell_count": len(cells),
        "cells": cells
    }

    save(DATA / "iride.json", payload)
    save(DATA / "iride_history.json", history)
    save(PUBLIC / "iride.json", payload)
    save(PUBLIC / "iride_history.json", history)

    print(
        f"ARGO OK: {len(cells)} celle, "
        f"scansione {timestamp(scan_time)}"
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ARGO ERROR: {exc}", file=sys.stderr)
        raise
  
