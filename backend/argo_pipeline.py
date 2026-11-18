#!/usr/bin/env python3
"""A.R.G.O. V2.2 - tracking 2D osservativo RainViewer."""
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
PUBLIC = ROOT / 'data'
PUBLIC.mkdir(parents=True, exist_ok=True)
API = 'https://api.rainviewer.com/public/weather-maps.json'
ZOOM, TILE, COLOR, OPTIONS = 6, 256, 2, '0_0'
LAT_MIN, LAT_MAX, LON_MIN, LON_MAX = 34., 48.5, 5., 20.
MIN_PIXELS, MAX_CELLS, MAX_SPEED = 25, 100, 130.
MIN_MINUTES, MAX_MINUTES, MAX_AGE = 4, 30, 45
HISTORY_HOURS, RECONNECT_KM = 6, 9.
SESSION = requests.Session()
SESSION.headers['User-Agent'] = 'ARGO-NowcastingItalia/2.2'


def stamp(ts):
    return datetime.fromtimestamp(int(ts), timezone.utc).isoformat().replace('+00:00', 'Z')


def epoch(s):
    return datetime.fromisoformat(s.replace('Z', '+00:00')).timestamp()


def mx(lon):
    return (lon + 180) / 360 * (2 ** ZOOM)


def my(lat):
    r = math.radians(max(-85.0511, min(85.0511, lat)))
    return (1 - math.asinh(math.tan(r)) / math.pi) / 2 * (2 ** ZOOM)


def geo(px, py, x0, y0):
    n = 2 ** ZOOM
    x, y = x0 + px / TILE, y0 + py / TILE
    return (math.degrees(math.atan(math.sinh(math.pi * (1 - 2*y/n)))),
            x/n*360 - 180)


def km(a, b):
    p1, p2 = math.radians(a['lat']), math.radians(b['lat'])
    dp = p2 - p1
    dl = math.radians(b['lon'] - a['lon'])
    h = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 12742 * math.asin(min(1., math.sqrt(h)))


def bearing(a, b):
    p1, p2 = math.radians(a['lat']), math.radians(b['lat'])
    dl = math.radians(b['lon'] - a['lon'])
    y = math.sin(dl)*math.cos(p2)
    x = math.cos(p1)*math.sin(p2)-math.sin(p1)*math.cos(p2)*math.cos(dl)
    return (math.degrees(math.atan2(y, x))+360) % 360


def frames():
    r = SESSION.get(API, timeout=25)
    r.raise_for_status()
    obj = r.json()
    host = obj['host'].rstrip('/')
    if not host.startswith('https://'):
        raise RuntimeError('Host RainViewer non HTTPS')
    past = sorted(obj['radar']['past'], key=lambda x: x['time'])
    if len(past) < 2:
        raise RuntimeError('Servono due scansioni')
    a, b = past[-2:]
    dt = (b['time'] - a['time'])/60
    age = (datetime.now(timezone.utc).timestamp() - b['time'])/60
    if not MIN_MINUTES <= dt <= MAX_MINUTES:
        raise RuntimeError(f'Intervallo radar anomalo: {dt} min')
    if not -5 <= age <= MAX_AGE:
        raise RuntimeError(f'Radar obsoleto: {age:.1f} min')
    return host, a, b, dt


def bounds():
    return (math.floor(mx(LON_MIN)), math.floor(mx(LON_MAX)),
            math.floor(my(LAT_MAX)), math.floor(my(LAT_MIN)))


def mosaic(host, frame):
    x0, x1, y0, y1 = bounds()
    image = np.zeros(((y1-y0+1)*TILE, (x1-x0+1)*TILE, 4), np.uint8)
    for y in range(y0, y1+1):
        for x in range(x0, x1+1):
            url = f"{host}{frame['path']}/{TILE}/{ZOOM}/{x}/{y}/{COLOR}/{OPTIONS}.png"
            r = SESSION.get(url, timeout=20)
            r.raise_for_status()
            with Image.open(io.BytesIO(r.content)) as im:
                if im.size != (TILE, TILE):
                    raise RuntimeError(f'Tile anomalo {x}/{y}')
                tile = np.asarray(im.convert('RGBA'))
            py, px = (y-y0)*TILE, (x-x0)*TILE
            image[py:py+TILE, px:px+TILE] = tile
    return image, x0, y0


def detect(image, x0, y0):
    hsv = cv2.cvtColor(image[:, :, :3], cv2.COLOR_RGB2HSV)
    sat, val = hsv[:, :, 1], hsv[:, :, 2]
    mask = ((image[:, :, 3] >= 100) & (sat >= 65) & (val >= 70)).astype('uint8')
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    cells = []
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < MIN_PIXELS:
            continue
        yy, xx = np.where(labels == label)
        weights = np.maximum(1., val[yy, xx].astype(float))
        px = float(np.average(xx, weights=weights))
        py = float(np.average(yy, weights=weights))
        lat, lon = geo(px, py, x0, y0)
        if not (LAT_MIN <= lat <= LAT_MAX and LON_MIN <= lon <= LON_MAX):
            continue
        cells.append({'lat': round(lat, 5), 'lon': round(lon, 5),
                      'area_px': area, 'intensity_index': round(float(weights.mean()/255), 3),
                      'max_dbz': None})
    return sorted(cells, key=lambda c: c['area_px'], reverse=True)[:MAX_CELLS]


def associate(old, new, dt):
    if not old or not new:
        return []
    maxdist = MAX_SPEED*dt/60
    cost = np.full((len(old), len(new)), 1e8)
    for i, a in enumerate(old):
        for j, b in enumerate(new):
            d = km(a, b)
            ratio = max(a['area_px'], b['area_px']) / max(1, min(a['area_px'], b['area_px']))
            if d > maxdist or ratio > 3:
                continue
            cost[i, j] = d/maxdist + .25*abs(math.log(ratio)) + .4*abs(a['intensity_index']-b['intensity_index'])
    rows, cols = linear_sum_assignment(cost)
    matches = []
    for i, j in zip(rows, cols):
        if cost[i, j] >= 1e8:
            continue
        row = sorted(v for v in cost[i] if v < 1e8)
        col = sorted(v for v in cost[:, j] if v < 1e8)
        if len(row) > 1 and row[1]-row[0] < .12:
            continue
        if len(col) > 1 and col[1]-col[0] < .12:
            continue
        matches.append((int(i), int(j)))
    return matches


def read_history():
    p = PUBLIC/'argo_history.json'
    if not p.exists():
        return {'tracks': {}, 'next_id': 1}
    try:
        h = json.loads(p.read_text(encoding='utf-8'))
        if h.get('source_id') != 'RAINVIEWER' or not isinstance(h.get('tracks'), dict):
            raise ValueError('Storico incompatibile')
        h['next_id'] = max(int(h.get('next_id', 1)), 1)
        return h
    except (OSError, ValueError, TypeError):
        return {'tracks': {}, 'next_id': 1}


def write(name, obj):
    p = PUBLIC/name
    tmp = p.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(p)


def main():
    host, older, newer, dt = frames()
    old_img, x0, y0 = mosaic(host, older)
    new_img, _, _ = mosaic(host, newer)
    old, cells = detect(old_img, x0, y0), detect(new_img, x0, y0)
    pairs = associate(old, cells, dt)
    history = read_history()
    tracks = history['tracks']
    next_id = history['next_id']
    # Migrazione: gli identificativi precedenti non vengono riutilizzati.
    for key in tracks:
        if key.startswith('ARGO-') and key[5:].isdigit():
            next_id = max(next_id, int(key[5:])+1)
    t0, t1 = stamp(older['time']), stamp(newer['time'])
    cutoff = newer['time'] - HISTORY_HOURS*3600
    tracks = {k: v for k, v in tracks.items()
              if v.get('observations') and epoch(v['observations'][-1]['timestamp']) >= cutoff}
    # Non creare una nuova identità se il workflow rielabora lo stesso frame.
    prior = [(key, tr['observations'][-1]) for key, tr in tracks.items()
             if tr['observations'][-1]['timestamp'] == t0]
    occupied = set()
    used_old = set()
    assigned = set()

    def allocate():
        nonlocal next_id
        while f'ARGO-{next_id:06d}' in tracks:
            next_id += 1
        key = f'ARGO-{next_id:06d}'
        next_id += 1
        return key

    for i, j in pairs:
        a, b = old[i], cells[j]
        choices = sorted((km(a, pos), key) for key, pos in prior if key not in occupied)
        # Richiede un candidato chiaramente migliore del secondo.
        key = None
        if choices and choices[0][0] <= RECONNECT_KM:
            if len(choices) == 1 or choices[1][0]-choices[0][0] >= 3:
                key = choices[0][1]
        if key is None:
            # Idempotenza: lo stesso frame può essere elaborato più volte.
            existing = sorted((km(b, tr['observations'][-1]), candidate)
                              for candidate, tr in tracks.items()
                              if tr['observations'][-1]['timestamp'] == t1
                              and candidate not in occupied)
            if existing and existing[0][0] < 1.0 and (len(existing) == 1 or existing[1][0]-existing[0][0] > 2):
                key = existing[0][1]
            else:
                key = allocate()
                tracks[key] = {'observations': [dict(lat=a['lat'], lon=a['lon'], timestamp=t0)]}
        occupied.add(key)
        used_old.add(i)
        assigned.add(j)
        obs = tracks[key]['observations']
        point = dict(lat=b['lat'], lon=b['lon'], timestamp=t1)
        if obs[-1]['timestamp'] == t0:
            obs.append(point)
        elif obs[-1]['timestamp'] != t1:
            raise RuntimeError('Traccia temporale incoerente')
        b['id'] = key
        b['track_id'] = key
        b['motion'] = {'speed_kmh': round(km(a, b)*60/dt, 2),
                       'direction_deg': round(bearing(a, b), 1),
                       'distance_km': round(km(a, b), 2),
                       'interval_min': dt, 'type': 'observed'}

    # Ogni centroide, anche senza vettore, riceve un ID persistibile.
    # Le tracce senza vettore non vengono collegate artificialmente.
    for j, cell in enumerate(cells):
        if j not in assigned:
            # Evita duplicati in caso di scansione già pubblicata.
            existing = [(km(cell, tr['observations'][-1]), key)
                        for key, tr in tracks.items()
                        if tr['observations'][-1]['timestamp'] == t1]
            existing.sort()
            if existing and existing[0][0] < 1.0 and (len(existing) == 1 or existing[1][0]-existing[0][0] > 2):
                key = existing[0][1]
            else:
                key = allocate()
                tracks[key] = {'observations': [dict(lat=cell['lat'], lon=cell['lon'], timestamp=t1)]}
            cell['id'] = key
            cell['track_id'] = key
            cell['motion'] = None
        cell['vectors'] = {}

    source = {'provider': 'RainViewer', 'attribution': 'Weather data by RainViewer',
              'url': 'https://www.rainviewer.com/', 'product': 'Radar composite PNG',
              'method': 'Segmentazione cromatica; nessun dBZ grezzo'}
    payload = {'schema_version': 4, 'timestamp': t1,
               'generated_at': datetime.now(timezone.utc).isoformat(),
               'source': source,
               'radar_tiles': {'host': host, 'path': newer['path'],
                               'zoom_max': 7, 'color': COLOR, 'options': OPTIONS},
               'cell_count': len(cells), 'observed_vectors': len(pairs), 'cells': cells}
    write('argo.json', payload)
    write('argo_history.json', {'schema_version': 4, 'source_id': 'RAINVIEWER',
                                'next_id': next_id, 'tracks': tracks})
    write('argo_quality.json', {'timestamp': t1, 'previous_timestamp': t0,
                                'interval_minutes': dt, 'previous_cells': len(old),
                                'current_cells': len(cells), 'matched_cells': len(pairs),
                                'quality_status': 'experimental'})
    print(f'ARGO V2.2: {len(cells)} centroidi, {len(pairs)} vettori, prossimo ID {next_id}')


if __name__ == '__main__':
    main()
    
