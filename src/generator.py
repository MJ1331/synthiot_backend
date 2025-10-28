import math
import random
from datetime import datetime, timedelta
from typing import Dict, Any, List, Iterator
from dateutil import parser as dateparser
import numpy as np
import io, csv

def build_timestamps(params: Dict[str, Any], max_points: int) -> List[datetime]:
    """
    Build timestamp list from params: expects start_iso, end_iso and freq_seconds.
    Caps at max_points to avoid runaway sizes.
    """
    start = dateparser.parse(params['start_iso'])
    end = dateparser.parse(params['end_iso'])
    freq = timedelta(seconds=int(params.get('freq_seconds', 3600)))
    timestamps = []
    t = start
    while t <= end and len(timestamps) < max_points:
        timestamps.append(t)
        t += freq
    return timestamps

def diurnal_baseline(ts: datetime, params: Dict[str, Any]) -> float:
    if 'temp_range' in params:
        tmin, tmax = params['temp_range']
    else:
        w = params.get('weather', 'normal')
        if w == 'clear':
            tmin, tmax = 18.0, 32.0
        elif w == 'rain':
            tmin, tmax = 12.0, 24.0
        else:
            tmin, tmax = 15.0, 28.0
    mean = (tmin + tmax) / 2.0
    amplitude = (tmax - tmin) / 2.0
    hour = ts.hour + ts.minute / 60.0
    phase = 15.0
    return mean + amplitude * math.sin(2 * math.pi * (hour - phase) / 24.0)

def generate_temperature_rows_iter(params: Dict[str, Any], max_points: int) -> Iterator[List[Any]]:
    timestamps = build_timestamps(params, max_points)
    if not timestamps:
        return
    if 'temp_range' in params and isinstance(params['temp_range'], (list, tuple)) and len(params['temp_range']) >= 2:
        tmin, tmax = float(params['temp_range'][0]), float(params['temp_range'][1])
    else:
        tmin, tmax = 15.0, 28.0
    seed = int(params.get('seed', 0))
    rnd = np.random.default_rng(seed)
    phi = 0.88
    sigma_proc = 0.25
    sigma_sensor = 0.45
    prev_noise = rnd.normal(0, sigma_proc)
    p_spike_start = 0.003
    p_spike_end = 0.2
    in_spike = False
    spike_effect = 0.0
    for t in timestamps:
        baseline = diurnal_baseline(t, params)
        if in_spike:
            if rnd.random() < p_spike_end:
                in_spike = False
                spike_effect = 0.0
        else:
            if rnd.random() < p_spike_start:
                in_spike = True
                spike_effect = rnd.choice([rnd.uniform(2.0, 6.0), -rnd.uniform(2.0, 6.0)])
        proc_noise = phi * prev_noise + rnd.normal(0, sigma_proc)
        prev_noise = proc_noise
        sensor_noise = rnd.normal(0, sigma_sensor)
        temp = baseline + proc_noise + sensor_noise
        if in_spike:
            temp += spike_effect
        temp = round(float(temp), 1)
        yield [t.isoformat() + "Z", temp]

def csv_stream_generator_from_iterator(header: List[str], row_iterator: Iterator[List[Any]]):
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(header)
    yield buf.getvalue()
    buf.seek(0); buf.truncate(0)
    for row in row_iterator:
        writer.writerow(row)
        yield buf.getvalue()
        buf.seek(0); buf.truncate(0)
