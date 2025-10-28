import re
from datetime import datetime, timedelta
import random

def parse_prompt_to_params(prompt: str, rows_hint: int = None, freq_hint: int = None):
    p = (prompt or "").lower()
    params = {}
    m = re.search(r'(\d+)\s*(rows|samples|points)', prompt, re.I)
    if m:
        params['rows'] = int(m.group(1))
    elif rows_hint:
        params['rows'] = int(rows_hint)
    mdate = re.search(r'(\d{4}-\d{2}-\d{2})', prompt)
    if mdate:
        day = mdate.group(1)
        params['start_iso'] = f"{day}T00:00:00Z"
        params['end_iso'] = f"{day}T23:59:59Z"
        params.setdefault('rows', 24)
        params.setdefault('freq_seconds', 3600)
    if 'hour' in p or 'hourly' in p:
        params.setdefault('freq_seconds', 3600)
    mmin = re.search(r'every\s+(\d+)\s*(minute|minutes|min)', p)
    if mmin:
        params['freq_seconds'] = int(mmin.group(1)) * 60
    if freq_hint:
        params.setdefault('freq_seconds', int(freq_hint))
    mloc = re.search(r'(in|for)\s+([A-Za-z][A-Za-z ,.]+)', prompt, re.I)
    if mloc:
        location = mloc.group(2).strip().split(' for ')[0].split(' in ')[0].strip()
        params['location'] = location
    if 'rain' in p or 'storm' in p or 'snow' in p:
        params['weather'] = 'rain'
    elif 'clear' in p or 'sunny' in p:
        params['weather'] = 'clear'
    else:
        params['weather'] = 'normal'
    mrange = re.search(r'(-?\d+(?:\.\d+)?)\s*(?:to|-)\\s*(-?\d+(?:\.\d+)?)', prompt, re.I)
    # fallback regex simpler
    if not mrange:
        mrange = re.search(r'(-?\d+(?:\.\d+)?)\s*(?:to|-)\\s*(-?\d+(?:\.\d+)?)', prompt, re.I)
    if mrange:
        try:
            tmin = float(mrange.group(1)); tmax = float(mrange.group(2))
            params['temp_range'] = [min(tmin, tmax), max(tmin, tmax)]
        except Exception:
            pass
    params.setdefault('rows', params.get('rows', 24))
    params.setdefault('freq_seconds', params.get('freq_seconds', 3600))
    try:
        total_seconds = (int(params['rows']) - 1) * int(params['freq_seconds'])
        start = datetime.utcnow() - timedelta(seconds=total_seconds)
        params.setdefault('start_iso', start.isoformat() + "Z")
    except Exception:
        params.setdefault('start_iso', (datetime.utcnow() - timedelta(hours=params['rows'] * (params['freq_seconds'] / 3600))).isoformat() + "Z")
    params.setdefault('end_iso', datetime.utcnow().isoformat() + "Z")
    params.setdefault('seed', random.randint(0, 2**31 - 1))
    return params
