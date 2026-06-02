"""
Advanced basic utility tools — extended with modern helpers for everyday dev needs.
"""
from __future__ import annotations

import datetime as dt
import json
import math
import random
import uuid
from typing import Any

from ..registry import tool


# ── Time & Date ─────────────────────────────────────────────────────

@tool(
    name="current_time",
    description="Get the current date and time (local and UTC) with timezone info and Unix timestamp.",
    parameters_schema={
        "type": "object",
        "properties": {
            "format": {
                "type": "string",
                "enum": ["full", "unix", "iso", "date"],
                "default": "full",
                "description": "Output format: full (local+UTC), unix (epoch), iso (ISO-8601), date (YYYY-MM-DD only)",
            },
        },
        "required": [],
    },
    category="utility",
    cacheable=True,
    cache_ttl=10,
)
async def current_time(params: dict[str, Any]) -> str:
    fmt = params.get("format", "full")
    now = dt.datetime.now()
    utc = dt.datetime.utcnow()

    if fmt == "unix":
        return str(dt.datetime.timestamp(now))
    if fmt == "iso":
        return now.isoformat(timespec="seconds")
    if fmt == "date":
        return now.strftime("%Y-%m-%d")

    return (
        f"Local: {now.isoformat(timespec='seconds')}  ({now.strftime('%A, %B %d, %Y')})\n"
        f"UTC:   {utc.isoformat(timespec='seconds')}\n"
        f"Unix:  {dt.datetime.timestamp(now):.0f}"
    )


@tool(
    name="date_calc",
    description="Perform date arithmetic: add/subtract days, weeks, months, or calculate difference between two dates.",
    parameters_schema={
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "enum": ["add", "subtract", "diff", "weekday", "age"],
                "description": "Operation type",
            },
            "date": {
                "type": "string",
                "description": "Base date in YYYY-MM-DD format (defaults to today)",
            },
            "days": {"type": "integer", "description": "Days to add/subtract"},
            "weeks": {"type": "integer", "description": "Weeks to add/subtract"},
            "months": {"type": "integer", "description": "Months to add/subtract"},
            "other_date": {"type": "string", "description": "Second date for 'diff' operation (YYYY-MM-DD)"},
        },
        "required": ["operation"],
    },
    category="utility",
    cacheable=True,
    cache_ttl=5,
)
async def date_calc(params: dict[str, Any]) -> str:
    op = params["operation"]
    base_str = params.get("date", dt.date.today().isoformat())

    try:
        base = dt.date.fromisoformat(base_str)
    except ValueError:
        return f"Error: invalid date '{base_str}' — use YYYY-MM-DD"

    if op == "weekday":
        return f"{base_str} is a {base.strftime('%A')}"

    if op == "age":
        today = dt.date.today()
        if base > today:
            return f"Date {base_str} is in the future"
        years = today.year - base.year
        if (today.month, today.day) < (base.month, base.day):
            years -= 1
        return f"Age from {base_str} to today: {years} year(s)"

    if op == "diff":
        other_str = params.get("other_date", "")
        if not other_str:
            return "Error: other_date required for 'diff' operation"
        try:
            other = dt.date.fromisoformat(other_str)
        except ValueError:
            return f"Error: invalid other_date '{other_str}'"
        delta = abs((base - other).days)
        return f"Difference between {base_str} and {other_str}: {delta} day(s) ({delta / 365.25:.1f} year(s))"

    # add / subtract
    days = params.get("days", 0) or 0
    weeks = params.get("weeks", 0) or 0
    months = params.get("months", 0) or 0

    delta = dt.timedelta(days=days + weeks * 7)
    if months:
        month = base.month + (months if op == "add" else -months)
        year = base.year + (month - 1) // 12
        month = ((month - 1) % 12) + 1
        try:
            result = base.replace(year=year, month=month)
        except ValueError:
            # Handle month overflow (e.g., Jan 31 + 1 month = Feb 28)
            import calendar
            _, last_day = calendar.monthrange(year, month)
            result = base.replace(year=year, month=month, day=min(base.day, last_day))
    else:
        result = base + delta if op == "add" else base - delta

    return f"{op.capitalize()} {abs(days):d}d / {abs(weeks):d}w / {abs(months):d}m → {result.isoformat()} ({result.strftime('%A')})"


# ── Math & Calculator ─────────────────────────────────────────────

@tool(
    name="calculator",
    description="Evaluate a mathematical expression. Supports +, -, *, /, **, sqrt, sin, cos, tan, log, log2, log10, exp, pi, e, tau, ceil, floor, abs, min, max, round.",
    parameters_schema={
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "Math expression e.g. '2 * (3 + 4) ** 2'",
            },
            "precision": {
                "type": "integer",
                "default": 6,
                "description": "Number of decimal places to round to",
            },
        },
        "required": ["expression"],
    },
    category="utility",
    cacheable=True,
    cache_ttl=60,
)
async def calculator(params: dict[str, Any]) -> str:
    expr = params.get("expression", "")
    precision = int(params.get("precision", 6))

    allowed = {k: getattr(math, k) for k in (
        "sqrt", "sin", "cos", "tan", "log", "log2", "log10", "exp", "pow",
        "floor", "ceil", "pi", "e", "tau", "radians", "degrees", "asin", "acos", "atan",
    )}
    allowed["abs"] = abs
    allowed["min"] = min
    allowed["max"] = max
    allowed["round"] = round
    allowed["int"] = int
    allowed["float"] = float
    allowed["str"] = str

    try:
        result = eval(expr, {"__builtins__": {}}, allowed)  # noqa: S307
        if isinstance(result, float):
            result = round(result, precision)
        return str(result)
    except Exception as e:
        return f"Error: {e}"


@tool(
    name="unit_convert",
    description="Convert between units: length (m, km, mi, ft, in), mass (kg, lb, oz, g), temperature (C, F, K), volume (L, gal, cup, fl_oz), speed (kmh, mph, mps, kn), data (B, KB, MB, GB, TB).",
    parameters_schema={
        "type": "object",
        "properties": {
            "value": {"type": "number", "description": "Numeric value to convert"},
            "from_unit": {"type": "string", "description": "Source unit"},
            "to_unit": {"type": "string", "description": "Target unit"},
            "category": {
                "type": "string",
                "enum": ["length", "mass", "temperature", "volume", "speed", "data"],
                "description": "Measurement category (auto-detected if omitted)",
            },
        },
        "required": ["value", "from_unit", "to_unit"],
    },
    category="utility",
    cacheable=True,
    cache_ttl=120,
)
async def unit_convert(params: dict[str, Any]) -> str:
    val = float(params["value"])
    from_u = params["from_unit"].lower().strip()
    to_u = params["to_unit"].lower().strip()

    # Length conversions (to meters)
    length_to_m = {
        "m": 1.0, "meter": 1.0, "meters": 1.0,
        "km": 1000.0, "kilometer": 1000.0, "kilometers": 1000.0,
        "mi": 1609.344, "mile": 1609.344, "miles": 1609.344,
        "ft": 0.3048, "foot": 0.3048, "feet": 0.3048,
        "in": 0.0254, "inch": 0.0254, "inches": 0.0254,
        "cm": 0.01, "centimeter": 0.01, "centimeters": 0.01,
        "mm": 0.001, "millimeter": 0.001, "millimeters": 0.001,
        "yd": 0.9144, "yard": 0.9144, "yards": 0.9144,
    }

    # Mass conversions (to kg)
    mass_to_kg = {
        "kg": 1.0, "kilogram": 1.0, "kilograms": 1.0,
        "g": 0.001, "gram": 0.001, "grams": 0.001,
        "lb": 0.453592, "lbs": 0.453592, "pound": 0.453592, "pounds": 0.453592,
        "oz": 0.0283495, "ounce": 0.0283495, "ounces": 0.0283495,
        "mg": 0.000001, "milligram": 0.000001, "milligrams": 0.000001,
        "ton": 907.185, "tonne": 1000.0, "metric_ton": 1000.0,
    }

    # Volume conversions (to liters)
    vol_to_l = {
        "l": 1.0, "liter": 1.0, "liters": 1.0, "litre": 1.0, "litres": 1.0,
        "ml": 0.001, "milliliter": 0.001, "milliliters": 0.001,
        "gal": 3.78541, "gallon": 3.78541, "gallons": 3.78541,
        "cup": 0.236588, "cups": 0.236588,
        "fl_oz": 0.0295735, "fl oz": 0.0295735, "fluid_ounce": 0.0295735,
        "qt": 0.946353, "quart": 0.946353, "quarts": 0.946353,
        "pt": 0.473176, "pint": 0.473176, "pints": 0.473176,
    }

    # Speed conversions (to km/h)
    speed_to_kmh = {
        "kmh": 1.0, "km/h": 1.0, "kph": 1.0, "kilometer_per_hour": 1.0,
        "mph": 1.60934, "mi/h": 1.60934, "mile_per_hour": 1.60934,
        "mps": 3.6, "m/s": 3.6, "meter_per_second": 3.6,
        "kn": 1.852, "knot": 1.852, "knots": 1.852,
    }

    # Data conversions (to bytes)
    data_to_b = {
        "b": 1.0, "byte": 1.0, "bytes": 1.0,
        "kb": 1024.0, "kilobyte": 1024.0, "kilobytes": 1024.0,
        "mb": 1048576.0, "megabyte": 1048576.0, "megabytes": 1048576.0,
        "gb": 1073741824.0, "gigabyte": 1073741824.0, "gigabytes": 1073741824.0,
        "tb": 1099511627776.0, "terabyte": 1099511627776.0, "terabytes": 1099511627776.0,
        "kib": 1024.0, "kibibyte": 1024.0,
        "mib": 1048576.0, "mebibyte": 1048576.0,
        "gib": 1073741824.0, "gibibyte": 1073741824.0,
        "tib": 1099511627776.0, "tebibyte": 1099511627776.0,
    }

    # Auto-detect category if not specified
    cat = params.get("category", "")
    if not cat:
        if from_u in length_to_m and to_u in length_to_m:
            cat = "length"
        elif from_u in mass_to_kg and to_u in mass_to_kg:
            cat = "mass"
        elif from_u in vol_to_l and to_u in vol_to_l:
            cat = "volume"
        elif from_u in speed_to_kmh and to_u in speed_to_kmh:
            cat = "speed"
        elif from_u in data_to_b and to_u in data_to_b:
            cat = "data"
        elif from_u in ("c", "f", "k") and to_u in ("c", "f", "k"):
            cat = "temperature"
        else:
            return f"Error: could not auto-detect category for '{from_u}' → '{to_u}'"

    try:
        if cat == "length":
            meters = val * length_to_m[from_u]
            result = meters / length_to_m[to_u]
        elif cat == "mass":
            kg = val * mass_to_kg[from_u]
            result = kg / mass_to_kg[to_u]
        elif cat == "temperature":
            if from_u == "c":
                kelvin = val + 273.15
            elif from_u == "f":
                kelvin = (val - 32) * 5 / 9 + 273.15
            else:
                kelvin = val
            if to_u == "c":
                result = kelvin - 273.15
            elif to_u == "f":
                result = (kelvin - 273.15) * 9 / 5 + 32
            else:
                result = kelvin
        elif cat == "volume":
            liters = val * vol_to_l[from_u]
            result = liters / vol_to_l[to_u]
        elif cat == "speed":
            kmh = val * speed_to_kmh[from_u]
            result = kmh / speed_to_kmh[to_u]
        elif cat == "data":
            bytes_val = val * data_to_b[from_u]
            result = bytes_val / data_to_b[to_u]
        else:
            return f"Error: unknown category '{cat}'"

        return f"{val} {from_u} = {result:.6g} {to_u}"
    except KeyError as e:
        return f"Error: unknown unit '{e}' in category '{cat}'"


# ── Random & UUID ────────────────────────────────────────────────

@tool(
    name="random_choice",
    description="Pick one or more random items from a list, or generate a random number in a range.",
    parameters_schema={
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of items to choose from",
            },
            "count": {
                "type": "integer",
                "default": 1,
                "description": "Number of items to pick (without replacement)",
            },
            "min": {"type": "number", "description": "Min for random number (overrides items)"},
            "max": {"type": "number", "description": "Max for random number (overrides items)"},
            "integer": {"type": "boolean", "default": True, "description": "Integer or float random"},
        },
        "required": [],
    },
    category="utility",
)
async def random_choice(params: dict[str, Any]) -> str:
    items = params.get("items") or []
    count = int(params.get("count", 1))
    min_v = params.get("min")
    max_v = params.get("max")

    if min_v is not None and max_v is not None:
        min_v, max_v = float(min_v), float(max_v)
        is_int = params.get("integer", True)
        if is_int:
            return str(random.randint(int(min_v), int(max_v)))
        return str(round(random.uniform(min_v, max_v), 6))

    if not items:
        return "Error: provide items or min/max range"

    if count <= 1:
        return random.choice(items)
    if count > len(items):
        count = len(items)
    chosen = random.sample(items, count)
    if count == 1:
        return chosen[0]
    return "\n".join(f"{i+1}. {c}" for i, c in enumerate(chosen))


@tool(
    name="generate_uuid",
    description="Generate UUIDs: v4 (random), v7 (time-ordered), or v5 (namespace-based).",
    parameters_schema={
        "type": "object",
        "properties": {
            "version": {
                "type": "integer",
                "enum": [4, 7, 5],
                "default": 4,
                "description": "UUID version: 4 (random), 7 (time-sortable), 5 (SHA-1 with namespace)",
            },
            "namespace": {
                "type": "string",
                "description": "Namespace UUID for v5 (e.g., '6ba7b810-9dad-11d1-80b4-00c04fd430c8' for DNS)",
            },
            "name": {
                "type": "string",
                "description": "Name to hash with namespace for v5",
            },
            "count": {
                "type": "integer",
                "default": 1,
                "description": "Number of UUIDs to generate",
            },
        },
        "required": [],
    },
    category="utility",
)
async def generate_uuid(params: dict[str, Any]) -> str:
    ver = int(params.get("version", 4))
    count = min(int(params.get("count", 1)), 100)

    uuids = []
    if ver == 4:
        for _ in range(count):
            uuids.append(str(uuid.uuid4()))
    elif ver == 7:
        # UUID v7: time-ordered (RFC 9562)
        for _ in range(count):
            import time as _t
            timestamp_ms = int(_t.time() * 1000)
            rand_bytes = random.randbytes(10)
            uuids.append(f"{timestamp_ms:08x}-{rand_bytes[:2].hex()}-7{rand_bytes[2:3].hex()}-{rand_bytes[3:5].hex()}-{rand_bytes[5:].hex()}")
    elif ver == 5:
        ns_str = params.get("namespace", "")
        name_str = params.get("name", "")
        if not ns_str or not name_str:
            return "Error: namespace and name required for UUID v5"
        try:
            ns = uuid.UUID(ns_str)
        except ValueError:
            return f"Error: invalid namespace UUID '{ns_str}'"
        uuids.append(str(uuid.uuid5(ns, name_str)))
    else:
        return f"Error: unsupported UUID version {ver}"

    if count == 1:
        return uuids[0]
    return "\n".join(uuids)


# ── String / Encoding Utilities ─────────────────────────────────

@tool(
    name="text_utils",
    description="Text processing utilities: count words/characters, base64 encode/decode, url encode/decode, hash (md5/sha1/sha256), JSON format/validate, case conversion.",
    parameters_schema={
        "type": "object",
        "properties": {
            "text": {"type": "string", "description": "Input text to process"},
            "operation": {
                "type": "string",
                "enum": [
                    "count", "base64_encode", "base64_decode", "url_encode", "url_decode",
                    "md5", "sha1", "sha256", "json_format", "json_validate",
                    "upper", "lower", "title", "capitalize", "reverse",
                    "lines", "unique_lines",
                ],
                "description": "Operation to perform",
            },
        },
        "required": ["text", "operation"],
    },
    category="utility",
    cacheable=True,
    cache_ttl=120,
)
async def text_utils(params: dict[str, Any]) -> str:
    text = params["text"]
    op = params["operation"]

    if op == "count":
        words = len(text.split())
        chars = len(text)
        lines = text.count("\n") + 1
        chars_no_space = len(text.replace(" ", ""))
        return f"Words: {words}, Chars: {chars} ({chars_no_space} no-space), Lines: {lines}"

    if op == "base64_encode":
        import base64
        return base64.b64encode(text.encode()).decode()

    if op == "base64_decode":
        import base64
        try:
            return base64.b64decode(text).decode(errors="replace")
        except Exception as e:
            return f"Error: {e}"

    if op == "url_encode":
        import urllib.parse
        return urllib.parse.quote(text)

    if op == "url_decode":
        import urllib.parse
        return urllib.parse.unquote(text)

    if op in ("md5", "sha1", "sha256"):
        import hashlib
        h = hashlib.new(op, text.encode())
        return h.hexdigest()

    if op == "json_format":
        try:
            parsed = json.loads(text)
            return json.dumps(parsed, indent=2, ensure_ascii=False)
        except json.JSONDecodeError as e:
            return f"Invalid JSON: {e}"

    if op == "json_validate":
        try:
            json.loads(text)
            return "Valid JSON"
        except json.JSONDecodeError as e:
            return f"Invalid JSON: {e}"

    if op == "reverse":
        return text[::-1]

    if op == "upper":
        return text.upper()
    if op == "lower":
        return text.lower()
    if op == "title":
        return text.title()
    if op == "capitalize":
        return text.capitalize()

    if op == "lines":
        lines = text.splitlines()
        return f"{len(lines)} lines:\n" + "\n".join(f"{i+1:>4}: {l}" for i, l in enumerate(lines))

    if op == "unique_lines":
        import collections
        lines = text.splitlines()
        seen = collections.OrderedDict.fromkeys(lines)
        return "\n".join(seen.keys())

    return f"Error: unknown operation '{op}'"


# ── IP / Network Info ────────────────────────────────────────────

@tool(
    name="ip_info",
    description="Get your public IP address and basic network information via ip-api.com (no API key needed).",
    parameters_schema={
        "type": "object",
        "properties": {},
        "required": [],
    },
    category="utility",
    cacheable=True,
    cache_ttl=300,
)
async def ip_info(_params: dict[str, Any]) -> str:
    import asyncio
    try:
        # Use asyncio subprocess to fetch via curl to avoid blocking the event loop
        proc = await asyncio.create_subprocess_exec(
            "curl", "-s", "--max-time", "10", "http://ip-api.com/json/",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=12)
        if proc.returncode != 0:
            # Fallback: use asyncio.open_connection for pure async HTTP
            import urllib.parse
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection("ip-api.com", 80), timeout=10
            )
            request = (
                "GET /json/ HTTP/1.1\r\n"
                "Host: ip-api.com\r\n"
                "Connection: close\r\n"
                "\r\n"
            )
            writer.write(request.encode())
            await writer.drain()
            response = b""
            while True:
                chunk = await asyncio.wait_for(reader.read(4096), timeout=5)
                if not chunk:
                    break
                response += chunk
            writer.close()
            await writer.wait_closed()
            # Parse HTTP response body
            body = response.split(b"\r\n\r\n", 1)[-1] if b"\r\n\r\n" in response else response
            data = json.loads(body.decode("utf-8", errors="replace"))
        else:
            data = json.loads(stdout.decode("utf-8", errors="replace"))

        if data.get("status") == "success":
            return (
                f"IP: {data.get('query', 'N/A')}\n"
                f"Location: {data.get('city', 'N/A')}, {data.get('regionName', 'N/A')}, {data.get('country', 'N/A')}\n"
                f"ISP: {data.get('isp', 'N/A')}\n"
                f"Org: {data.get('org', 'N/A')}\n"
                f"AS: {data.get('as', 'N/A')}\n"
                f"Coordinates: {data.get('lat', 'N/A')}, {data.get('lon', 'N/A')}\n"
                f"Timezone: {data.get('timezone', 'N/A')}"
            )
        return f"Error: {data.get('message', 'Unknown error')}"
    except Exception as e:
        return f"Error fetching IP info: {e}"
