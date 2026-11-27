"""Local digital-twin demonstration using the Sentinela API."""
import csv
import os
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

from fastapi import HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.routing import Mount

from app import Measurements, app
from preprocessing import CSV_FIELDS

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "demo_gemeo"
FIRST_UDI = 4127
LAST_UDI = 4206

app.mount("/demo-assets", StaticFiles(directory=ASSETS), name="demo-assets")


@app.get("/demo", include_in_schema=False)
def dashboard():
    return FileResponse(ASSETS / "index.html")


@app.get("/demo/leituras")
def replay():
    dataset = Path(os.getenv("SENTINELA_DATASET", str(ROOT.parent / "ai4i2020.csv")))
    if not dataset.is_file():
        raise HTTPException(503, "Coloque ai4i2020.csv ao lado de backend/ ou configure SENTINELA_DATASET.")

    start = datetime(2020, 1, 1, 12, tzinfo=timezone.utc)
    try:
        with dataset.open(encoding="utf-8-sig", newline="") as stream:
            rows = [row for row in csv.DictReader(stream) if FIRST_UDI <= int(row["UDI"]) <= LAST_UDI]
        rows.sort(key=lambda row: int(row["UDI"]))
        if [int(row["UDI"]) for row in rows] != list(range(FIRST_UDI, LAST_UDI + 1)):
            raise ValueError("Sequência incompleta")
        readings = []
        for index, row in enumerate(rows):
            values = {
                name: row[column] if name == "type" else float(Decimal(row[column]))
                for column, name in CSV_FIELDS.items()
            }
            readings.append({
                "udi": int(row["UDI"]),
                "timestamp": (start + timedelta(seconds=30 * index)).isoformat(),
                "measurements": Measurements(**values).model_dump(),
            })
    except (KeyError, ValueError, ArithmeticError, OSError):
        raise HTTPException(503, "O CSV não contém a sequência ou as colunas esperadas do AI4I.") from None
    return {"machine_id": "MAQ-01", "source": "AI4I 2020 · replay simulado", "readings": readings}


# The repository serves its frontend through a catch-all mount at "/".
# Keep that mount last so the demo routes and assets take precedence.
routes = app.router.routes
for route in routes:
    if isinstance(route, Mount) and route.path == "" and route.name == "front":
        routes.remove(route)
        routes.append(route)
        break
