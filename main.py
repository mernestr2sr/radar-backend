from fastapi import FastAPI, Response, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import boto3
from botocore import UNSIGNED
from botocore.config import Config
import pyart
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO
import tempfile
import os
from datetime import datetime

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

s3 = boto3.client('s3', region_name='us-east-1', config=Config(signature_version=UNSIGNED))
BUCKET = 'noaa-nexrad-level2'


@app.get("/")
def root():
    return {"status": "online", "message": "Radar backend v2 — with velocity"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/scans/{site}/{date}")
def list_scans(site: str, date: str):
    try:
        y, m, d = date.split('-')
        prefix = f"{y}/{m}/{d}/{site.upper()}/"
        response = s3.list_objects_v2(Bucket=BUCKET, Prefix=prefix, MaxKeys=300)
        if 'Contents' not in response:
            return {"site": site, "date": date, "scans": []}
        scans = [
            obj['Key'].split('/')[-1]
            for obj in response['Contents']
            if not obj['Key'].endswith('_MDM')
        ]
        return {"site": site.upper(), "date": date, "scans": scans, "count": len(scans)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/render/{site}/{filename}")
def render_velocity(site: str, filename: str, product: str = "velocity"):
    try:
        date_part = filename[4:12]
        y = date_part[0:4]
        m = date_part[4:6]
        d = date_part[6:8]
        s3_key = f"{y}/{m}/{d}/{site.upper()}/{filename}"

        with tempfile.NamedTemporaryFile(delete=False, suffix='.ar2v') as tmp:
            s3.download_fileobj(BUCKET, s3_key, tmp)
            tmp_path = tmp.name

        radar = pyart.io.read_nexrad_archive(tmp_path)

        fig = plt.figure(figsize=(8, 8), facecolor='black')
        ax = fig.add_subplot(111)

        display = pyart.graph.RadarDisplay(radar)

        if product == "velocity":
            display.plot('velocity', sweep=1, ax=ax,
                         vmin=-40, vmax=40, cmap='pyart_NWSVel',
                         colorbar_flag=False, title_flag=False, axislabels_flag=False)
        else:
            display.plot('reflectivity', sweep=0, ax=ax,
                         vmin=-20, vmax=75, cmap='pyart_NWSRef',
                         colorbar_flag=False, title_flag=False, axislabels_flag=False)

        ax.set_facecolor('black')
        ax.set_xticks([])
        ax.set_yticks([])

        buf = BytesIO()
        plt.savefig(buf, format='png', facecolor='black', bbox_inches='tight', pad_inches=0, dpi=80)
        plt.close(fig)
        buf.seek(0)

        os.unlink(tmp_path)

        return Response(content=buf.getvalue(), media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Render error: {str(e)}")
