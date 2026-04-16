from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def root():
    return {"status": "online", "message": "Radar backend is running"}

@app.get("/health")
def health():
    return {"status": "ok"}
