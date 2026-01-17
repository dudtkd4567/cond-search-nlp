# server.py
# -*- coding: utf-8 -*-
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from cond_search import CondSearchEngine

app = FastAPI(title="CondSearch NLU Server")

ENGINE = None

class SearchReq(BaseModel):
    query: str = Field(..., min_length=1)
    topk: int = Field(200, ge=1, le=2000)
    per_clause_topk: int = Field(15, ge=1, le=200)

@app.on_event("startup")
def startup():
    global ENGINE
    base_dir = os.environ.get("COND_BASE_DIR", r"C:\삼성증권\POPHTSN\data\finddata")
    model = os.environ.get("COND_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    eng = CondSearchEngine(base_dir=base_dir, model_name=model)
    eng.load()                 # init 병목은 여기서 1회만
    ENGINE = eng

@app.get("/health")
def health():
    if ENGINE is None:
        raise HTTPException(status_code=503, detail="not ready")
    return {"ok": True}

@app.post("/search")
def search(req: SearchReq):
    if ENGINE is None:
        raise HTTPException(status_code=503, detail="Engine not ready")
    return ENGINE.search(req.query, topk=req.topk, per_clause_topk=req.per_clause_topk)
