from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import logging

from hasse import (
    build_tree, 
    compute_affine_for_all_paths, 
    compute_affine_action,
    get_adjoint_weight,
    compute_gradation
)

log = logging.getLogger("uvicorn.error")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

frontend_dir = Path(__file__).resolve().parent.parent / "frontend"

if frontend_dir.exists() and frontend_dir.is_dir():
    app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")
    log.info(f"Serving frontend from: {frontend_dir}")
else:
    log.warning(f"Frontend directory not found at {frontend_dir}. Static files will not be served. "
                "Ensure frontend/ exists next to backend/.")

@app.get("/", response_class=HTMLResponse)
async def get_index():
    index_path = frontend_dir / "index.html"
    if index_path.exists():
        return index_path.read_text(encoding="utf-8")
    return HTMLResponse(
        "<h2>Lie Algebra Cohomology API</h2>"
        "<p>Backend is running! Put your frontend files in a folder called <code>frontend</code> next to <code>backend</code>.</p>"
        "<p>API endpoints:</p>"
        "<ul>"
        "<li>POST /compute/{group} - Compute Hasse tree</li>"
        "<li>POST /cohomology/{group} - Compute cohomology for all paths</li>"
        "</ul>",
        status_code=200,
    )


@app.post("/compute/{group}")
async def compute_group(group: str, request: Request):
    try:
        data = await request.json()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}")
    
    try:
        n = int(data.get("n", 4))
        l = int(data.get("l", 2))
        selected = data.get("selected", [])
        if not isinstance(selected, list):
            raise ValueError("selected must be a list of 1-based indices")
        selected = [int(x) for x in selected]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid input: {e}")

    try:
        tree = build_tree(group, n, selected, l)
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

    return {"tree": tree}


@app.post("/cohomology/{group}")
async def cohomology_all_paths(group: str, request: Request):
    try:
        data = await request.json()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}")
    
    try:
        n = int(data.get("n"))
        l = int(data.get("l", 1))
        selected = data.get("selected", [])
        if not isinstance(selected, list):
            raise ValueError("selected must be a list")
        selected = [int(x) for x in selected]
        
        use_adjoint = bool(data.get("use_adjoint", False))

        if use_adjoint:
            weight = get_adjoint_weight(group, n)
        else:
            weight = [1] * n
            
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid input: {e}")

    try:
        tree = build_tree(group, n, selected, l)
        coho = compute_affine_for_all_paths(group, n, selected, l, weight)

        gradations = compute_gradation(group, n, selected, l, weight)
        
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

    return {
        "tree": tree, 
        "cohomology_results": coho,
        "gradations": gradations,
        "weight_used": weight,
        "use_adjoint": use_adjoint

    }
