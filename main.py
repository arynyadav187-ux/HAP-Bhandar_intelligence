import os
import json
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import fitz  # PyMuPDF
from pydantic import BaseModel
from google import genai
from google.genai import types

app = FastAPI(title="HAP Bhandar API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_PATH = os.path.join("data", "plans.json")

# Schema matching your exact data/plans.json structure
class Evidence(BaseModel):
    page: int
    quote: str

class AlertThresholds(BaseModel):
    yellow: float | None
    orange: float | None
    red: float | None
    evidence: Evidence

class LegalBacking(BaseModel):
    act_cited: str | None
    section_cited: str | None
    enforceable_penalties: bool
    evidence: Evidence

class Funding(BaseModel):
    has_dedicated_budget: bool
    source: str
    evidence: Evidence

class WHOElements(BaseModel):
    lead_body: bool
    early_warning: bool
    public_comms: bool
    indoor_heat_reduction: bool
    vulnerable_groups: bool
    health_preparedness: bool
    urban_planning: bool
    surveillance_me: bool

class PlanExtraction(BaseModel):
    id: str
    name: str
    state: str
    tier: str
    triggers: AlertThresholds
    legal: LegalBacking
    funding: Funding
    who_elements: WHOElements

def load_plans():
    if not os.path.exists(DATA_PATH):
        return []
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.get("/api/plans")
async def get_plans():
    plans = load_plans()
    enriched = []
    for p in plans:
        who = p.get("who_elements", {})
        present_count = sum(1 for v in who.values() if v is True)
        who_score = round((present_count / 8) * 100, 1)

        flags = []
        if not p.get("funding", {}).get("has_dedicated_budget", False):
            flags.append("UNFUNDED MANDATE")
        if not p.get("legal", {}).get("enforceable_penalties", False):
            flags.append("MISSING STATUTORY TEETH")
        if not who.get("indoor_heat_reduction", False) and not who.get("urban_planning", False):
            flags.append("NO STRUCTURAL COOLING")

        p["who_score"] = who_score
        p["red_flags"] = flags
        enriched.append(p)
    return JSONResponse(content=enriched)

@app.post("/api/extract")
async def extract_live_pdf(
    file: UploadFile = File(...),
    name: str = Form("Uploaded City HAP"),
    state: str = Form("India"),
    tier: str = Form("City")
):
    pdf_bytes = await file.read()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    full_text = ""
    for i, page in enumerate(doc):
        full_text += f"\n--- PAGE {i + 1} ---\n" + page.get_text("text")

    api_key = os.environ.get("AQ.Ab8RN6K8s95EtP1d1zhx8EdrKZiIl9ff7E8xyiuS-dpjQIbWYw")
    if not api_key:
        return JSONResponse(status_code=400, content={"error": "GEMINI_API_KEY environment variable not set."})

    client = genai.Client(api_key=api_key)
    prompt = f"""
    Extract the Heat Action Plan details for {name} ({state}, Tier: {tier}).
    Every claim MUST include the exact page number and verbatim quote from the text.
    If statutory penalties or dedicated budgets are not explicitly written, set booleans to false.
    """

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[full_text[:150000], prompt],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=PlanExtraction,
            temperature=0.1,
        ),
    )

    new_plan = json.loads(response.text)
    new_plan["id"] = f"IN-LIVE-{len(load_plans()) + 1}"
    new_plan["name"] = name
    new_plan["state"] = state
    new_plan["tier"] = tier

    plans = load_plans()
    plans.insert(0, new_plan)
    with open(DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(plans, f, indent=2)

    return JSONResponse(content=new_plan)