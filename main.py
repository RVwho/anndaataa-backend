import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel

load_dotenv()

class CropAnalysisResult(BaseModel):
    disease: str
    confidence: float
    primary_chemical: str
    application_steps: str

class MandiPrice(BaseModel):
    crop: str
    price_per_quintal: int
    trend: str

class KVKLocation(BaseModel):
    name: str
    distance_km: float
    inventory_match: bool

class TimelineStep(BaseModel):
    phase: str
    action: str

class CropRecommendation(BaseModel):
    crop_name: str
    brief_reason: str
    timeline: list[TimelineStep]

class FarmPlannerResponse(BaseModel):
    soil_type: str
    recommended_crops: list[CropRecommendation]

app = FastAPI()

@app.head("/")
@app.get("/")
def health_check():
    return {"status": "AnnDaataa API is awake and running!"}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

INDIAN_LOCATIONS = {
    "Andhra Pradesh": ["Guntur", "Anantapur", "West Godavari", "Kurnool", "Krishna"],
    "Arunachal Pradesh": ["Papum Pare", "Changlang", "Lohit", "West Kameng", "Namsai"],
    "Assam": ["Nagaon", "Sonitpur", "Barpeta", "Jorhat", "Cachar"],
    "Bihar": ["Champaran", "Rohtas", "Nalanda", "Begusarai", "Muzaffarpur"],
    "Chhattisgarh": ["Raipur", "Durg", "Bastar", "Bilaspur", "Dhamtari"],
    "Goa": ["North Goa", "South Goa", "Ponda", "Mapusa", "Margao"],
    "Gujarat": ["Anand", "Rajkot", "Mehsana", "Junagadh", "Banaskantha"],
    "Haryana": ["Karnal", "Kurukshetra", "Hisar", "Sirsa", "Sonipat"],
    "Himachal Pradesh": ["Kangra", "Mandi", "Shimla", "Kullu", "Solan"],
    "Jharkhand": ["Ranchi", "Hazaribagh", "Dumka", "Jamshedpur", "Palamu"],
    "Karnataka": ["Mandya", "Dharwad", "Belagavi", "Shimoga", "Tumakuru"],
    "Kerala": ["Wayanad", "Palakkad", "Idukki", "Alappuzha", "Kottayam"],
    "Madhya Pradesh": ["Hoshangabad", "Ujjain", "Dewas", "Indore", "Vidisha"],
    "Maharashtra": ["Pune", "Nashik", "Nagpur", "Satara", "Ahmednagar"],
    "Manipur": ["Imphal West", "Thoubal", "Bishnupur", "Kakching", "Ukhrul"],
    "Meghalaya": ["East Khasi Hills", "West Jaintia Hills", "Ri-Bhoi", "West Garo Hills"],
    "Mizoram": ["Aizawl", "Lunglei", "Champhai", "Kolasib", "Serchhip"],
    "Nagaland": ["Dimapur", "Kohima", "Mokokchung", "Wokha", "Phek"],
    "Odisha": ["Bargarh", "Ganjam", "Cuttack", "Balasore", "Sambalpur"],
    "Punjab": ["Ludhiana", "Amritsar", "Bathinda", "Patiala", "Jalandhar"],
    "Rajasthan": ["Jhunjhunu", "Sri Ganganagar", "Kota", "Alwar", "Jaipur"],
    "Sikkim": ["Gangtok", "Geyzing", "Namchi", "Mangan", "Pakyong"],
    "Tamil Nadu": ["Thanjavur", "Coimbatore", "Madurai", "Salem", "Erode"],
    "Telangana": ["Nizamabad", "Karimnagar", "Warangal", "Medak", "Khammam"],
    "Tripura": ["West Tripura", "South Tripura", "Dhalai", "Unakoti", "Khowai"],
    "Uttar Pradesh": ["Hapur", "Meerut", "Bareilly", "Varanasi", "Gorakhpur"],
    "Uttarakhand": ["Udham Singh Nagar", "Haridwar", "Dehradun", "Nainital", "Almora"],
    "West Bengal": ["Burdwan", "Hooghly", "Murshidabad", "Nadia", "Midnapore"],
    "Andaman and Nicobar Islands": ["Port Blair", "Car Nicobar", "Havelock", "Mayabunder"],
    "Chandigarh": ["Chandigarh", "Sector 17", "Manimajra", "Hallomajra"],
    "Dadra and Nagar Haveli and Daman and Diu": ["Silvassa", "Daman", "Diu", "Khanvel"],
    "Lakshadweep": ["Kavaratti", "Minicoy", "Agatti", "Amini"],
    "Delhi": ["New Delhi", "Najafgarh", "Narela", "Mehrauli", "Shahdara"],
    "Puducherry": ["Puducherry", "Karaikal", "Mahe", "Yanam"],
    "Jammu and Kashmir": ["Srinagar", "Jammu", "Anantnag", "Baramulla", "Kathua"],
    "Ladakh": ["Leh", "Kargil", "Nubra", "Zanskar"]
}

FLATTENED_LOCATIONS = [
    f"{district}, {state}"
    for state, districts in INDIAN_LOCATIONS.items()
    for district in districts
]

FARM_PLANNER_CACHE = {}

@app.post("/api/analyze-crop", response_model=CropAnalysisResult)
async def analyze_crop(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Only images are allowed.")
    try:
        image_bytes = await file.read()
        image_part = types.Part.from_bytes(data=image_bytes, mime_type=file.content_type)
        prompt_text = "You are an agricultural expert. Analyze the image and return ONLY a valid JSON object with these exact keys: disease (string), confidence (float), primary_chemical (string, strictly just the name of the most effective pesticide/fungicide/treatment), and application_steps (string, the detailed instructions)."
        
        try:
            response = client.models.generate_content(
                model="gemini-3.5-flash",
                contents=[prompt_text, image_part],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=CropAnalysisResult,
                ),
            )
        except Exception as ai_err:
            print(f"AI Service Exception: {ai_err}")
            raise HTTPException(status_code=503, detail="AI Service Unavailable. Please check your internet connection.")

        if response.parsed:
            return response.parsed
        return CropAnalysisResult.model_validate_json(response.text)
    except HTTPException as http_err:
        raise http_err
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/mandi-prices", response_model=list[MandiPrice])
async def get_mandi_prices():
    return [
        MandiPrice(crop="Wheat (Kanak)", price_per_quintal=2420, trend="up"),
        MandiPrice(crop="Paddy (Basmati)", price_per_quintal=4150, trend="up"),
        MandiPrice(crop="Mustard Seeds", price_per_quintal=5680, trend="down"),
        MandiPrice(crop="Tomato (Tamatar)", price_per_quintal=1850, trend="up")
    ]

@app.get("/api/kvk-locations", response_model=list[KVKLocation])
async def get_kvk_locations(treatment: str = None):
    has_treatment = treatment is not None and treatment.strip() != ""
    treatment_lower = treatment.lower() if has_treatment else ""
    match_keywords = ["fungicide", "biological", "pesticide", "neem", "copper", "chemical", "carbendazim"]
    inv_match_1 = False
    inv_match_2 = False
    inv_match_3 = False
    if has_treatment:
        inv_match_1 = any(kw in treatment_lower for kw in match_keywords)
        inv_match_2 = "bio" in treatment_lower or "neem" in treatment_lower or "organic" in treatment_lower
        inv_match_3 = "chemical" in treatment_lower or "copper" in treatment_lower or "carbendazim" in treatment_lower
    else:
        inv_match_1 = True
        inv_match_2 = False
        inv_match_3 = True
    return [
        KVKLocation(name="KVK Central Research Station", distance_km=1.8, inventory_match=inv_match_1),
        KVKLocation(name="Govt Agricultural Supply Depot", distance_km=3.5, inventory_match=inv_match_2),
        KVKLocation(name="Krishi Vigyan Kendra Extension", distance_km=6.2, inventory_match=inv_match_3)
    ]

@app.get("/api/locations", response_model=list[str])
async def get_locations(q: str = ""):
    q = q.strip().lower()
    if not q:
        return []
    matches = []
    for loc in FLATTENED_LOCATIONS:
        if q in loc.lower():
            matches.append(loc)
    def get_sort_key(loc_str: str) -> int:
        parts = loc_str.lower().split(", ")
        district = parts[0]
        state = parts[1] if len(parts) > 1 else ""
        if district.startswith(q):
            return 0
        elif state.startswith(q):
            return 1
        return 2
    matches.sort(key=get_sort_key)
    return matches[:10]

@app.get("/api/farm-planner", response_model=FarmPlannerResponse)
async def get_farm_planner(location: str = "New Delhi"):
    normalized_location = location.lower().strip()
    if not normalized_location:
        normalized_location = "new delhi"
    if normalized_location in FARM_PLANNER_CACHE:
        return FARM_PLANNER_CACHE[normalized_location]
    prompt_text = (
        f"You are an expert agricultural scientist and plant pathologist. "
        f"Analyze the soil composition and agricultural environment of location: '{location}'. "
        f"Provide the typical soil type of this region, and recommend exactly 3 highly suitable crops for rotating or growing here. "
        f"For each recommended crop, provide a brief reason and a realistic growing timeline consisting of 4-5 sequential phases. "
        f"For the crop_name field, you MUST strictly output the common English name followed by the common Hindi name in Devanagari script in parentheses. "
        f"For example: 'Wheat (गेहूं)' or 'Mustard (सरसों)'. Absolutely DO NOT output botanical or scientific names (like Triticum aestivum)."
    )
    try:
        response = client.models.generate_content(
            model="gemini-3.5-flash",
            contents=[prompt_text],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=FarmPlannerResponse,
            ),
        )
        result = None
        if response.parsed:
            result = response.parsed
        else:
            result = FarmPlannerResponse.model_validate_json(response.text)
        FARM_PLANNER_CACHE[normalized_location] = result
        return result
    except Exception as e:
        print(f"Farm Planner AI Service Exception: {e}")
        raise HTTPException(status_code=503, detail="Agri-planner service temporarily offline. Please verify connectivity.")
