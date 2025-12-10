import os
import numpy as np
from typing import List, Dict
import cohere
import pdfplumber
from io import BytesIO
from sklearn.preprocessing import normalize
from fastapi import FastAPI, UploadFile, File, HTTPException, Body
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

COHERE_API_KEY = os.getenv("COHERE_API_KEY", "")
co = cohere.ClientV2(api_key=COHERE_API_KEY)

def _embed_texts(texts: List[str]) -> List[List[float]]:
    resp = co.embed(texts=texts, model="embed-multilingual-v3.0", input_type="search_document")
    return resp.embeddings.float

def cosine_sim(a: List[float], b: List[float]) -> float:
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def match_skills(resume_skills: List[str], jd_skills: List[str]) -> Dict[str, float]:
    if not resume_skills or not jd_skills:
        return {s: 0.0 for s in jd_skills}
    texts = resume_skills + jd_skills
    embs = _embed_texts(texts)
    n = len(resume_skills)
    resume_embs = embs[:n]
    jd_embs = embs[n:]
    scores = {}
    for i, jd_emb in enumerate(jd_embs):
        best = 0.0
        for r_emb in resume_embs:
            s = cosine_sim(jd_emb, r_emb)
            if s > best:
                best = s
        scores[jd_skills[i]] = best
    return scores

def compute_overall_score(skill_scores: Dict[str, float], experience_years: float, jd_seniority: str) -> float:
    if skill_scores:
        avg_skill = sum(skill_scores.values()) / len(skill_scores)
    else:
        avg_skill = 0.0
    seniority_map = {"junior": 1.0, "mid": 3.0, "senior": 6.0, "lead": 8.0}
    expected = seniority_map.get(jd_seniority.lower() if jd_seniority else "", 3.0)
    exp_score = max(0.0, min(1.0, experience_years / expected)) if experience_years else 0.0
    overall = 0.6 * avg_skill + 0.4 * exp_score
    return round(overall * 100, 2)

# -----------------------------
# FastAPI app & models
# -----------------------------

app = FastAPI()

class JDRequest(BaseModel):
    job_description: str

class MatchRequest(BaseModel):
    job_description: str
    candidate_profile: str = None  # Optional if resume_file is provided

class OfferEmailRequest(BaseModel):
    candidate_name: str
    job_title: str
    salary: str
    joining_date: str
    hr_name: str

class RejectionEmailRequest(BaseModel):
    candidate_name: str
    job_title: str
    hr_name: str
    reason: str = "We regret to inform you that we will not be moving forward with your application at this time."

class ResumeParseRequest(BaseModel):
    resume_text: str = None  # Optional for file upload

class ParsedResume(BaseModel):
    name: str
    email: str = ""
    phone: str = ""
    experience_years: int = 0
    skills: List[str] = []
    education: str = ""
    certifications: List[str] = []
    summary: str  # Formatted text for matching

class BulkMatchRequest(BaseModel):
    job_description: str
    candidates: List[str]  # each candidate profile as string

# -----------------------------
# Endpoints
# -----------------------------

@app.get("/")
def root():
    return {"message": "HR Agent API running successfully!"}

# Parse resume
@app.post("/parse-resume", response_model=ParsedResume)
async def parse_resume(resume_text: str = Body(None), resume_file: UploadFile = File(None)):
    if resume_file:
        # Read file content
        content = await resume_file.read()
        filename = resume_file.filename.lower()
        if filename.endswith('.pdf'):
            # Extract text from PDF
            with pdfplumber.open(BytesIO(content)) as pdf:
                resume_text = ''
                for page in pdf.pages:
                    resume_text += page.extract_text() or ''
        else:
            # Assume text file
            try:
                resume_text = content.decode('utf-8')
            except UnicodeDecodeError:
                resume_text = content.decode('latin-1')  # Fallback encoding
    elif not resume_text:
        raise HTTPException(status_code=400, detail="Either resume_text or resume_file must be provided")
    
    prompt = f"""
    Parse this resume text and extract the following information in a clean, HR-friendly JSON format:

    Required fields:
    - name: Full name of the candidate
    - email: Email address (if available, otherwise empty string)
    - phone: Phone number (if available, otherwise empty string)
    - experience_years: Total years of professional experience (integer)
    - skills: List of technical and soft skills mentioned
    - education: Highest education qualification (string)
    - certifications: List of certifications (if any)
    - summary: A concise 2-3 sentence professional summary suitable for candidate matching

    Output only valid JSON matching this structure. If information is not available, use appropriate defaults.

    Resume:
    {resume_text}
    """
    resp = co.chat(
        messages=[{"role": "user", "content": prompt}],
        model="command-r-plus-08-2024",
        max_tokens=500
    )
    
    # Parse the JSON response
    import json
    try:
        parsed = json.loads(resp.message.content[0].text.strip())
        return ParsedResume(**parsed)
    except:
        # Fallback if JSON parsing fails
        return ParsedResume(
            name="Unknown",
            summary=resp.message.content[0].text.strip()
        )

# Extract skills
@app.post("/extract-skills")
def extract_skills(req: JDRequest):
    prompt = f"""
    Extract exact technical + soft skills from this Job Description.
    Output strictly as a JSON list.

    JD:
    {req.job_description}
    """
    resp = co.chat(
        messages=[{"role": "user", "content": prompt}],
        model="command-r-plus-08-2024",
        max_tokens=200
    )
    return {"skills": resp.message.content[0].text.strip()}

# Generate interview questions
@app.post("/generate-questions")
def generate_questions(req: JDRequest):
    prompt = f"""
    Create 10 interview questions strictly based on this JD.
    Return a JSON list of questions only.

    JD:
    {req.job_description}
    """
    resp = co.chat(
        messages=[{"role": "user", "content": prompt}],
        model="command-r-plus-08-2024",
        max_tokens=300
    )
    return {"questions": resp.message.content[0].text.strip()}

# Match candidate
@app.post("/match-candidate")
async def match_candidate(job_description: str = Body(...), candidate_profile: str = Body(None), resume_file: UploadFile = File(None)):
    # If resume file is provided, parse it first
    if resume_file:
        # Read file content
        content = await resume_file.read()
        filename = resume_file.filename.lower()
        if filename.endswith('.pdf'):
            # Extract text from PDF
            with pdfplumber.open(BytesIO(content)) as pdf:
                resume_text = ''
                for page in pdf.pages:
                    resume_text += page.extract_text() or ''
        else:
            # Assume text file
            try:
                resume_text = content.decode('utf-8')
            except UnicodeDecodeError:
                resume_text = content.decode('latin-1')  # Fallback encoding
        
        # Parse resume using LLM
        parse_prompt = f"""
        Parse this resume text and extract the following information in a clean, HR-friendly JSON format:

        Required fields:
        - name: Full name of the candidate
        - email: Email address (if available, otherwise empty string)
        - phone: Phone number (if available, otherwise empty string)
        - experience_years: Total years of professional experience (integer)
        - skills: List of technical and soft skills mentioned
        - education: Highest education qualification (string)
        - certifications: List of certifications (if any)
        - summary: A concise 2-3 sentence professional summary suitable for candidate matching

        Output only valid JSON matching this structure. If information is not available, use appropriate defaults.

        Resume:
        {resume_text}
        """
        parse_resp = co.chat(
            messages=[{"role": "user", "content": parse_prompt}],
            model="command-r-plus-08-2024",
            max_tokens=500
        )
        
        # Parse the JSON response
        import json
        try:
            parsed = json.loads(parse_resp.message.content[0].text.strip())
            candidate_profile = parsed.get('summary', resume_text)  # Use summary for matching
        except:
            candidate_profile = resume_text  # Fallback to original text
    
    elif not candidate_profile:
        raise HTTPException(status_code=400, detail="Either candidate_profile or resume_file must be provided")
    
    prompt = f"""
    Compare the job description with the candidate profile.
    Provide:
    - Match percentage (0-100)
    - Missing skills
    - Key strengths
    - Overall assessment
    Output strictly in clean JSON.

    JD: {job_description}
    Candidate: {candidate_profile}
    """
    resp = co.chat(
        messages=[{"role": "user", "content": prompt}],
        model="command-r-plus-08-2024",
        max_tokens=400
    )
    return {"match_result": resp.message.content[0].text.strip()}

# Generate offer email
@app.post("/generate-offer-email")
def generate_offer_email(req: OfferEmailRequest):
    prompt = f"""
    Write a professional and warm HR offer email.

    Candidate Name: {req.candidate_name}
    Job Title: {req.job_title}
    Salary: {req.salary}
    Joining Date: {req.joining_date}
    HR Name: {req.hr_name}

    Output as plain text. Include Subject, Greeting, Body, and Closing.
    """
    resp = co.chat(
        messages=[{"role": "user", "content": prompt}],
        model="command-r-plus-08-2024",
        max_tokens=350
    )
    return {"offer_email": resp.message.content[0].text.strip()}

# Generate rejection email
@app.post("/generate-rejection-email")
def generate_rejection_email(req: RejectionEmailRequest):
    prompt = f"""
    Write a polite and professional HR rejection email.

    Candidate Name: {req.candidate_name}
    Job Title: {req.job_title}
    Reason: {req.reason}
    HR Name: {req.hr_name}

    Output as plain text. Include Subject, Greeting, Body, and Closing.
    """
    resp = co.chat(
        messages=[{"role": "user", "content": prompt}],
        model="command-r-plus-08-2024",
        max_tokens=250
    )
    return {"rejection_email": resp.message.content[0].text.strip()}

@app.post("/bulk-candidate-match")
def bulk_candidate_match(req: BulkMatchRequest):
    candidates = req.candidates
    jd_text = req.job_description

    # Embed JD + all candidates
    texts = [jd_text] + candidates
    embeddings = _embed_texts(texts)
    jd_emb = embeddings[0]
    candidate_embs = embeddings[1:]

    # Compute cosine similarity
    ranking = []
    for i, emb in enumerate(candidate_embs):
        score = cosine_sim(jd_emb, emb) * 100  # 0-100 scale
        ranking.append({
            "candidate": candidates[i],
            "score": round(score, 2)
        })

    # Sort descending by score
    ranking_sorted = sorted(ranking, key=lambda x: x["score"], reverse=True)

    return {"ranking": ranking_sorted}