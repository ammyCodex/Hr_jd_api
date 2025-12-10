import os
import numpy as np
from typing import List, Dict
import cohere
from sklearn.preprocessing import normalize
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

COHERE_API_KEY = os.getenv("COHERE_API_KEY", "")
co = cohere.ClientV2(api_key=COHERE_API_KEY)

# -----------------------------
# Matching Logic with Embeddings
# -----------------------------

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
    candidate_profile: str

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
class BulkMatchRequest(BaseModel):
    job_description: str
    candidates: List[str]  # each candidate profile as string

# -----------------------------
# Endpoints
# -----------------------------

@app.get("/")
def root():
    return {"message": "HR Agent API running successfully!"}

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
def match_candidate(req: MatchRequest):
    prompt = f"""
    Compare the job description with the candidate profile.
    Provide:
    - Match percentage
    - Missing skills
    - Strengths
    Output strictly in clean JSON.

    JD: {req.job_description}
    Candidate: {req.candidate_profile}
    """
    resp = co.chat(
        messages=[{"role": "user", "content": prompt}],
        model="command-r-plus-08-2024",
        max_tokens=300
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