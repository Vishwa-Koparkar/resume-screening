import json
import time
import streamlit as st
from google import genai
from google.genai import types

MODEL = "gemini-3.5-flash-lite"

CATEGORIES = [
    "Data Science", "Java Developer", "Python Developer", "Web Designing", "HR", "Hadoop",
    "DevOps Engineer", "Testing", "Automation Testing", "Mechanical Engineer", "Sales",
    "Operations Manager", "ETL Developer", "Blockchain", "Arts", "Database", "Health and fitness",
    "Electrical Engineering", "PMO", "Business Analyst", "DotNet Developer",
    "Network Security Engineer", "Civil Engineer", "SAP Developer", "Advocate",
]


def get_client(api_key):
    return genai.Client(api_key=api_key)


def ask_llm(client, prompt, attempts=3):
    """Call Gemini and parse the JSON answer. Retries on rate limits, network errors and bad JSON."""
    for attempt in range(attempts):
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0),
            )
            return json.loads(response.text)
        except Exception:
            if attempt == attempts - 1:
                raise
            time.sleep(2 ** (attempt + 1))


def clean_score(value):
    """Turn the LLM's match score into an integer between 0 and 100."""
    try:
        return max(0, min(100, int(float(str(value).strip().rstrip("%")))))
    except ValueError:
        return 0


@st.cache_data(show_spinner=False)
def analyze_resume(_client, resume_text):
    prompt = f"""You are an expert HR recruiter. Read the resume below and return JSON with these keys:
"summary": a 3-4 sentence professional summary of the candidate written in third person, covering
their education, main skills, strongest projects or experience, and the kind of role they suit
"highlights": list of 3-4 key achievements from the resume, keeping any numbers or results
"category": the single best job category for this resume, chosen only from this list: {CATEGORIES}
"strengths": list of 3 short strengths of the resume
"improvements": list of 3 short, specific suggestions to improve the resume
"interview_questions": list of 5 technical interview questions based on the resume

Resume:
{resume_text[:6000]}"""
    return ask_llm(_client, prompt)


@st.cache_data(show_spinner=False)
def recommend_jobs(_client, resume_text, jobs_text, top_n=5):
    prompt = f"""You are a job recommendation system. Compare the resume with every job below
and pick the {top_n} most suitable jobs for this candidate.

Return JSON with one key "recommendations": a list of {top_n} objects sorted from best to worst match,
each with these keys:
"job_id": the job id from the list
"match_score": integer from 0 to 100 showing how well the resume fits the job
"reason": one sentence explaining why this job suits the candidate

Jobs:
{jobs_text}

Resume:
{resume_text[:6000]}"""
    recommendations = ask_llm(_client, prompt)["recommendations"]
    for rec in recommendations:
        rec["match_score"] = clean_score(rec.get("match_score"))
    return recommendations


@st.cache_data(show_spinner=False)
def screen_resume(_client, resume_text, job_title, job_description):
    prompt = f"""You are an HR recruiter screening a candidate for this job.

Job title: {job_title}
Job description: {job_description}

Judge the overall fit (skills, projects, experience level and domain) and return JSON with these keys:
"match_score": integer from 0 to 100
"reason": one sentence explaining the score

Resume:
{resume_text[:6000]}"""
    result = ask_llm(_client, prompt)
    result["match_score"] = clean_score(result.get("match_score"))
    return result
