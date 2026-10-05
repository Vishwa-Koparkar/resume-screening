import os
from pathlib import Path
import pandas as pd
import streamlit as st
from nlp_utils import (extract_skills, extract_required_skills, extract_email, extract_phone, extract_name, extract_links,
                       extract_education, extract_experience_years, experience_level, extract_sections,
                       parse_skill_list, skill_gap, summarize, read_resume)
from llm import get_client, analyze_resume, recommend_jobs, screen_resume, MODEL

st.set_page_config(page_title="Resume Screening", layout="wide")
st.title("Resume Screening and Job Recommendation")

api_key = st.sidebar.text_input("Gemini API Key", value=os.environ.get("GEMINI_API_KEY", ""), type="password")
st.sidebar.write("Model:", MODEL)
if not api_key:
    st.warning("Enter your Gemini API key in the sidebar to start.")
    st.stop()
client = get_client(api_key)

jobs = pd.read_csv(Path(__file__).parent / "data" / "jobs.csv")
jobs["skills"] = jobs["description"].apply(extract_required_skills)
jobs_text = "\n".join(f"{j.job_id} | {j.title} | {j.description}" for j in jobs.itertuples())


def custom_job_inputs(key):
    title = st.text_input("Job Title", key=f"{key}_title", placeholder="e.g. Machine Learning Intern")
    col1, col2 = st.columns(2)
    company = col1.text_input("Company (optional)", key=f"{key}_company")
    experience = col2.text_input("Experience Required (optional)", key=f"{key}_exp", placeholder="e.g. 0-2 years")
    skills = st.text_input("Required Skills (comma separated)", key=f"{key}_skills",
                           placeholder="e.g. Python, SQL, Machine Learning")
    description = st.text_area("Job Description", key=f"{key}_desc", height=150,
                               placeholder="Paste or type the job description here")
    details = description
    if company:
        details += f"\nCompany: {company}"
    if experience:
        details += f"\nExperience required: {experience}"
    if skills:
        details += f"\nRequired skills: {skills}"
    required = parse_skill_list(skills) or extract_required_skills(description)
    return {"title": title.strip(), "company": company.strip(), "details": details.strip(),
            "description": description.strip(), "required": required}


def show_match_card(title, subtitle, ai_score, reason, matched, missing):
    with st.container(border=True):
        col1, col2 = st.columns([4, 1])
        col1.subheader(title)
        if subtitle:
            col1.write(subtitle)
        col2.metric("Match", f"{ai_score}%")
        st.write("**Why:**", reason)
        st.write("**Matched Skills:**", ", ".join(matched) or "-")
        st.write("**Skills to Learn:**", ", ".join(missing) or "-")


tab1, tab2 = st.tabs(["For Candidates", "For Recruiters"])

with tab1:
    uploaded = st.file_uploader("Upload your resume (PDF or TXT)", type=["pdf", "txt"])
    if uploaded:
        text = read_resume(uploaded)
        with st.spinner("Analyzing resume with the LLM..."):
            try:
                analysis = analyze_resume(client, text)
                recommendations = recommend_jobs(client, text, jobs_text)
            except Exception as e:
                st.error(f"LLM request failed: {e}")
                st.stop()

        st.header("1. Extracted Details (NLP)")
        years = extract_experience_years(text)
        education = extract_education(text)
        links = extract_links(text)
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Name:**", extract_name(text) or "-")
            st.write("**Email:**", extract_email(text) or "-")
            st.write("**Phone:**", extract_phone(text) or "-")
            for site, link in links.items():
                st.write(f"**{site}:**", link)
            st.write("**Experience:**", f"{years} years ({experience_level(years)})")
            st.write("**Predicted Category (LLM):**", analysis["category"])
        with col2:
            st.write("**Education:**")
            for degree in education["degrees"]:
                st.write("-", degree)
            if not education["degrees"]:
                st.write("-")
            if education["cgpa"]:
                st.write("**CGPA:**", education["cgpa"])
            if education["percentage"]:
                st.write("**Percentage:**", education["percentage"])
            skills = extract_skills(text)
            st.write(f"**Skills Found ({len(skills)}):**", ", ".join(skills))

        with st.expander("2. Resume Sections (NLP)"):
            sections = extract_sections(text)
            if sections:
                for name, content in sections.items():
                    st.markdown(f"**{name}**")
                    st.text(content)
            else:
                st.write("No section headings found in this resume.")

        with st.expander("3. Resume Summary"):
            st.subheader("Abstractive Summary (LLM)")
            st.info(analysis["summary"])
            st.write("**Key Highlights:**")
            for h in analysis["highlights"]:
                st.write("-", h)
            st.subheader("Extractive Summary (NLP)")
            st.caption("Top sentences picked from the resume by word-frequency scoring after "
                       "tokenization, stopword removal and lemmatization.")
            sentences = summarize(text)
            for sentence in sentences:
                st.write("-", sentence)
            if not sentences:
                st.write("No complete sentences found in this resume.")

        st.header("4. Top 5 Recommended Jobs")
        st.caption("The LLM picks the jobs and gives the match score. Matched skills and skills to learn "
                   "are found with NLP skill extraction.")
        job_info = jobs.set_index("job_id")
        for rec in recommendations:
            if rec["job_id"] not in job_info.index:
                continue
            job = job_info.loc[rec["job_id"]]
            matched, missing = skill_gap(text, job["skills"])
            show_match_card(job["title"], f"{job['company']} | {job['location']} | {job['experience']}",
                            rec["match_score"], rec["reason"], matched, missing)

        st.header("5. Check Match for Your Own Job")
        st.write("Have a specific job in mind? Enter its details to see how well your resume fits.")
        with st.form("own_job_form"):
            own = custom_job_inputs("own")
            submitted = st.form_submit_button("Check Match")
        if submitted:
            if not own["title"] or not own["description"]:
                st.warning("Please enter at least the job title and job description.")
            else:
                with st.spinner("Comparing your resume with this job..."):
                    try:
                        result = screen_resume(client, text, own["title"], own["details"])
                        st.session_state["own_result"] = (uploaded.name, own, result)
                    except Exception as e:
                        st.error(f"LLM request failed: {e}")
        saved = st.session_state.get("own_result")
        if saved and saved[0] == uploaded.name:
            own, result = saved[1], saved[2]
            matched, missing = skill_gap(text, own["required"])
            show_match_card(own["title"], own["company"], result["match_score"], result["reason"],
                            matched, missing)

        st.header("6. AI Resume Feedback (LLM)")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Strengths")
            for s in analysis["strengths"]:
                st.write("-", s)
        with col2:
            st.subheader("Improvements")
            for s in analysis["improvements"]:
                st.write("-", s)
        st.subheader("Likely Interview Questions")
        for i, q in enumerate(analysis["interview_questions"], 1):
            st.write(f"{i}. {q}")

with tab2:
    st.subheader("Job Details")
    job = custom_job_inputs("screen")

    st.subheader("Candidate Resumes")
    files = st.file_uploader("Upload resumes", type=["pdf", "txt"], accept_multiple_files=True)
    threshold = st.slider("Shortlist threshold (%)", 0, 100, 60)

    if files and (not job["title"] or not job["description"]):
        st.warning("Please enter at least the job title and job description.")
    elif files:
        rows = []
        progress = st.progress(0, text="Screening resumes...")
        for i, file in enumerate(files):
            text = read_resume(file)
            try:
                result = screen_resume(client, text, job["title"], job["details"])
            except Exception as e:
                st.error(f"LLM request failed for {file.name}: {e}")
                continue
            matched, missing = skill_gap(text, job["required"])
            status = "Shortlisted" if result["match_score"] >= threshold else "Rejected"
            rows.append([extract_name(text) or file.name, extract_email(text), extract_experience_years(text),
                         result["match_score"], ", ".join(matched), ", ".join(missing),
                         result["reason"], status])
            progress.progress((i + 1) / len(files), text=f"Screened {i + 1} of {len(files)}")
        if rows:
            ranked = pd.DataFrame(rows, columns=["Name", "Email", "Experience (yrs)", "Match %",
                                                 "Matched Skills", "Missing Skills", "Reason", "Status"])
            ranked = ranked.sort_values("Match %", ascending=False)
            st.dataframe(ranked, hide_index=True)
            st.bar_chart(ranked.set_index("Name")["Match %"])
            st.write(f"**{(ranked['Status'] == 'Shortlisted').sum()} of {len(ranked)} resumes shortlisted**")