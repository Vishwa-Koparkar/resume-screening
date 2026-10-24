import requests
import streamlit as st

API_URL = "https://jsearch.p.rapidapi.com/search-v2"
API_HOST = "jsearch.p.rapidapi.com"


def search_jobs(api_key, query, location, country):
    search = f"{query} in {location}" if location else query
    response = requests.get(
        API_URL,
        headers={"X-RapidAPI-Key": api_key.strip(), "X-RapidAPI-Host": API_HOST},
        params={"query": search, "country": country, "date_posted": "month", "num_pages": 1},
        timeout=30,
    )
    if not response.ok:
        raise RuntimeError(f"JSearch returned {response.status_code}: {response.text[:300]}")
    data = response.json().get("data") or []
    if isinstance(data, dict):
        # search-v2 nests the listings inside the data object instead of returning a bare list
        data = next((v for v in data.values() if isinstance(v, list)), [])
    return data


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_live_jobs(api_key, queries, location="", country="in", per_query=5):
    """Search current openings for each job title through JSearch, which collects listings from LinkedIn,
    Naukri, Indeed, Glassdoor, Unstop and other job sites. Returns a list of simple job dictionaries
    with duplicates removed."""
    jobs = []
    seen = set()
    for query in queries:
        for item in search_jobs(api_key, query, location, country)[:per_query]:
            key = (item.get("job_title"), item.get("employer_name"))
            if key in seen:
                continue
            seen.add(key)
            place = ", ".join(p for p in [item.get("job_city"), item.get("job_state"), item.get("job_country")] if p)
            options = item.get("apply_options") or []
            jobs.append({
                "job_id": f"L{len(jobs) + 1}",
                "title": item.get("job_title") or "",
                "company": item.get("employer_name") or "",
                "location": "Remote" if item.get("job_is_remote") else place,
                "type": (item.get("job_employment_type") or "").title(),
                "posted": (item.get("job_posted_at_datetime_utc") or "")[:10],
                "source": item.get("job_publisher") or "",
                "link": item.get("job_apply_link") or (options[0]["apply_link"] if options else ""),
                "description": item.get("job_description") or "",
            })
    return jobs
