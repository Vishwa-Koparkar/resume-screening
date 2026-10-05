import io
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize

for path, pkg in [("tokenizers/punkt", "punkt"), ("tokenizers/punkt_tab", "punkt_tab"),
                  ("corpora/stopwords", "stopwords"), ("corpora/wordnet.zip", "wordnet")]:
    try:
        nltk.data.find(path)
    except LookupError:
        nltk.download(pkg, quiet=True)

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

SECTION_HEADINGS = {
    "Summary": ["summary", "objective", "career objective", "profile", "about me"],
    "Education": ["education", "academic details", "qualification", "qualifications"],
    "Experience": ["experience", "work experience", "internship", "internships", "employment"],
    "Projects": ["projects", "academic projects", "personal projects"],
    "Skills": ["skills", "technical skills", "key skills", "skill details"],
    "Certifications": ["certifications", "certificates", "achievements"],
}

BASE_DIR = Path(__file__).parent

with open(BASE_DIR / "data" / "skills.txt") as f:
    SKILLS = [line.strip().lower() for line in f if line.strip()]

# Other ways of writing a skill, so "ML" in a resume matches "machine learning" in a job.
ALIASES = {
    "machine learning": ["ml"], "deep learning": ["dl"], "natural language processing": ["nlp"],
    "named entity recognition": ["ner"], "javascript": ["js", "es6"], "typescript": ["ts"],
    "node.js": ["node", "nodejs", "node js"], "express.js": ["expressjs"],
    "react": ["react.js", "reactjs"], "next.js": ["nextjs"], "vue": ["vue.js", "vuejs"],
    "html": ["html5"], "css": ["css3"], "postgresql": ["postgres"], "mongodb": ["mongo"],
    "kubernetes": ["k8s"], "scikit-learn": ["sklearn", "scikit learn"],
    "aws": ["amazon web services"], "gcp": ["google cloud", "google cloud platform"],
    "azure": ["microsoft azure"], "ci/cd": ["cicd", "ci cd"],
    "rest api": ["rest apis", "restful api", "restful apis"],
    "vector databases": ["vector database", "vector db", "vector store"],
    "llm": ["llms", "large language model", "large language models"],
    "rag": ["retrieval augmented generation", "retrieval-augmented generation"],
    "hugging face": ["huggingface"], "power bi": ["powerbi"], "c#": ["csharp"],
    "sql server": ["mssql"], "pl/sql": ["plsql"], "asp.net": ["asp net"], "sap abap": ["abap"],
    "data visualization": ["data visualisation"], "a/b testing": ["ab testing", "a/b tests"],
    "ms project": ["microsoft project"], "spring boot": ["springboot"],
    "microservices": ["microservice"], "firewalls": ["firewall"], "smart contracts": ["smart contract"],
    "test cases": ["test case"], "test plans": ["test plan"], "user stories": ["user story"],
}
ALIAS_TO_SKILL = {alias: skill for skill, names in ALIASES.items() for alias in names}

OPTIONAL_MARKERS = r"\b(plus|desirable|preferred|nice to have|good to have|bonus)\b"


def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|\S+@\S+", " ", text)
    text = re.sub(r"[^a-z0-9+# ]", " ", text)
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words and len(t) > 1]
    return " ".join(tokens)


def has_skill(skill, text):
    """True if the skill or any of its aliases appears in the (lowercase) text."""
    for name in [skill] + ALIASES.get(skill, []):
        if re.search(r"(?<![a-z0-9])" + re.escape(name) + r"(?![a-z0-9])", text):
            return True
    return False


def extract_skills(text):
    text = text.lower()
    return [skill for skill in SKILLS if has_skill(skill, text)]


def extract_required_skills(description):
    """Skills in a job description, ignoring sentences that only call them a plus or preferred."""
    sentences = re.split(r"(?<=[.!?])\s+", description)
    required = [s for s in sentences if not re.search(OPTIONAL_MARKERS, s, re.I)]
    return extract_skills(" ".join(required))


def extract_email(text):
    match = re.search(r"[\w.+-]+@[\w-]+\.[\w.]+", text)
    return match.group() if match else ""


def extract_phone(text):
    match = re.search(r"(\+91[\s-]?)?[6-9]\d{4}[\s-]?\d{5}", text)
    return match.group() if match else ""


def extract_name(text):
    headings = [w for words in SECTION_HEADINGS.values() for w in words]
    for line in text.strip().splitlines()[:5]:
        line = line.strip()
        words = line.replace(".", " ").split()
        if 2 <= len(words) <= 4 and all(w.isalpha() for w in words) and line.lower() not in headings:
            return line.title()
    return ""


def extract_links(text):
    links = {}
    github = re.search(r"(https?://)?(www\.)?github\.com/[\w-]+", text, re.I)
    linkedin = re.search(r"(https?://)?(www\.)?linkedin\.com/in/[\w-]+", text, re.I)
    if github:
        links["GitHub"] = github.group()
    if linkedin:
        links["LinkedIn"] = linkedin.group()
    return links


DEGREES = r"\b(B\.?\s?Tech|B\.?E\.?|B\.?Sc|BCA|BBA|B\.?Com|M\.?\s?Tech|M\.?E\.?|M\.?Sc|MCA|MBA|Ph\.?D|Diploma|HSC|SSC)\b"


def extract_education(text):
    degrees = []
    for line in text.splitlines():
        if re.search(DEGREES, line) and len(line.split()) <= 20:
            degrees.append(line.strip())
    cgpa = re.search(r"(CGPA|GPA|CPI)\s*[:\-]?\s*(\d{1,2}(\.\d{1,2})?)", text, re.I)
    percentage = re.search(r"(\d{2}(\.\d{1,2})?)\s?%", text)
    return {
        "degrees": list(dict.fromkeys(degrees))[:3],
        "cgpa": cgpa.group(2) if cgpa else "",
        "percentage": percentage.group(1) + "%" if percentage and not cgpa else "",
    }


def extract_experience_years(text):
    experience_text = extract_sections(text).get("Experience", "")
    current_year = datetime.now().year

    # Years the resume states directly, e.g. "2 years of experience"
    stated = [float(y) for y in re.findall(r"(\d+(?:\.\d+)?)\+?\s*(?:years?|yrs?)\b", text, re.I)]

    # Date ranges in the experience section; overlapping ranges are merged so no year is counted twice
    ranges = []
    for start, end in re.findall(r"((?:19|20)\d\d)\s*(?:-|–|—|to)\s*((?:19|20)\d\d|present|current)",
                                 experience_text, re.I):
        end_year = current_year if end.lower() in ("present", "current") else int(end)
        if end_year > int(start):
            ranges.append((int(start), end_year))
    dated_years, last_end = 0, None
    for start, end in sorted(ranges):
        if last_end is None or start > last_end:
            dated_years += end - start
            last_end = end
        elif end > last_end:
            dated_years += end - last_end
            last_end = end

    # Durations like "3 months" in the experience section (internships) are added up
    months = sum(float(m) / 12 for m in re.findall(r"(\d+)\s*months?\b", experience_text, re.I))

    return round(max(stated + [dated_years + months]), 1)


def experience_level(years):
    if years < 1:
        return "Fresher"
    if years < 3:
        return "Junior"
    if years < 6:
        return "Mid-level"
    return "Senior"


def parse_skill_list(text):
    return [s.strip().lower() for s in re.split(r"[,;\n]", text) if s.strip()]


def skill_gap(resume_text, required_skills):
    resume = resume_text.lower()
    required_skills = list(dict.fromkeys(ALIAS_TO_SKILL.get(s.lower(), s.lower()) for s in required_skills))
    matched, missing = [], []
    for skill in required_skills:
        if has_skill(skill, resume):
            matched.append(skill)
        else:
            missing.append(skill)
    return matched, missing


def extract_sections(text):
    sections = {}
    current = None
    for line in text.splitlines():
        heading = line.strip().lower().rstrip(":")
        found = None
        for section, words in SECTION_HEADINGS.items():
            if heading in words:
                found = section
        if found:
            current = found
            sections[current] = []
        elif current and line.strip():
            sections[current].append(line.strip())
    return {k: "\n".join(v) for k, v in sections.items() if v}


def split_sentences(text):
    text = re.sub(r"[•●▪◦■]", "\n", text)
    chunks = []
    for line in text.splitlines():
        line = line.strip(" -*\t")
        if not line:
            continue
        if chunks and not chunks[-1].endswith((".", "!", "?", ":")) and line[0].islower():
            chunks[-1] += " " + line
        else:
            chunks.append(line)
    sentences = []
    for chunk in chunks:
        sentences.extend(sent_tokenize(chunk))
    return sentences


def is_good_sentence(sentence):
    words = sentence.split()
    if len(words) < 8 or len(words) > 45:
        return False
    if re.search(r"http|www\.|@|github\.com|linkedin", sentence.lower()):
        return False
    capitalized = sum(1 for w in words if w[0].isupper())
    return capitalized / len(words) < 0.6


def summarize(text, n=3):
    sentences = list(dict.fromkeys(s for s in split_sentences(text) if is_good_sentence(s)))
    freq = Counter(clean_text(text).split())
    if not sentences or not freq:
        return []
    top_freq = max(freq.values())
    scores = {}
    for i, sentence in enumerate(sentences):
        lemmas = clean_text(sentence).split()
        scores[i] = sum(freq[w] / top_freq for w in lemmas) / (len(lemmas) + 1)
    best = sorted(sorted(scores, key=scores.get, reverse=True)[:n])
    return [sentences[i] for i in best]


def read_resume(file):
    data = file.getvalue()
    if file.name.lower().endswith(".pdf"):
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(data))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    return data.decode("utf-8", errors="ignore")
