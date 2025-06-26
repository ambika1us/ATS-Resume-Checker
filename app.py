# ats_resume_matcher.py

import os
import base64
import subprocess
import streamlit as st
import docx2txt
import fitz  # PyMuPDF
import spacy
from spacy.matcher import PhraseMatcher
from sentence_transformers import SentenceTransformer, util
from fpdf import FPDF
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
from openai import RateLimitError
import spacy.cli


# === App Config ===
st.set_page_config("ATS Resume Matcher", layout="centered")
st.title("📄 Resume vs Job Description Matcher")
st.caption("Upload your resume and paste a job description to evaluate match, get feedback, and optimize.")

# === Credential Setup ===
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# === Load Models ===
@st.cache_resource
def load_models():
    try:

        spacy.cli.download("en_core_web_sm")
        return SentenceTransformer("all-MiniLM-L6-v2"), spacy.load("en_core_web_sm")

    except OSError:
        subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
        return SentenceTransformer("all-MiniLM-L6-v2"), spacy.load("en_core_web_sm")

sbert_model, nlp = load_models()

# === Keyword Category Map ===
keyword_categories = {
    "bigquery": "Tool", "snowflake": "Tool", "redshift": "Tool", "postgres": "Tool", "dbt": "Tool",
    "pipelines": "Concept", "reportingstack": "Tool", "looker": "Tool", "metabase": "Tool",
    "domo": "Tool", "power": "Tool", "microsoft": "Tool", "studio": "Tool", "visualization": "Tool",
    "dashboards": "Tool", "platforms": "Tool", "statistics": "Concept", "statistical": "Concept",
    "structured": "Concept", "insight": "Concept", "intelligence": "Concept", "kpis": "Concept",
    "interpretation": "Concept", "reporting": "Concept", "analytics": "Concept",
    "communication": "Soft Skill", "stakeholders": "Soft Skill", "concisely": "Soft Skill",
    "comfortable": "Soft Skill", "clearly": "Soft Skill", "guide": "Soft Skill",
    "informative": "Soft Skill", "responsibilities": "Soft Skill", "consulting": "Domain",
    "bcg": "Domain", "fortune": "Domain", "startups": "Domain", "firm": "Domain", "domain": "Domain",
    "remote": "Work Style", "hybrid": "Work Style", "currently": "Work Style", "immediately": "Work Style",
    "degrees": "Credential", "knowledge": "Credential", "engineering": "Credential"
}

# === Utility Functions ===
def extract_text(file):
    if file.name.endswith(".pdf"):
        doc = fitz.open(stream=file.read(), filetype="pdf")
        return " ".join([page.get_text() for page in doc])
    elif file.name.endswith(".docx"):
        return docx2txt.process(file)
    return ""

def extract_keywords(text):
    doc = nlp(text.lower())
    return set([token.text for token in doc if token.pos_ in ["NOUN", "PROPN"] and not token.is_stop and token.is_alpha and len(token.text) > 2])

def phrase_match(text, phrases):
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    patterns = [nlp.make_doc(p) for p in phrases]
    matcher.add("PHRASES", patterns)
    doc = nlp(text)
    return set([doc[start:end].text.lower() for _, start, end in matcher(doc)])

def semantic_match(jd_keywords, resume_keywords, threshold=0.8):
    matched = set()
    for jd_kw in jd_keywords:
        jd_vec = sbert_model.encode(jd_kw, convert_to_tensor=True)
        for res_kw in resume_keywords:
            res_vec = sbert_model.encode(res_kw, convert_to_tensor=True)
            if util.pytorch_cos_sim(jd_vec, res_vec).item() >= threshold:
                matched.add(jd_kw)
                break
    return matched

def compute_similarity(resume, jd):
    vectorizer = TfidfVectorizer(stop_words="english")
    vectors = vectorizer.fit_transform([resume, jd])
    return round(cosine_similarity(vectors[0:1], vectors[1:2])[0][0] * 100, 2)

def detect_company_context(jd_text):
    jd = jd_text.lower()
    if any(k in jd for k in ["startup", "nimble", "zero to one"]):
        return "Startup"
    elif any(k in jd for k in ["consulting", "client-facing", "advisory"]):
        return "Consulting"
    elif any(k in jd for k in ["product", "roadmap", "user-first"]):
        return "Product"
    return "General"

def group_keywords(keywords):
    grouped = defaultdict(list)
    for kw in keywords:
        grouped[keyword_categories.get(kw.lower(), "Other")].append(kw)
    return grouped

def generate_pdf(score, keywords):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Resume Match Report", ln=True, align="C")
    pdf.ln(10)
    pdf.cell(200, 10, txt=f"Match Score: {score}%", ln=True)
    pdf.ln(5)
    pdf.multi_cell(0, 10, txt="Missing Keywords:\n" + ", ".join(keywords))
    return pdf.output(dest='S').encode('latin-1')

from openai import RateLimitError

def rewrite_resume(text, missing_keywords, client):
    if not text.strip():
        return "❗ Please enter your current resume summary to rewrite."

    prompt = f"Rewrite the following resume summary to include these keywords: {', '.join(missing_keywords[:12])}. Keep it professional and natural.\n\nOriginal:\n{text}\n\nImproved:"

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        return response.choices[0].message.content.strip()

    except RateLimitError:
        st.error("🚫 You've exceeded your OpenAI quota. Please check your billing settings.")
        return "❌ Unable to generate improved summary due to quota limits."

# === UI Layout ===
uploaded_resume = st.file_uploader("📎 Upload Resume", type=["pdf", "docx"])
jd_text = st.text_area("📝 Paste Job Description")
match_mode = st.radio("🔄 Matching Mode", ["Strict (Exact)", "Semantic (Fuzzy)"], horizontal=True)

if uploaded_resume and jd_text:
    resume_text = extract_text(uploaded_resume)
    resume_keywords = extract_keywords(resume_text)
    jd_keywords = extract_keywords(jd_text)

    matched = resume_keywords if match_mode == "Strict (Exact)" else semantic_match(jd_keywords, resume_keywords)
    matched.update(phrase_match(resume_text, ["data pipeline", "stakeholder communication"]))

    missing = sorted(jd_keywords - matched)
    match_score = compute_similarity(resume_text, jd_text)
    context = detect_company_context(jd_text)
    grouped = group_keywords(missing)

    st.success(f"✅ Match Score: **{match_score}%**")

    if grouped:
        st.markdown("### 🧩 Missing Keywords by Category")
        for cat, kws in grouped.items():
            st.markdown(f"**{cat}**: {', '.join(sorted(set(kws)))}")

    summaries = {
        "Startup": "🚀 *Fast-moving data professional energized by building from zero to one...*",
        "Consulting": "📊 *Strategic, data-driven analyst experienced in delivering insights...*",
        "Product": "🛠️ *Analyst passionate about building data products that inform decisions...*",
        "General": "📈 *Data professional skilled in analytics, dashboards, and BI strategy...*"
    }

    st.markdown("### ✨ Tailored Resume Summary Suggestion")
    st.info(summaries.get(context, summaries["General"]))

    st.markdown("### 📄 Downloadable PDF Report")
    if st.button("Generate Match Report PDF"):
        pdf_bytes = generate_pdf(match_score, missing)
        b64 = base64.b64encode(pdf_bytes).decode()
        href = f'<a href="data:application/pdf;base64,{b64}" download="match_report.pdf">📥 Download Match Report</a>'
        st.markdown(href, unsafe_allow_html=True)

    st.markdown("### ✍️ GPT-Based Resume Summary Rewrite")
    original = st.text_area("Paste your current resume summary to improve", height=150)
    if st.button("Rewrite with GPT"):
        with st.spinner("Crafting your improved summary..."):
            improved = rewrite_resume(original, missing, client)
            st.markdown("#### Improved Summary")
            st.write(improved)