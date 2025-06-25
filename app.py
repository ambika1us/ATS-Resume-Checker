import streamlit as st
import docx2txt
import fitz  # PyMuPDF
import spacy
from spacy.matcher import PhraseMatcher
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util
from fpdf import FPDF
import base64
import openai
from collections import defaultdict

import spacy
import subprocess

def download_spacy_model():
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
        return spacy.load("en_core_web_sm")

# === Load Models ===
@st.cache_resource
def load_models():
    return SentenceTransformer("all-MiniLM-L6-v2"), download_spacy_model()

sbert_model, nlp = load_models()

# === Keyword Categories ===
keyword_categories = {
    "bigquery": "Tool", "snowflake": "Tool", "redshift": "Tool", "postgres": "Tool",
    "dbt": "Tool", "pipelines": "Concept", "reportingstack": "Tool", "looker": "Tool",
    "metabase": "Tool", "domo": "Tool", "power": "Tool", "microsoft": "Tool", "studio": "Tool",
    "visualization": "Tool", "dashboards": "Tool", "platforms": "Tool", "statistics": "Concept",
    "statistical": "Concept", "structured": "Concept", "insight": "Concept", "intelligence": "Concept",
    "kpis": "Concept", "interpretation": "Concept", "reporting": "Concept", "analytics": "Concept",
    "communication": "Soft Skill", "stakeholders": "Soft Skill", "concisely": "Soft Skill",
    "comfortable": "Soft Skill", "clearly": "Soft Skill", "guide": "Soft Skill",
    "informative": "Soft Skill", "responsibilities": "Soft Skill", "consulting": "Domain",
    "bcg": "Domain", "fortune": "Domain", "startups": "Domain", "firm": "Domain", "domain": "Domain",
    "remote": "Work Style", "hybrid": "Work Style", "currently": "Work Style", "immediately": "Work Style",
    "degrees": "Credential", "knowledge": "Credential", "engineering": "Credential"
}

# === Utilities ===
def extract_text(file):
    if file.name.endswith(".pdf"):
        doc = fitz.open(stream=file.read(), filetype="pdf")
        return " ".join([page.get_text() for page in doc])
    elif file.name.endswith(".docx"):
        return docx2txt.process(file)
    return ""

def extract_keywords(text):
    doc = nlp(text.lower())
    return set([
        token.text for token in doc
        if token.pos_ in ["NOUN", "PROPN"] and not token.is_stop and token.is_alpha and len(token.text) > 2
    ])

def compute_similarity(resume, jd):
    vectorizer = TfidfVectorizer(stop_words='english')
    vectors = vectorizer.fit_transform([resume, jd])
    return round(cosine_similarity(vectors[0:1], vectors[1:2])[0][0] * 100, 2)

def semantic_match(jd_keywords, resume_keywords, threshold=0.8):
    matched = set()
    for jd_kw in jd_keywords:
        jd_vec = sbert_model.encode(jd_kw, convert_to_tensor=True)
        for res_kw in resume_keywords:
            res_vec = sbert_model.encode(res_kw, convert_to_tensor=True)
            sim = util.pytorch_cos_sim(jd_vec, res_vec).item()
            if sim >= threshold:
                matched.add(jd_kw)
                break
    return matched

def build_phrase_matcher(phrases):
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    patterns = [nlp.make_doc(p) for p in phrases]
    matcher.add("PHRASES", patterns)
    return matcher

def phrase_match(text, phrases):
    matcher = build_phrase_matcher(phrases)
    doc = nlp(text)
    return set([doc[start:end].text.lower() for _, start, end in matcher(doc)])

def group_keywords_by_category(keywords, category_map):
    grouped = defaultdict(list)
    for kw in keywords:
        label = category_map.get(kw.lower(), "Other")
        grouped[label].append(kw)
    return grouped

def detect_company_context(jd_text):
    jd = jd_text.lower()
    if any(w in jd for w in ["startup", "nimble", "zero to one"]):
        return "Startup"
    elif any(w in jd for w in ["consulting", "client-facing", "advisory"]):
        return "Consulting"
    elif any(w in jd for w in ["product", "roadmap", "user-first"]):
        return "Product"
    return "General"

def generate_pdf_report(score, missing_keywords):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Resume Match Report", ln=True, align="C")
    pdf.ln(10)
    pdf.cell(200, 10, txt=f"Match Score: {score}%", ln=True)
    pdf.ln(5)
    pdf.multi_cell(0, 10, txt="Missing Keywords:\n" + ", ".join(missing_keywords))
    return pdf.output(dest='S').encode('latin-1')

def rewrite_resume_section(text, missing_keywords):
    prompt = f"Rewrite the following resume summary to include these keywords: {', '.join(missing_keywords[:12])}. Keep it professional and natural.\n\nOriginal:\n{text}\n\nImproved:"
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    return response.choices[0].message["content"].strip()

# === Streamlit App ===
st.set_page_config("ATS Resume Matcher", layout="centered")
st.title("📄 Resume vs Job Description Matcher")
st.caption("Upload your resume and paste a job description to evaluate match, get feedback, and optimize.")

openai.api_key = st.secrets.get("OPENAI_API_KEY")

uploaded_resume = st.file_uploader("📎 Upload Resume", type=["pdf", "docx"])
jd_text = st.text_area("📝 Paste Job Description")
match_mode = st.radio("🔄 Matching Mode", ["Strict (Exact)", "Semantic (Fuzzy)"], horizontal=True)

if uploaded_resume and jd_text:
    resume_text = extract_text(uploaded_resume)
    resume_keywords = extract_keywords(resume_text)
    jd_keywords = extract_keywords(jd_text)

    if match_mode == "Strict (Exact)":
        matched = resume_keywords
    else:
        matched = semantic_match(jd_keywords, resume_keywords)

    matched.update(phrase_match(resume_text, ["data pipeline", "stakeholder communication"]))
    missing_keywords = sorted(jd_keywords - matched)
    grouped = group_keywords_by_category(missing_keywords, keyword_categories)
    match_score = compute_similarity(resume_text, jd_text)
    context = detect_company_context(jd_text)

    st.success(f"✅ Match Score: **{match_score}%**")

    if grouped:
        st.markdown("### 🧩 Missing Keywords by Category")
        for cat, kws in grouped.items():
            st.markdown(f"**{cat}**: " + ", ".join(sorted(set(kws))))

    st.markdown("### ✨ Tailored Resume Summary Suggestion")
    summaries = {
        "Startup": "🚀 *Fast-moving data professional energized by building from zero to one. Experienced in creating scalable analytics stacks using Redshift, Postgres, and open-source tools. Comfortable as data engineer, storyteller, and product partner driving growth at speed.*",
        "Consulting": "📊 *Strategic, data-driven analyst experienced in delivering insights to enterprise clients. Proficient with BigQuery, dbt, Looker. Trusted partner to stakeholders across hybrid teams solving complex problems.*",
        "Product": "🛠️ *Analyst passionate about building data products that inform user-centric decisions. Skilled in dashboarding, KPIs, growth metrics, and cloud data platforms like Snowflake and Metabase.*",
        "General": "📈 *Data professional with expertise in analytics, dashboards, and BI strategy. Skilled in translating data into clear business value using modern tools and stakeholder alignment.*"
    }
    st.info(summaries.get(context, summaries["General"]))

    st.markdown("### 📄 Downloadable PDF Report")
    if st.button("Generate Match Report PDF"):
        pdf_bytes = generate_pdf_report(match_score, missing_keywords)
        b64 = base64.b64encode(pdf_bytes).decode()
        href = f'<a href="data:application/pdf;base64,{b64}" download="match_report.pdf">📥 Download Match Report</a>'
        st.markdown(href, unsafe_allow_html=True)

    st.markdown("### ✍️ GPT-Based Resume Summary Rewrite")
    original = st.text_area("Paste your current resume summary to improve", height=150)
    if openai.api_key and st.button("Rewrite with GPT"):
        with st.spinner("Crafting your improved summary..."):
            improved = rewrite_resume_section(original, missing_keywords)
            st.markdown("####"*10)