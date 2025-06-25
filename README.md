# ATS-Resume-Checker App

An intelligent resume matcher that compares your resume to a job description using both strict and semantic keyword extraction. Features include:

- Match scoring via TF-IDF
- Phrase matching and semantic similarity using spaCy and SentenceTransformers
- Categorized keyword feedback
- Tailored resume summary suggestions by job type
- 📄 PDF report generation
- ✍️ GPT-based summary rewrite (requires OpenAI API Key)

## 🚀 Setup

1. Clone or download this project.
2. Install dependencies:

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
