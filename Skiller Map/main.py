from flask import Flask, render_template, request
import json
from werkzeug.utils import secure_filename
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
import pdfplumber
import openai
from github import Github
from dotenv import load_dotenv
import os

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


def extract_text_from_pdf(pdf_path):
    full_text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                full_text += text + "\n"
    return full_text


def ask_gpt_to_extract_skills(resume_text):
    prompt = f"""
Extract technical skills from the resume and return them as a valid JSON object with values from 1 to 5:
{{
  "Python": 4,
  "React": 5
}}
Resume Text:
{resume_text}
"""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You extract skills in JSON format from resumes."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=800,
        )
        content = response['choices'][0]['message']['content']

        def clean_gpt_response(gpt_response):
            if gpt_response.startswith("```"):
                gpt_response = "\n".join(gpt_response.split("\n")[1:])
            if gpt_response.endswith("```"):
                gpt_response = "\n".join(gpt_response.split("\n")[:-1])
            return gpt_response.strip()

        clean_content = clean_gpt_response(content)
        return json.loads(clean_content)
    except json.JSONDecodeError as e:
        print("JSON decode error:", e)
        print("GPT response was:", content)
        return {}
    except Exception as e:
        print("OpenAI API error:", e)
        return {}


def fetch_github_stats(username):
    g = Github(GITHUB_TOKEN)
    try:
        user = g.get_user(username)
        repos = user.get_repos()
    except Exception:
        return {}
    language_count = {}
    for repo in repos:
        try:
            langs = repo.get_languages()
            for lang, count in langs.items():
                language_count[lang] = language_count.get(lang, 0) + count
        except:
            continue
    return language_count


def generate_resume_skill_plot(resume_skills):
    skills = list(resume_skills.keys())
    levels = list(resume_skills.values())

    plt.figure(figsize=(10, 6))
    plt.barh(skills, levels, color='teal')
    plt.xlim(0, 5)
    plt.xlabel("Skill Level (Out of 5)")
    plt.title("Resume Skills Ranking")
    plt.grid(axis='x')
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def generate_github_language_plot(github_lang_stats):
    if not github_lang_stats:
        return None
    langs = list(github_lang_stats.keys())
    counts = list(github_lang_stats.values())

    plt.figure(figsize=(10, 6))
    plt.bar(langs, counts, color='coral')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel("Bytes of Code")
    plt.title("GitHub Language Usage")
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def generate_repo_language_count_plot(username):
    g = Github(GITHUB_TOKEN)
    try:
        user = g.get_user(username)
        repos = user.get_repos()
    except Exception:
        return None

    lang_repo_count = {}
    for repo in repos:
        lang = repo.language
        if lang:
            lang_repo_count[lang] = lang_repo_count.get(lang, 0) + 1

    if not lang_repo_count:
        return None

    langs = list(lang_repo_count.keys())
    counts = list(lang_repo_count.values())

    plt.figure(figsize=(10, 6))
    plt.bar(langs, counts, color='mediumseagreen')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel("Number of Repositories")
    plt.title("Number of Repositories by Language")
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def generate_skill_level_distribution_plot(resume_skills):
    levels = list(resume_skills.values())
    counts = [levels.count(i) for i in range(1, 6)]

    plt.figure(figsize=(8, 5))
    plt.bar(range(1, 6), counts, color='slateblue')
    plt.xlabel("Skill Level")
    plt.ylabel("Number of Skills")
    plt.title("Skill Level Distribution")
    plt.xticks(range(1, 6))
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def get_suggestions(resume_skills, github_lang_stats):
    skill_data = {
        "resume_skills": resume_skills,
        "github_stats": github_lang_stats
    }
    prompt = f"""
Based on the following data, suggest 3 career paths and 3 complementary skills to learn:
{json.dumps(skill_data, indent=2)}
"""
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You're a career advisor AI."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=600,
    )
    return response['choices'][0]['message']['content']


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if 'resume' not in request.files or request.files['resume'].filename == '':
            return render_template("index.html", error="Please upload a resume PDF.")
        resume_file = request.files['resume']
        github_username = request.form.get("github_username", "").strip()
        if not github_username:
            return render_template("index.html", error="Please enter your GitHub username.")

        filename = secure_filename(resume_file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        resume_file.save(filepath)

        resume_text = extract_text_from_pdf(filepath)
        resume_skills = ask_gpt_to_extract_skills(resume_text)
        github_lang_stats = fetch_github_stats(github_username)
        suggestions = get_suggestions(resume_skills, github_lang_stats)

        resume_skill_img = generate_resume_skill_plot(resume_skills)
        github_lang_img = generate_github_language_plot(github_lang_stats)
        repo_lang_img = generate_repo_language_count_plot(github_username)
        skill_dist_img = generate_skill_level_distribution_plot(resume_skills)

        return render_template(
            "index.html",
            suggestions=suggestions,
            github_lang_stats=github_lang_stats,
            resume_skills=resume_skills,
            github_username=github_username,
            resume_skill_img=resume_skill_img,
            github_lang_img=github_lang_img,
            repo_lang_img=repo_lang_img,
            skill_dist_img=skill_dist_img,
        )
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
