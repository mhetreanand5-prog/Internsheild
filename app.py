import os
import pickle
import re
import uuid
from typing import List, Tuple

from flask import Flask, jsonify, redirect, render_template, request, session
from werkzeug.utils import secure_filename

from utils import (
    ai_generated_probability,
    analyze_domains_and_emails,
    build_link_verification,
    clean_extracted_text,
    extract_text_from_file,
    fetch_history,
    highlight_suspicious_terms,
    init_db,
    save_history,
)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".pdf", ".docx"}
MAX_FILE_SIZE = 8 * 1024 * 1024

app = Flask(__name__)
app.secret_key = "secret123"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


with open(os.path.join(BASE_DIR, "model.pkl"), "rb") as model_file:
    model = pickle.load(model_file)

with open(os.path.join(BASE_DIR, "vectorizer.pkl"), "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

init_db()


DEMO_USERS = {
    "Mhetre@00": "1715",
    "Mokal@00": "1715",
    "Mhatre@00": "1715",
    "Akash@00": "1715",
    "admin": "1234",
}


def allowed_file(filename: str) -> bool:
    return os.path.splitext(filename.lower())[1] in ALLOWED_EXTENSIONS


def rule_based_score(text: str) -> Tuple[int, List[str]]:
    text_lower = text.lower()
    score = 0
    reasons = []

    keyword_weights = {
        "payment": 15,
        "registration fee": 18,
        "registration fees": 18,
        "registration amount": 18,
        "security deposit": 20,
        "joining amount": 20,
        "processing fee": 18,
        "interview fee": 20,
        "slot booking": 16,
        "money": 12,
        "investment": 14,
        "training fee": 16,
        "bank account": 16,
        "send your id": 14,
        "visa": 10,
        "lottery": 14,
        "whatsapp": 8,
        "telegram": 10,
        "otp": 12,
        "urgent joining": 10,
        "guaranteed income": 18,
        "limited vacancy": 10,
        "earn daily": 16,
        "daily income": 14,
        "no interview": 14,
        "dm now": 10,
        "work from home": 6,
        "congratulations you have been selected": 18,
    }

    trust_signals = {
        "responsibilities": 4,
        "qualifications": 4,
        "job description": 4,
        "notice period": 3,
        "official careers page": 5,
        "equal opportunity employer": 5,
        "benefits": 3,
        "interview rounds": 3,
        "reporting manager": 3,
    }

    for keyword, weight in keyword_weights.items():
        if keyword in text_lower:
            score += weight
            reasons.append(f"Contains suspicious phrase: '{keyword}'")

    for phrase, reduction in trust_signals.items():
        if phrase in text_lower:
            score -= reduction

    if re.search(r"(\$|rs|inr)?\s?[5-9]{2,}[0-9]{3,}", text_lower):
        score += 10
        reasons.append("Mentions unusually high or unrealistic salary figures.")

    if not any(word in text_lower for word in ["company", "organization", "firm", "careers"]):
        score += 8
        reasons.append("Missing clear company details or official hiring context.")

    if "work from home" in text_lower and "experience" not in text_lower:
        score += 8
        reasons.append("Work-from-home offer lacks clear experience requirements.")

    if "immediate joining" in text_lower or "limited seats" in text_lower:
        score += 8
        reasons.append("Uses urgency tactics to pressure quick action.")

    if "no experience" in text_lower and re.search(r"earn|salary|income", text_lower):
        score += 10
        reasons.append("Promises earnings while requiring little or no experience.")

    if "contact on whatsapp" in text_lower or "message on whatsapp" in text_lower:
        score += 10
        reasons.append("Pushes conversation to WhatsApp instead of official hiring channels.")

    if "apply immediately" in text_lower or "join today" in text_lower:
        score += 8
        reasons.append("Encourages rushed decision-making without normal hiring steps.")

    return min(score, 100), reasons


def analyze_text(text: str) -> dict:
    cleaned_text = clean_extracted_text(text)
    X = vectorizer.transform([cleaned_text])
    ml_prob = model.predict_proba(X)[0][1] * 100

    rule_score, reasons = rule_based_score(cleaned_text)
    domain_analysis = analyze_domains_and_emails(cleaned_text)
    link_verification = build_link_verification(domain_analysis)
    ai_prob = ai_generated_probability(cleaned_text)

    combined_fake = (ml_prob * 0.5) + (rule_score * 0.25) + (domain_analysis["domain_risk_score"] * 0.25)
    combined_fake = min(combined_fake, 100)
    combined_real = 100 - combined_fake
    confidence = max(combined_fake, combined_real)

    enhanced_reasons = reasons + domain_analysis["domain_findings"]
    if ai_prob >= 60:
        enhanced_reasons.append("The text looks overly generic or formulaic, which can indicate AI-generated scam content.")

    if combined_fake < 35:
        risk = "LOW"
        conclusion = "This job looks relatively safe."
        insight = "No major scam patterns were detected across the model, rules, and domain checks."
        recommendation = [
            "Still verify the company website before applying.",
            "Apply only through trusted job portals or official career pages.",
            "Do not share sensitive personal documents too early.",
            "Cross-check the recruiter on LinkedIn or the company website.",
        ]
    elif combined_fake < 65:
        risk = "MEDIUM"
        conclusion = "This job looks suspicious."
        insight = "Some scam signals were detected, so verify carefully before taking action."
        recommendation = [
            "Verify company details before proceeding.",
            "Check whether the recruiter email and company domain match.",
            "Search online for reviews, complaints, or scam reports.",
            "Avoid paying any fee until legitimacy is confirmed.",
            "Ask for an official job description or company email confirmation.",
        ]
    else:
        risk = "HIGH"
        conclusion = "This job is likely fake."
        insight = "Multiple scam indicators were detected in the content and contact details."
        recommendation = [
            "Do not pay any money or registration fee.",
            "Do not share Aadhaar, PAN, bank, or OTP details.",
            "Block or report the recruiter if they pressure you urgently.",
            "Apply only through verified company websites or trusted portals.",
            "Report the listing to the job platform if possible.",
        ]

    return {
        "fake_probability": round(combined_fake, 2),
        "real_probability": round(combined_real, 2),
        "accuracy": round(confidence, 2),
        "risk_level": risk,
        "reasons": enhanced_reasons if enhanced_reasons else ["No major red flags"],
        "conclusion": conclusion,
        "system_insight": insight,
        "recommendation": recommendation,
        "ml_score": round(ml_prob, 2),
        "rule_score": rule_score,
        "domain_risk_score": domain_analysis["domain_risk_score"],
        "email_analysis": domain_analysis["email_analysis"],
        "urls": domain_analysis["urls"],
        "domains": domain_analysis["domains"],
        "link_verdict": link_verification["link_verdict"],
        "link_message": link_verification["link_message"],
        "link_checked": link_verification["link_checked"],
        "ai_generated_probability": ai_prob,
        "highlighted_text": highlight_suspicious_terms(cleaned_text),
        "cleaned_text": cleaned_text,
    }


def save_uploaded_files(files) -> Tuple[List[str], List[str], List[str]]:
    saved_paths = []
    labels = []
    extracted_chunks = []

    for file_storage in files:
        if not file_storage or not file_storage.filename:
            continue

        original_name = secure_filename(file_storage.filename)
        extension = os.path.splitext(original_name.lower())[1]
        if extension not in ALLOWED_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {original_name}")

        file_storage.stream.seek(0, os.SEEK_END)
        size = file_storage.stream.tell()
        file_storage.stream.seek(0)
        if size > MAX_FILE_SIZE:
            raise ValueError(f"File too large: {original_name}. Max allowed size is 8 MB.")

        unique_name = f"{uuid.uuid4()}_{original_name}"
        destination = os.path.join(app.config["UPLOAD_FOLDER"], unique_name)
        file_storage.save(destination)
        saved_paths.append(destination)
        labels.append(original_name)

        extracted = extract_text_from_file(destination, extension)
        if extracted:
            extracted_chunks.append(extracted)

    return saved_paths, labels, extracted_chunks


@app.errorhandler(413)
def file_too_large(_error):
    return jsonify({"error": "Uploaded content is too large. Keep files under 8 MB each."}), 413


@app.route("/")
def login():
    return render_template("login.html")


@app.route("/login", methods=["POST"])
def do_login():
    username = request.form["username"]
    password = request.form["password"]

    if DEMO_USERS.get(username) == password:
        session["user"] = username
        return redirect("/home")

    return render_template("login.html", error="Invalid credentials")


@app.route("/home")
def home():
    if "user" not in session:
        return redirect("/")
    return render_template("index.html", username=session["user"])


@app.route("/history")
def history():
    if "user" not in session:
        return redirect("/")
    return render_template("history.html", username=session["user"], records=fetch_history())


@app.route("/logout")
def logout():
    session.clear()
    return redirect("/")


@app.route("/predict", methods=["POST"])
def predict():
    if "user" not in session:
        return jsonify({"error": "Login required"}), 401

    saved_paths = []
    file_labels = []

    try:
        text = clean_extracted_text(request.form.get("text", ""))
        files = request.files.getlist("files")

        if files:
            saved_paths, file_labels, extracted_chunks = save_uploaded_files(files)
            combined_file_text = clean_extracted_text(" ".join(extracted_chunks))
            if combined_file_text:
                text = clean_extracted_text(f"{text} {combined_file_text}") if text else combined_file_text

        if not text:
            return jsonify({"error": "Enter text or upload a supported file (JPG, PNG, PDF, DOCX)."}), 400

        result = analyze_text(text)
        result["extracted_text"] = text[:500]
        result["input_label"] = ", ".join(file_labels) if file_labels else "Manual text input"

        save_history(result["input_label"], result["fake_probability"], result["risk_level"])
        return jsonify(result)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        print("PREDICT ERROR:", exc)
        return jsonify({"error": f"Server error while analyzing content: {exc}"}), 500
    finally:
        for saved_path in saved_paths:
            if os.path.exists(saved_path):
                try:
                    os.remove(saved_path)
                except OSError:
                    pass


@app.route("/about")
def about():
    return render_template("about.html")


if __name__ == "__main__":
    app.run(debug=True)
