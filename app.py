

# ============================================================
# app.py - Main Flask Application
# Fake Job & Internship Detector
# ============================================================

from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import pickle
import pytesseract
from PIL import Image
import io
import re
import os

# ── App Setup ──────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = "fakejob_secret_2024"  # Used to encrypt session data

# ── Tesseract Path (Windows users: update this path) ───────
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
# On Linux/Mac this usually works automatically if installed via apt/brew

# ── Load ML Model & Vectorizer ─────────────────────────────
try:
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    print("[INFO] Model and vectorizer loaded successfully.")
except Exception as e:
    print(f"[WARNING] Could not load model: {e}")
    model = None
    vectorizer = None

# ── Hardcoded Users (simple login system) ──────────────────
USERS = {
    "admin": "admin123",
    "student": "pass123",
    "demo": "demo"
}

# ── Rule-Based Scam Keywords & Patterns ────────────────────

# High-risk keywords that are very commonly found in fake jobs
HIGH_RISK_KEYWORDS = [
    "registration fee", "pay fee", "deposit required", "upfront payment",
    "send money", "wire transfer", "western union", "money gram",
    "work from home earn", "guaranteed income", "no experience needed",
    "earn per day", "earn daily", "unlimited earning", "be your own boss",
    "click ads", "ad clicking", "data entry earn", "part time earn lakhs",
    "weekly payment", "direct payment", "whatsapp to join",
    "call immediately", "limited seats", "urgent hiring 500 people",
    "no interview", "selected directly", "100% job guarantee",
    "government job guaranteed", "railway job", "lic job offer",
]

# Medium-risk keywords
MEDIUM_RISK_KEYWORDS = [
    "work from home", "part time", "no qualification required",
    "freshers welcome", "immediate joining", "apply now urgent",
    "high salary", "attractive package", "contact hr on whatsapp",
    "limited vacancies", "don't miss", "golden opportunity",
    "international company", "multinational", "foreign company hiring",
    "apply fast", "last date today",
]

# Patterns that suggest REAL jobs
REAL_JOB_SIGNALS = [
    "responsibilities", "requirements", "qualifications",
    "about the company", "job description", "we are looking for",
    "apply through", "official website", "careers page",
    "background check", "interview process", "onboarding",
    "employee benefits", "health insurance", "annual leave",
    "performance review", "team collaboration",
]


def rule_based_analysis(text):
    """
    Analyze job text using rule-based keyword matching.
    Returns a score and list of reasons.

    Score meaning:
      - Positive score = more likely FAKE
      - Negative score = more likely REAL
    """
    text_lower = text.lower()
    reasons = []
    score = 0  # starts neutral

    # ── Check High Risk Keywords ───────────────────────────
    for keyword in HIGH_RISK_KEYWORDS:
        if keyword in text_lower:
            score += 20  # very suspicious
            reasons.append(f"Contains high-risk phrase: '{keyword}'")

    # ── Check Medium Risk Keywords ─────────────────────────
    for keyword in MEDIUM_RISK_KEYWORDS:
        if keyword in text_lower:
            score += 8
            reasons.append(f"Contains suspicious phrase: '{keyword}'")

    # ── Check Real Job Signals ─────────────────────────────
    real_count = 0
    for signal in REAL_JOB_SIGNALS:
        if signal in text_lower:
            score -= 10  # reduces fake score
            real_count += 1
    if real_count >= 3:
        reasons.append(f"Contains {real_count} professional job description signals (looks structured)")

    # ── Salary Check: unrealistic numbers ─────────────────
    # Look for patterns like "50000 per day", "1 lakh per week" etc.
    salary_per_day = re.findall(r'(\d[\d,]+)\s*(per day|daily|a day)', text_lower)
    for match in salary_per_day:
        amount_str = match[0].replace(",", "")
        try:
            amount = int(amount_str)
            if amount > 2000:  # More than 2000/day is suspicious for most entry roles
                score += 25
                reasons.append(f"Unrealistic daily salary mentioned: {match[0]} per day")
        except:
            pass

    # ── Check for missing company information ──────────────
    has_company = any(word in text_lower for word in [
        "pvt ltd", "private limited", "inc.", "corporation",
        "technologies", "solutions", "services", "ltd"
    ])
    if not has_company:
        score += 10
        reasons.append("No proper company name or registration mentioned")

    # ── Check for urgency tactics ──────────────────────────
    urgency_words = ["urgent", "immediately", "today only", "last chance", "hurry"]
    urgency_found = [w for w in urgency_words if w in text_lower]
    if len(urgency_found) >= 2:
        score += 15
        reasons.append(f"Uses multiple urgency pressure tactics: {urgency_found}")

    # ── Check text length: very short descriptions are suspicious
    word_count = len(text.split())
    if word_count < 30:
        score += 10
        reasons.append("Very short job description (less than 30 words) — lacks professional detail")

    # ── Contact method check ──────────────────────────────
    if "whatsapp" in text_lower and "apply" in text_lower:
        score += 15
        reasons.append("Asks candidates to apply via WhatsApp (unprofessional recruitment method)")

    if "@gmail.com" in text_lower or "@yahoo.com" in text_lower:
        score += 12
        reasons.append("Uses personal email (Gmail/Yahoo) instead of official company email")

    return score, reasons


def get_ml_prediction(text):
    """
    Use the trained ML model to predict fake vs real.
    Returns fake_probability (0 to 1).
    """
    if model is None or vectorizer is None:
        # If model not loaded, return neutral 0.5
        return 0.5

    try:
        text_vec = vectorizer.transform([text])
        # Get probability: [prob_real, prob_fake] — depends on how you trained it
        proba = model.predict_proba(text_vec)[0]

        # Assuming label 0 = real, label 1 = fake
        # Adjust index if your model is different
        fake_prob = proba[1] if len(proba) > 1 else proba[0]
        return float(fake_prob)
    except Exception as e:
        print(f"[ERROR] ML prediction failed: {e}")
        return 0.5


def combine_scores(ml_fake_prob, rule_score):
    """
    Combine ML score and rule-based score into final fake probability.

    ml_fake_prob: float 0 to 1 (from ML model)
    rule_score: int (higher = more fake, can be negative = real)

    We normalize rule_score to 0-1 range, then average both.
    """
    # Normalize rule score: cap between -50 and 200, then scale to 0-1
    rule_normalized = max(0, min(rule_score, 200)) / 200.0

    # Weighted combination: 60% ML, 40% rule-based
    # You can adjust these weights
    combined = (0.60 * ml_fake_prob) + (0.40 * rule_normalized)

    # Clamp between 0 and 1
    combined = max(0.0, min(1.0, combined))
    return combined


def get_risk_level(fake_prob):
    """Convert fake probability to risk level label."""
    if fake_prob >= 0.70:
        return "HIGH"
    elif fake_prob >= 0.40:
        return "MEDIUM"
    else:
        return "LOW"


def generate_conclusion(fake_prob, risk_level):
    """Generate a human-readable conclusion sentence."""
    if risk_level == "HIGH":
        return "This job posting shows strong indicators of being FRAUDULENT. Avoid applying and do not share personal or financial information."
    elif risk_level == "MEDIUM":
        return "This job posting has several suspicious elements. Proceed with caution and verify the company before applying."
    else:
        return "This job posting appears to be LEGITIMATE based on our analysis. Standard verification is still recommended."


def generate_system_insight(fake_prob, rule_score, reasons):
    """Generate a summary insight about what patterns were detected."""
    if fake_prob >= 0.70:
        return (
            f"Our system detected {len(reasons)} suspicious pattern(s) in this posting. "
            "The language and structure closely match known job scam patterns in our training data. "
            "Financial requests, unrealistic promises, or unprofessional contact methods were found."
        )
    elif fake_prob >= 0.40:
        return (
            f"Our system detected {len(reasons)} pattern(s) that partially match scam profiles. "
            "The posting has some legitimate elements but also contains red flags. "
            "Manual verification is strongly advised."
        )
    else:
        return (
            "The posting contains professional language, structured job requirements, and legitimate company signals. "
            "Our ML model and rule engine both suggest this is likely a genuine opportunity."
        )


def generate_recommendation(risk_level):
    """Give actionable advice based on risk level."""
    if risk_level == "HIGH":
        return [
            "Do NOT pay any registration or training fee",
            "Do NOT share your Aadhaar, PAN, or bank details",
            "Report this job posting to cybercrime.gov.in",
            "Block the sender if contacted via WhatsApp or SMS",
            "Warn your friends and family about this posting"
        ]
    elif risk_level == "MEDIUM":
        return [
            "Verify the company on MCA (Ministry of Corporate Affairs) website",
            "Check company reviews on LinkedIn or Glassdoor",
            "Never pay any upfront fees for any job",
            "Use official company website to apply, not WhatsApp/SMS",
            "Research the recruiter on LinkedIn before responding"
        ]
    else:
        return [
            "Verify the job on the company's official careers page",
            "Research the company on LinkedIn and Glassdoor",
            "Prepare properly for the interview process",
            "Never share sensitive documents before an official offer letter",
            "Trust your instincts — if something feels wrong, it probably is"
        ]


def analyze_job(text):
    """
    Main analysis function that combines everything.
    Returns a complete analysis report as a dictionary.
    """
    # Step 1: ML Model Prediction
    ml_fake_prob = get_ml_prediction(text)

    # Step 2: Rule-Based Analysis
    rule_score, reasons = rule_based_analysis(text)

    # Step 3: Combine Scores
    final_fake_prob = combine_scores(ml_fake_prob, rule_score)
    final_real_prob = 1.0 - final_fake_prob

    # Step 4: Risk Level
    risk_level = get_risk_level(final_fake_prob)

    # Step 5: Build Full Report
    report = {
        "fake_probability": round(final_fake_prob * 100, 1),
        "real_probability": round(final_real_prob * 100, 1),
        "risk_level": risk_level,
        "conclusion": generate_conclusion(final_fake_prob, risk_level),
        "reasons": reasons if reasons else ["No specific red flags detected in the text"],
        "system_insight": generate_system_insight(final_fake_prob, rule_score, reasons),
        "recommendation": generate_recommendation(risk_level),
        "ml_score": round(ml_fake_prob * 100, 1),
        "rule_score": rule_score
    }

    return report


# ── Routes ─────────────────────────────────────────────────

@app.route("/", methods=["GET", "POST"])
def login():
    """Login page route."""
    error = None

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()

        # Check credentials against hardcoded users
        if username in USERS and USERS[username] == password:
            session["user"] = username  # Save user in session
            return redirect(url_for("home"))
        else:
            error = "Invalid username or password. Try: admin / admin123"

    return render_template("login.html", error=error)


@app.route("/logout")
def logout():
    """Clear session and go back to login."""
    session.clear()
    return redirect(url_for("login"))


@app.route("/home")
def home():
    """Main chat interface - requires login."""
    if "user" not in session:
        return redirect(url_for("login"))
    return render_template("index.html", username=session["user"])


@app.route("/about")
def about():
    """About page."""
    return render_template("about.html")


@app.route("/predict", methods=["POST"])
def predict():
    """
    API endpoint that processes job text or image.
    Returns JSON with the full analysis report.
    """
    # Must be logged in
    if "user" not in session:
        return jsonify({"error": "Unauthorized"}), 401

    text = ""

    # ── Check if image was uploaded ────────────────────────
    if "image" in request.files and request.files["image"].filename != "":
        image_file = request.files["image"]
        try:
            # Open image using Pillow
            img = Image.open(io.BytesIO(image_file.read()))
            # Extract text using Tesseract OCR
            text = pytesseract.image_to_string(img)
            print(f"[OCR] Extracted text ({len(text)} chars)")
        except Exception as e:
            return jsonify({"error": f"OCR failed: {str(e)}"}), 500

    # ── Otherwise use typed text ───────────────────────────
    elif "text" in request.form and request.form["text"].strip():
        text = request.form["text"].strip()

    else:
        return jsonify({"error": "Please provide job text or upload an image."}), 400

    # ── Minimum text check ─────────────────────────────────
    if len(text.split()) < 5:
        return jsonify({"error": "Text too short. Please provide more details."}), 400

    # ── Run Analysis ───────────────────────────────────────
    report = analyze_job(text)
    report["extracted_text"] = text[:500] + "..." if len(text) > 500 else text

    return jsonify(report)


# ── Run App ────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, port=5000)
