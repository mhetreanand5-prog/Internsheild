from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import pickle
import re
import os

app = Flask(__name__)
app.secret_key = "fakejob_secret_2024"

# ---------------- LOAD MODEL ----------------
try:
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    print("[INFO] Model loaded")
except Exception as e:
    print("[WARNING] Model not loaded:", e)
    model = None
    vectorizer = None

# ---------------- USERS ----------------
USERS = {
    "admin": "admin123",
    "student": "pass123",
    "demo": "demo",
    "Mhetre@00": "1715",
    "Mokal@00": "1715",
    "Mhatre@00": "1715",
    "Akash@00": "1715"
}

# ---------------- RULE BASED ----------------
HIGH_RISK_KEYWORDS = ["registration fee","pay fee","send money","no experience needed"]
MEDIUM_RISK_KEYWORDS = ["work from home","part time","urgent hiring"]
REAL_JOB_SIGNALS = ["responsibilities","requirements","job description"]

def rule_based_analysis(text):
    text_lower = text.lower()
    score = 0
    reasons = []

    for k in HIGH_RISK_KEYWORDS:
        if k in text_lower:
            score += 20
            reasons.append(k)

    for k in MEDIUM_RISK_KEYWORDS:
        if k in text_lower:
            score += 8
            reasons.append(k)

    for k in REAL_JOB_SIGNALS:
        if k in text_lower:
            score -= 10

    return score, reasons

# ---------------- ML ----------------
def get_ml_prediction(text):
    if model is None or vectorizer is None:
        return 0.5
    try:
        vec = vectorizer.transform([text])
        proba = model.predict_proba(vec)[0]
        return float(proba[1])
    except:
        return 0.5

def combine_scores(ml, rule):
    rule_norm = max(0, min(rule, 200)) / 200
    return (0.6 * ml) + (0.4 * rule_norm)

def get_risk_level(p):
    if p >= 0.7:
        return "HIGH"
    elif p >= 0.4:
        return "MEDIUM"
    return "LOW"

# ---------------- ANALYSIS ----------------
def analyze_job(text):
    ml = get_ml_prediction(text)
    rule, reasons = rule_based_analysis(text)

    final = combine_scores(ml, rule)
    risk = get_risk_level(final)

    return {
        "fake_probability": round(final * 100,1),
        "real_probability": round((1-final)*100,1),
        "risk_level": risk,
        "reasons": reasons or ["No major red flags"]
    }

# ---------------- ROUTES ----------------
@app.route("/", methods=["GET","POST"])
def login():
    error = None

    if request.method == "POST":
        u = request.form.get("username", "").strip()
        p = request.form.get("password", "").strip()

        print("DEBUG LOGIN:", u, p)

        if u in USERS and USERS[u] == p:
            session["user"] = u
            return redirect(url_for("home"))
        else:
            error = "Invalid username or password"

    return render_template("login.html", error=error)

# ✅ FIX IS HERE (ALREADY INCLUDED)
@app.route("/home")
def home():
    if "user" not in session:
        return redirect(url_for("login"))
    
    return render_template("index.html", username=session.get("user"))

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "user" not in session:
        return jsonify({"error":"login required"}),401

    text = request.form.get("text","").strip()

    if not text:
        return jsonify({"error":"enter text"}),400

    report = analyze_job(text)
    return jsonify(report)

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

# ---------------- RUN ----------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)