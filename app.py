import os
import pickle
import re
import uuid

from flask import Flask, render_template, request, redirect, jsonify, session
from PIL import Image
import pytesseract

# ---------------- INIT ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)
app.secret_key = 'secret123'

# ---------------- TESSERACT PATH ----------------
# Use an env override when provided, otherwise rely on the executable
# available in the current OS PATH (works for Render/Linux deployments).
pytesseract.pytesseract.tesseract_cmd = os.getenv("TESSERACT_PATH", "tesseract")

# ---------------- LOAD MODEL ----------------
with open(os.path.join(BASE_DIR, 'model.pkl'), 'rb') as model_file:
    model = pickle.load(model_file)

with open(os.path.join(BASE_DIR, 'vectorizer.pkl'), 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# ---------------- UPLOAD FOLDER ----------------
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# ---------------- RULE BASED ----------------
def rule_based_score(text):
    text_lower = text.lower()
    score = 0
    reasons = []

    keywords = ['payment', 'registration fees', 'money', 'investment',
                'training fee', 'bank account', 'send your id', 'visa', 'lottery']

    for kw in keywords:
        if kw in text_lower:
            score += 15
            reasons.append(f"Contains suspicious keyword: '{kw}'")

    if re.search(r'(\$|rs|inr)?\s?[5-9]{2,}[0-9]{3,}', text_lower):
        score += 10
        reasons.append("Unrealistic salary mentioned")

    if not any(w in text_lower for w in ['company', 'organization', 'firm']):
        score += 10
        reasons.append("No company details mentioned")

    if re.search(r'\b(gmail|yahoo|hotmail)\.com\b', text_lower):
        score += 10
        reasons.append("Uses free email provider")

    score = min(score, 100)
    return score, reasons


# ---------------- ANALYSIS ----------------
def analyze_text(text):
    X = vectorizer.transform([text])
    ml_prob = model.predict_proba(X)[0][1] * 100

    rule_score, reasons = rule_based_score(text)

    final_fake = (ml_prob * 0.6) + (rule_score * 0.4)
    final_real = 100 - final_fake

    if final_fake < 35:
        risk = "LOW"
        conclusion = "This job looks safe."
        insight = "No major scam patterns detected."
        recommendation = ["Still verify company before applying."]
    elif final_fake < 65:
        risk = "MEDIUM"
        conclusion = "This job looks suspicious."
        insight = "Some scam signals detected."
        recommendation = ["Verify company details before proceeding."]
    else:
        risk = "HIGH"
        conclusion = "This job is likely FAKE."
        insight = "Multiple scam indicators detected."
        recommendation = ["DO NOT pay money.", "Avoid sharing personal info."]

    return {
        "fake_probability": round(final_fake, 2),
        "real_probability": round(final_real, 2),
        "risk_level": risk,
        "reasons": reasons if reasons else ["No major red flags"],
        "conclusion": conclusion,
        "system_insight": insight,
        "recommendation": recommendation,
        "ml_score": round(ml_prob, 2),
        "rule_score": rule_score
    }


# ---------------- OCR ----------------
def extract_text_from_image(path):
    try:
        img = Image.open(path)

        # 🔥 Improve OCR accuracy
        img = img.convert('L')  # grayscale
        img = img.resize((img.width * 2, img.height * 2))  # upscale
        img = img.point(lambda x: 0 if x < 150 else 255, '1')  # threshold

        custom_config = r'--oem 3 --psm 6'

        text = pytesseract.image_to_string(img, config=custom_config)

        return text.strip()

    except pytesseract.pytesseract.TesseractNotFoundError:
        print("OCR ERROR: Tesseract executable not found.")
        return ""
    except Exception as e:
        print("OCR ERROR:", e)
        return ""


# ---------------- ROUTES ----------------
@app.route('/')
def login():
    return render_template('login.html')


@app.route('/login', methods=['POST'])
def do_login():
    u = request.form['username']
    p = request.form['password']

    if u == 'admin' and p == '1234':
        session['user'] = u
        return redirect('/home')

    return render_template('login.html', error="Invalid credentials")


@app.route('/home')
def home():
    if 'user' not in session:
        return redirect('/')
    return render_template('index.html', username=session['user'])


@app.route('/logout')
def logout():
    session.clear()
    return redirect('/')


# ---------------- MAIN PREDICT ----------------
@app.route('/predict', methods=['POST'])
def predict():
    if 'user' not in session:
        return jsonify({'error': 'Login required'}), 401

    text = ""
    extracted_text = ""

    # IMAGE CASE
    if 'image' in request.files and request.files['image'].filename != '':
        img = request.files['image']

        # ✅ unique filename fix
        filename = str(uuid.uuid4()) + ".png"
        path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        img.save(path)

        extracted_text = extract_text_from_image(path)

        print("Extracted Text:", extracted_text[:200])  # debug

        if not extracted_text:
            return jsonify({
                "error": "Could not read text from image. Ensure Tesseract is installed on the server and try a clearer image."
            }), 400

        text = extracted_text

    # TEXT CASE
    if not text:
        text = request.form.get('text', '').strip()

    if not text:
        return jsonify({'error': 'Enter text or upload image'}), 400

    result = analyze_text(text)

    result["extracted_text"] = extracted_text[:500] if extracted_text else ""

    return jsonify(result)


@app.route('/about')
def about():
    return render_template('about.html')


# ---------------- RUN ----------------
if __name__ == '__main__':
    app.run(debug=True)
