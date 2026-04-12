import os
import pickle
import re
import uuid

import numpy as np
import pytesseract
from flask import Flask, jsonify, redirect, render_template, request, session
from PIL import Image, ImageOps
from rapidocr_onnxruntime import RapidOCR


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)
app.secret_key = "secret123"

rapid_ocr = RapidOCR()
pytesseract.pytesseract.tesseract_cmd = os.getenv("TESSERACT_PATH", "tesseract")


with open(os.path.join(BASE_DIR, "model.pkl"), "rb") as model_file:
    model = pickle.load(model_file)

with open(os.path.join(BASE_DIR, "vectorizer.pkl"), "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)


UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


def rule_based_score(text):
    text_lower = text.lower()
    score = 0
    reasons = []

    keywords = [
        "payment",
        "registration fees",
        "money",
        "investment",
        "training fee",
        "bank account",
        "send your id",
        "visa",
        "lottery",
    ]

    for kw in keywords:
        if kw in text_lower:
            score += 15
            reasons.append(f"Contains suspicious keyword: '{kw}'")

    if re.search(r"(\$|rs|inr)?\s?[5-9]{2,}[0-9]{3,}", text_lower):
        score += 10
        reasons.append("Unrealistic salary mentioned")

    if not any(w in text_lower for w in ["company", "organization", "firm"]):
        score += 10
        reasons.append("No company details mentioned")

    if re.search(r"\b(gmail|yahoo|hotmail)\.com\b", text_lower):
        score += 10
        reasons.append("Uses free email provider")

    score = min(score, 100)
    return score, reasons


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
        "rule_score": rule_score,
    }


def extract_text_from_image(path):
    try:
        with Image.open(path) as img:
            img = ImageOps.exif_transpose(img)
            original = img.convert("RGB")

            max_side = 1800
            if max(original.size) > max_side:
                original.thumbnail((max_side, max_side))

            grayscale = original.convert("L")
            upscaled = grayscale.resize((grayscale.width * 2, grayscale.height * 2))
            thresholded = upscaled.point(lambda x: 0 if x < 170 else 255, "L")

        ocr_candidates = [
            np.asarray(original, dtype=np.uint8),
            np.asarray(grayscale, dtype=np.uint8),
            np.asarray(upscaled, dtype=np.uint8),
            np.asarray(thresholded, dtype=np.uint8),
        ]

        best_text = ""
        for candidate in ocr_candidates:
            rapid_result, _ = rapid_ocr(candidate)
            if not rapid_result:
                continue

            candidate_text = " ".join(
                line[1].strip()
                for line in rapid_result
                if len(line) > 1 and isinstance(line[1], str) and line[1].strip()
            ).strip()

            if len(candidate_text) > len(best_text):
                best_text = candidate_text

        if best_text:
            return best_text

        fallback_image = Image.fromarray(np.asarray(upscaled, dtype=np.uint8))
        text = pytesseract.image_to_string(fallback_image, config="--oem 3 --psm 6")
        return text.strip()
    except pytesseract.pytesseract.TesseractNotFoundError:
        print("OCR ERROR: Tesseract executable not found.")
        return ""
    except Exception as e:
        print("OCR ERROR:", e)
        return ""


@app.route("/")
def login():
    return render_template("login.html")


@app.route("/login", methods=["POST"])
def do_login():
    username = request.form["username"]
    password = request.form["password"]

    demo_users = {
        "Mhetre@00": "1715",
        "Mokal@00": "1715",
        "Mhatre@00": "1715",
        "Akash@00": "1715",
        "admin": "1234",
    }

    if demo_users.get(username) == password:
        session["user"] = username
        return redirect("/home")

    return render_template("login.html", error="Invalid credentials")


@app.route("/home")
def home():
    if "user" not in session:
        return redirect("/")
    return render_template("index.html", username=session["user"])


@app.route("/logout")
def logout():
    session.clear()
    return redirect("/")


@app.route("/predict", methods=["POST"])
def predict():
    if "user" not in session:
        return jsonify({"error": "Login required"}), 401

    text = ""
    extracted_text = ""
    path = None

    try:
        if "image" in request.files and request.files["image"].filename != "":
            image_file = request.files["image"]

            ext = os.path.splitext(image_file.filename)[1].lower() or ".png"
            if ext not in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}:
                ext = ".png"

            filename = f"{uuid.uuid4()}{ext}"
            path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            image_file.save(path)

            extracted_text = extract_text_from_image(path)
            print("Extracted Text:", extracted_text[:200])

            if not extracted_text:
                return jsonify(
                    {
                        "error": "Could not read text from image. Try a clearer image with larger, high-contrast text."
                    }
                ), 400

            text = extracted_text

        if not text:
            text = request.form.get("text", "").strip()

        if not text:
            return jsonify({"error": "Enter text or upload image"}), 400

        result = analyze_text(text)
        result["extracted_text"] = extracted_text[:500] if extracted_text else ""
        return jsonify(result)
    except Exception as e:
        print("PREDICT ERROR:", e)
        return jsonify({"error": f"Server error while analyzing image/text: {e}"}), 500
    finally:
        if path and os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass


@app.route("/about")
def about():
    return render_template("about.html")


if __name__ == "__main__":
    app.run(debug=True)
