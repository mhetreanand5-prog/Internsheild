import html
import os
import re
import sqlite3
from datetime import datetime
from urllib.parse import urlparse

import fitz
import numpy as np
import pytesseract
from docx import Document
from PIL import Image, ImageOps


DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "history.db")
FREE_EMAIL_PROVIDERS = {
    "gmail.com",
    "yahoo.com",
    "hotmail.com",
    "outlook.com",
    "live.com",
    "protonmail.com",
    "icloud.com",
}
RISK_KEYWORDS = [
    "payment",
    "registration fee",
    "registration fees",
    "registration amount",
    "security deposit",
    "joining amount",
    "processing fee",
    "interview fee",
    "slot booking",
    "money",
    "investment",
    "training fee",
    "bank account",
    "send your id",
    "urgent joining",
    "limited vacancy",
    "earn daily",
    "daily income",
    "telegram",
    "lottery",
    "whatsapp",
    "otp",
    "visa",
    "guaranteed income",
    "no interview",
    "dm now",
    "work from home",
    "congratulations you have been selected",
    "without interview",
    "weekly payments",
    "ppo guaranteed",
    "limited slots",
    "offer valid",
    "refund after first week salary",
]
GENERIC_AI_PHRASES = [
    "dynamic work environment",
    "kickstart your career",
    "excellent communication skills",
    "fast-paced environment",
    "esteemed organization",
    "growth opportunities",
    "highly motivated candidate",
    "competitive salary package",
]
UNCOMMON_TLDS = {
    "xyz",
    "top",
    "click",
    "buzz",
    "monster",
    "work",
    "zip",
    "review",
}

_rapid_ocr = None


def get_rapid_ocr():
    global _rapid_ocr
    if _rapid_ocr is None:
        try:
            from rapidocr_onnxruntime import RapidOCR
            _rapid_ocr = RapidOCR()
        except Exception:
            _rapid_ocr = False
    return _rapid_ocr


def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS prediction_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            input_label TEXT NOT NULL,
            fake_probability REAL NOT NULL,
            risk_level TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()


def save_history(input_label, fake_probability, risk_level):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        INSERT INTO prediction_history (input_label, fake_probability, risk_level, created_at)
        VALUES (?, ?, ?, ?)
        """,
        (
            input_label[:255],
            float(fake_probability),
            risk_level,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        ),
    )
    conn.commit()
    conn.close()


def fetch_history(limit=50):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        SELECT input_label, fake_probability, risk_level, created_at
        FROM prediction_history
        ORDER BY id DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    conn.close()
    return rows


def clean_extracted_text(text):
    text = text or ""
    text = text.replace("\x00", " ")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"([!?.,]){2,}", r"\1", text)
    return text.strip()


def is_readable_text(text):
    text = (text or "").strip()
    if len(text) < 40:
        return False

    alpha_chars = sum(char.isalpha() for char in text)
    digit_chars = sum(char.isdigit() for char in text)
    weird_chars = sum(not (char.isalnum() or char.isspace() or char in ".,!?@:/%$&()-'\"") for char in text)
    total_chars = max(len(text), 1)

    alpha_ratio = alpha_chars / total_chars
    weird_ratio = weird_chars / total_chars
    digit_ratio = digit_chars / total_chars

    words = re.findall(r"[A-Za-z]{2,}", text)
    if len(words) < 8:
        return False

    long_consonant_chunks = re.findall(r"[bcdfghjklmnpqrstvwxyz]{6,}", text.lower())
    if long_consonant_chunks:
        return False

    return alpha_ratio >= 0.45 and weird_ratio <= 0.08 and digit_ratio <= 0.35


def extract_text_from_image(path):
    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img)
        img = img.convert("L")
        if max(img.size) > 1400:
            img.thumbnail((1400, 1400))
        img = img.resize((max(img.width, 1) * 2, max(img.height, 1) * 2))
        image_array = np.asarray(img, dtype=np.uint8)

    try:
        text = pytesseract.image_to_string(Image.fromarray(image_array), config="--oem 3 --psm 6")
        text = clean_extracted_text(text)
        if text:
            return text
    except pytesseract.pytesseract.TesseractNotFoundError:
        pass

    rapid_ocr = get_rapid_ocr()
    if not rapid_ocr:
        return ""

    rapid_result, _ = rapid_ocr(image_array)
    if not rapid_result:
        return ""

    text = " ".join(
        line[1].strip()
        for line in rapid_result
        if len(line) > 1 and isinstance(line[1], str) and line[1].strip()
    )
    return clean_extracted_text(text)


def extract_text_from_pdf(path):
    chunks = []
    with fitz.open(path) as pdf:
        for page in pdf:
            chunks.append(page.get_text("text"))
    return clean_extracted_text(" ".join(chunks))


def extract_text_from_docx(path):
    doc = Document(path)
    chunks = [paragraph.text for paragraph in doc.paragraphs if paragraph.text.strip()]
    return clean_extracted_text(" ".join(chunks))


def extract_text_from_file(path, extension):
    if extension in {".png", ".jpg", ".jpeg"}:
        return extract_text_from_image(path)
    if extension == ".pdf":
        return extract_text_from_pdf(path)
    if extension == ".docx":
        return extract_text_from_docx(path)
    raise ValueError("Unsupported file type")


def detect_urls(text):
    pattern = r"(https?://[^\s]+|www\.[^\s]+|\b[a-zA-Z0-9-]+\.(?:com|in|org|net|co|io|xyz|top|site|live|info|biz|work|click|buzz)(?:/[^\s]*)?)"
    return re.findall(pattern, text, flags=re.IGNORECASE)


def detect_emails(text):
    return re.findall(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", text)


def normalize_domain(value):
    candidate = value.strip().lower()
    if "@" in candidate:
        candidate = candidate.split("@", 1)[1]
    if not candidate.startswith(("http://", "https://")):
        candidate = "https://" + candidate
    parsed = urlparse(candidate)
    return parsed.netloc.lower().replace("www.", "")


def infer_company_tokens(text):
    company_patterns = [
        r"company[:\-]\s*([A-Za-z0-9 &.-]+)",
        r"organization[:\-]\s*([A-Za-z0-9 &.-]+)",
        r"firm[:\-]\s*([A-Za-z0-9 &.-]+)",
    ]
    tokens = set()
    for pattern in company_patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            pieces = re.findall(r"[A-Za-z]{3,}", match.group(1).lower())
            tokens.update(pieces)
    return tokens


def analyze_domains_and_emails(text):
    urls = detect_urls(text)
    emails = detect_emails(text)
    company_tokens = infer_company_tokens(text)

    domain_score = 0
    domain_findings = []
    email_findings = []
    extracted_domains = []

    for url in urls:
        domain = normalize_domain(url)
        extracted_domains.append(domain)
        base = domain.split(".")[0]
        tld = domain.split(".")[-1] if "." in domain else ""

        if tld in UNCOMMON_TLDS:
            domain_score += 12
            domain_findings.append(f"Domain uses an uncommon TLD: {domain}")
        if sum(ch.isdigit() for ch in base) >= 3 or len(re.findall(r"[bcdfghjklmnpqrstvwxyz]{5,}", base)) > 0:
            domain_score += 10
            domain_findings.append(f"Domain looks random or autogenerated: {domain}")
        if base.count("-") >= 2:
            domain_score += 6
            domain_findings.append(f"Domain contains excessive hyphens: {domain}")

    for email in emails:
        email_domain = normalize_domain(email)
        extracted_domains.append(email_domain)

        if email_domain in FREE_EMAIL_PROVIDERS:
            domain_score += 8
            email_findings.append(f"{email} uses a free email provider.")

        if company_tokens:
            base = email_domain.split(".")[0]
            if not any(token in base for token in company_tokens):
                domain_score += 6
                email_findings.append(
                    f"{email} does not clearly match the stated company name."
                )

    if not urls and not emails:
        domain_findings.append("No URLs or recruiter email addresses found.")

    return {
        "urls": urls,
        "domains": sorted(set(extracted_domains)),
        "domain_risk_score": min(domain_score, 100),
        "domain_findings": domain_findings,
        "email_analysis": email_findings or ["No major email mismatches detected."],
    }


def build_link_verification(domain_analysis):
    urls = domain_analysis.get("urls", [])
    domains = domain_analysis.get("domains", [])
    score = domain_analysis.get("domain_risk_score", 0)

    if not urls and not domains:
        return {
            "link_verdict": "No link detected",
            "link_message": "Paste a website URL to verify whether it looks suspicious or relatively safe.",
            "link_checked": False,
        }

    primary_target = urls[0] if urls else domains[0]

    if score < 25:
        verdict = "Safe-looking link"
        message = f"{primary_target} does not show strong scam-domain signals, but you should still verify the site manually."
    elif score < 55:
        verdict = "Suspicious link"
        message = f"{primary_target} shows some risky patterns. Check the domain carefully before trusting it."
    else:
        verdict = "Likely fake link"
        message = f"{primary_target} shows multiple risky domain patterns and should not be trusted without strong verification."

    return {
        "link_verdict": verdict,
        "link_message": message,
        "link_checked": True,
    }


def ai_generated_probability(text):
    text_lower = text.lower()
    score = 8

    phrase_hits = sum(1 for phrase in GENERIC_AI_PHRASES if phrase in text_lower)
    score += phrase_hits * 10

    sentences = [segment.strip() for segment in re.split(r"[.!?]", text_lower) if segment.strip()]
    if sentences:
        unique_ratio = len(set(sentences)) / len(sentences)
        if unique_ratio < 0.7:
            score += 20

    repeated_words = re.findall(r"\b(\w+)\b(?:\s+\1\b){2,}", text_lower)
    if repeated_words:
        score += 15

    average_sentence_length = (
        sum(len(sentence.split()) for sentence in sentences) / len(sentences)
        if sentences
        else 0
    )
    if average_sentence_length > 22:
        score += 12

    if text_lower.count("we are excited") or text_lower.count("ideal candidate"):
        score += 10

    return min(round(score, 2), 100)


def highlight_suspicious_terms(text):
    if not is_readable_text(text):
        return ""

    escaped_text = html.escape(text)
    for keyword in sorted(RISK_KEYWORDS, key=len, reverse=True):
        pattern = re.compile(re.escape(html.escape(keyword)), flags=re.IGNORECASE)
        escaped_text = pattern.sub(
            lambda match: f"<mark class=\"risk-highlight\">{match.group(0)}</mark>",
            escaped_text,
        )
    return escaped_text.replace("\n", "<br>")
