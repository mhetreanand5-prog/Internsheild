#!/usr/bin/env bash
set -e

apt-get update
apt-get install -y tesseract-ocr
pip install --upgrade pip
pip install -r requirements.txt
