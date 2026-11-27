"""LensGuard 분석 서버.

브라우저가 영상에서 뽑은 프레임(JPEG, base64)을 받아 Claude 비전 API로 분석하고
스마트(AI) 안경 / 초소형 카메라 의심 위치를 JSON으로 돌려준다.

- 프레임은 메모리에서만 처리하고 디스크에 저장하지 않는다.
- 사람의 신원을 식별하지 않는다. 장비(안경, 렌즈) 유무와 위치만 본다.
- ANTHROPIC_API_KEY 가 없으면 MOCK 모드로 동작한다 (UI 테스트용 가짜 결과).

실행:
    export ANTHROPIC_API_KEY=sk-ant-...
    python3 app.py
"""
import base64
import json
import os
import random
import re
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from flask import Flask, jsonify, request, send_from_directory

HERE = os.path.dirname(os.path.abspath(__file__))
API_URL = "https://api.anthropic.com/v1/messages"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
# 기본 모델. 필요하면 Render 환경변수로 바꾼다.
CLAUDE_MODEL = os.environ.get("LENSGUARD_MODEL", "claude-sonnet-5-5")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")


def get_provider() -> str:
    """어떤 AI를 쓸지 결정한다. 'anthropic' | 'gemini' | 'mock'.

    LENSGUARD_PROVIDER 로 강제할 수 있고, 없으면 설정된 키를 보고 고른다
    (둘 다 있으면 Anthropic 우선).
    """
    forced = os.environ.get("LENSGUARD_PROVIDER", "").strip().lower()
    has_a = bool(os.environ.get("ANTHROPIC_API_KEY", "").strip())
    has_g = bool(os.environ.get("GEMINI_API_KEY", "").strip())
    if forced == "gemini" and has_g:
        return "gemini"
    if forced == "anthropic" and has_a:
        return "anthropic"
    if has_a:
        return "anthropic"
    if has_g:
        return "gemini"
    return "mock"


def current_model(provider: str) -> str:
    return {"anthropic": CLAUDE_MODEL, "gemini": GEMINI_MODEL}.get(provider, "mock")


MAX_FRAMES = 12
MAX_FRAME_BYTES = 1_500_000
MAX_CROPS = 16
MAX_CROP_BYTES = 800_000

app = Flask(__name__, static_folder=os.path.join(HERE, "static"))
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024

SYSTEM_PROMPT = """You are a visual inspection assistant that helps people protect their privacy by \
spotting recording devices worn on a person. You examine a single video frame.

Look for:
1. Smart / AI glasses (e.g. camera-equipped eyewear: a small lens or LED on the frame front, \
thick temples with sensors/microphones, a camera module near the hinge or bridge). \
Ordinary prescription glasses and sunglasses are NOT smart glasses.
2. Other small camera candidates on or near the person: pinhole lens on buttons, pens, \
badges, hats, necklaces, bags, watches, or clothing.

Rules:
- Never identify who the person is and never describe their face beyond what is needed to \
locate eyewear.
- Be honest about uncertainty. Use low confidence when the view is blurry, small, or ambiguous. \
Do NOT claim certainty from one frame.
- Coordinates are normalized 0-1 relative to the image: x,y = top-left corner, w,h = size.
  Example: a box covering the right half of the image is {"x":0.5,"y":0,"w":0.5,"h":1}.
- If no person or no candidate is visible, say so.
- Check the face FIRST. If the person has ANY eyewear on (eyeglasses, sunglasses, smart glasses), \
set glasses.worn=true and box the whole eyewear. Only say worn=false if you can see the eyes \
clearly and there is no eyewear at all. Always report the face box when a face is visible, \
even if you are unsure about glasses.

Reply with ONLY a JSON object, no prose, in this schema:
{
  "person_visible": bool,
  "face": {"visible": bool, "bbox": {"x":0,"y":0,"w":0,"h":0} or null},
  "glasses": {"worn": bool, "kind": "none|regular|sunglasses|smart_glasses_suspected|smart_glasses_likely",
              "confidence": 0-1, "bbox": {"x":0,"y":0,"w":0,"h":0} or null, "evidence": "short"},
  "camera_candidates": [
    {"where": "short location, e.g. '
