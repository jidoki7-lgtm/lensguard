"""LensGuard 분석 서버.

브라우저가 영상에서 뽑은 프레임(JPEG, base64)을 받아 비전 API로 분석하고
스마트(AI) 안경 / 초소형 카메라 의심 위치를 JSON으로 돌려준다.

- 프레임은 메모리에서만 처리하고 디스크에 저장하지 않는다.
- 사람의 신원을 식별하지 않는다. 장비(안경, 렌즈) 유무와 위치만 본다.
- API 키가 없으면 MOCK 모드로 동작한다 (UI 테스트용 가짜 결과).
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
CLAUDE_MODEL = os.environ.get("LENSGUARD_MODEL", "claude-sonnet-4-5")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")


def get_provider() -> str:
    """어떤 AI를 쓸지 결정한다. 'anthropic' | 'gemini' | 'mock'."""
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
Ordinary prescription glasses and sunglasses are NOT smart glasses, unless ambiguous at a distance.
2. Other small camera candidates on or near the person: pinhole lens on buttons, pens, \
badges, hats, necklaces, bags, watches, or clothing.

Rules:
- Never identify who the person is and never describe their face beyond what is needed to \
locate eyewear.
- Use high-recall detection when eyewear or a person is small, distant, angled, or compressed: \
if thick frames, dark temples, or hinge/corner details might be smart glasses, mark \
kind="smart_glasses_suspected" with honest moderate confidence so a close-up crop can verify it.
- Coordinates are normalized 0-1 relative to the image: x,y = top-left corner, w,h = size.
  Example: a box covering the right half of the image is {"x":0.5,"y":0,"w":0.5,"h":1}.
- Check the face FIRST. If the person has ANY eyewear on (eyeglasses, sunglasses, smart glasses), \
set glasses.worn=true and box the whole eyewear. Always report the face box when a face is visible, \
even if you are unsure about glasses.

Reply with ONLY a JSON object, no prose, in this schema:
{
  "person_visible": bool,
  "face": {"visible": bool, "bbox": {"x":0,"y":0,"w":0,"h":0} or null},
  "glasses": {"worn": bool, "kind": "none|regular|sunglasses|smart_glasses_suspected|smart_glasses_likely",
              "confidence": 0-1, "bbox": {"x":0,"y":0,"w":0,"h":0} or null, "evidence": "short"},
  "camera_candidates": [
    {"where": "short location, e.g. 'left collar button'", "kind": "pinhole|lens|led|unknown",
     "confidence": 0-1, "bbox": {"x":0,"y":0,"w":0,"h":0}, "evidence": "short"}
  ],
  "image_quality": "good|fair|poor",
  "notes": "short"
}"""


def decode_image(data, limit=MAX_FRAME_BYTES):
    """data URL, base64 문자열, 또는 {'image': ...} 딕셔너리를 검증한다. (b64, 오류메시지) 반환."""
    if isinstance(data, dict):
        data = data.get("image") or data.get("dataUrl") or data.get("data")
    if not isinstance(data, str) or not data.strip():
        return None, "잘못된 이미지 데이터"
    b64 = re.sub(r"\s+", "", data.split(",", 1)[-1])
    if not b64:
        return None, "잘못된 이미지 데이터"
    pad = len(b64) % 4
    if pad:
        b64 += "=" * (4 - pad)
    try:
        raw = base64.b64decode(b64, validate=False)
    except Exception:
        return None, "잘못된 이미지 데이터"
    if len(raw) < 32:
        return None, "잘못된 이미지 데이터"
    if len(raw) > limit:
        return None, f"이미지가 너무 큽니다 ({limit // 1_000_000:.1f}MB 초과)"
    return b64, None


def extract_json(text: str):
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            return json.loads(m.group(0))
        raise


def norm_box(b):
    """AI가 돌려준 박스를 {x,y,w,h} (0~1)로 통일한다. 못 읽으면 None."""
    try:
        if isinstance(b, (list, tuple)) and len(b) == 4:
            y0, x0, y1, x1 = (float(v) for v in b)
            vals = (x0, y0, x1, y1)
        elif isinstance(b, dict):
            if all(k in b for k in ("xmin", "ymin", "xmax", "ymax")):
                x0, y0, x1, y1 = (float(b[k]) for k in ("xmin", "ymin", "xmax", "ymax"))
            elif all(k in b for k in ("x", "y", "w", "h")):
                x0, y0 = float(b["x"]), float(b["y"])
                x1, y1 = x0 + float(b["w"]), y0 + float(b["h"])
            elif "box_2d" in b:
                return norm_box(b["box_2d"])
            else:
                return None
            vals = (x0, y0, x1, y1)
        else:
            return None
        m = max(abs(v) for v in vals)
        if m > 1000.5:
            return None
        scale = 1000.0 if m > 1.5 else 1.0
        x0, y0, x1, y1 = (min(1.0, max(0.0, v / scale)) for v in vals)
        if x1 - x0 <= 0.002 or y1 - y0 <= 0.002:
            return None
        return {"x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0}
    except (TypeError, ValueError):
        return None


BOX_KEYS = ("bbox", "lens_bbox", "box_2d")


def normalize_boxes(obj):
    """결과 JSON 안의 모든 박스 필드를 norm_box로 정리한다 (제자리 수정)."""
    if isinstance(obj, dict):
        for k in list(obj):
            if k in BOX_KEYS:
                nb = norm_box(obj[k]) if obj[k] is not None else None
                if k == "box_2d":
                    del obj[k]
                    if nb and "bbox" not in obj:
                        obj["bbox"] = nb
                else:
                    obj[k] = nb
            else:
                normalize_boxes(obj[k])
    elif isinstance(obj, list):
        for it in obj:
            normalize_boxes(it)
    return obj


REFINE_COMMON = """You are a visual inspection assistant that helps people protect their privacy by \
spotting recording devices. You are given a CLOSE-UP CROP cut out of a larger video frame. \
An earlier coarse pass flagged this region for a closer look (it may have been wrong).

Rules:
- Never identify who the person is. Describe only equipment and objects.
- Be honest about uncertainty. Low resolution, blur or glare => lower confidence. Never claim certainty.
- Coordinates are normalized 0-1 relative to THIS CROP: x,y = top-left corner, w,h = size.
- Reply with ONLY a JSON object, no prose."""

REFINE_PROMPTS = {
    "glasses": REFINE_COMMON + """

Task: this crop shows the eye area of a person's face or eyewear frame. First decide whether ANY eyewear is visible \
(thin metal or rimless frames and tinted lenses count; if you see none, answer kind="none"). \
If eyewear is present, inspect it closely, especially the two outer front corners of the frame \
next to the hinges. Smart / AI glasses usually show some of: a small round camera \
lens or a pair of lens openings in the front frame corner near the hinge or bridge, a tiny LED \
or sensor window next to the lens, unusually thick temple arms (battery/electronics), \
touch pads or speaker openings on the temples, a charging-contact notch at the nose bridge. \
Ordinary glasses, sunglasses and plain fashion frames have none of these.

Schema:
{"kind": "none|regular|sunglasses|smart_glasses_suspected|smart_glasses_likely",
 "confidence": 0-1,
 "lens_found": bool,
 "lens_bbox": {"x":0,"y":0,"w":0,"h":0} or null,
 "features_seen": ["short feature", "..."],
 "features_missing": ["short", "..."],
 "evidence": "one short sentence"}""",
    "camera": REFINE_COMMON + """

Task: decide whether this spot is a hidden/miniature camera. Look for a tiny round dark glass \
lens (often with a thin ring or a purple/blue coating reflection), a pinhole in a button, pen clip, \
badge, hat, bag strap, or jewelry, or a small LED beside it. Ordinary buttons, screws, stitching \
holes, fabric texture and logos are NOT cameras.

Schema:
{"is_camera": "no|unlikely|possible|likely",
 "confidence": 0-1,
 "kind": "pinhole|lens|led|unknown",
 "object": "what the object is, e.g. 'shirt button'",
 "lens_bbox": {"x":0,"y":0,"w":0,"h":0} or null,
 "evidence": "one short sentence"}""",
    "glint": REFINE_COMMON + """

This crop is centred on a small bright reflection that appeared while the scene was lit by a \
phone flashlight. Camera lenses return light straight back, producing a sharp, tiny, very bright \
point. Decide what produced it. A lens usually looks like a small round dark glass disc with a \
bright pinpoint, a thin ring around it, and sits flush on a surface or frame. Other things that \
glint: eyes/corneas, eyeglass lenses, jewelry, metal buttons, zippers, screws, watch glass, \
sequins, wet or glossy surfaces, screens.

Schema:
{"source": "camera_lens|eye|eyeglass_lens|jewelry|metal_hardware|screen|glossy_surface|other",
 "is_camera": "no|unlikely|possible|likely",
 "confidence": 0-1,
 "object": "short description of the object holding the reflection",
 "lens_bbox": {"x":0,"y":0,"w":0,"h":0} or null,
 "evidence": "one short sentence"}""",
}


def _post_json(req: urllib.request.Request, retries: int = 3) -> dict:
    """429(한도 초과)·5xx는 잠깐 쉬었다가 다시 시도한다."""
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=90) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 529) and attempt < retries - 1:
                time.sleep(2 ** attempt + random.random())
                continue
            raise


def call_model(provider: str, jpeg_b64: str, system: str = SYSTEM_PROMPT,
               prompt: str = "Analyze this frame. JSON only.", max_tokens: int = 900) -> dict:
    if provider == "gemini":
        r = call_gemini(os.environ["GEMINI_API_KEY"].strip(), jpeg_b64, system, prompt, max_tokens)
    else:
        r = call_claude(os.environ["ANTHROPIC_API_KEY"].strip(), jpeg_b64, system, prompt, max_tokens)
    if not isinstance(r, dict):
        raise ValueError("bad_json_shape")
    return normalize_boxes(r)


def call_gemini(api_key: str, jpeg_b64: str, system: str, prompt: str, max_tokens: int) -> dict:
    body = {
        "systemInstruction": {"parts": [{"text": system}]},
        "contents": [{
            "role": "user",
            "parts": [
                {"inlineData": {"mimeType": "image/jpeg", "data": jpeg_b64}},
                {"text": prompt},
            ],
        }],
        "generationConfig": {
            "temperature": 0.2,
            "responseMimeType": "application/json",
            "maxOutputTokens": max(2048, max_tokens * 2),
        },
    }
    req = urllib.request.Request(
        GEMINI_URL.format(model=GEMINI_MODEL),
        data=json.dumps(body).encode(),
        headers={"content-type": "application/json", "x-goog-api-key": api_key},
        method="POST",
    )
    data = _post_json(req)
    cands = data.get("candidates") or
