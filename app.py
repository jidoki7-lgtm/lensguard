"""LensGuard 분석 서버.

브라우저가 영상에서 뽑은 프레임(JPEG, base64)을 받아 Gemini 비전 API로 분석하고
스마트(AI) 안경 / 초소형 카메라 의심 위치를 JSON으로 돌려준다.

- 프레임은 메모리에서만 처리하고 디스크에 저장하지 않는다.
- 사람의 신원을 식별하지 않는다. 장비(안경, 렌즈) 유무와 위치만 본다.
- GEMINI_API_KEY 가 없으면 MOCK 모드로 동작한다 (UI 테스트용 가짜 결과).

실행:
    export GEMINI_API_KEY=AIza...
    python3 app.py
"""
import base64
import json
import os
import random
import re
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from flask import Flask, jsonify, request, send_from_directory

HERE = os.path.dirname(os.path.abspath(__file__))
API_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"
MODEL = os.environ.get("LENSGUARD_MODEL", "gemini-3.8-flash")
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
- If no person or no candidate is visible, say so.

Reply with ONLY a JSON object, no prose, in this schema:
{
  "person_visible": bool,
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
    """data URL 또는 base64 문자열을 검증한다. (b64, 오류메시지) 반환."""
    if not isinstance(data, str):
        return None, "잘못된 이미지 데이터"
    b64 = data.split(",", 1)[-1]
    try:
        raw = base64.b64decode(b64, validate=True)
    except Exception:
        return None, "잘못된 이미지 데이터"
    if len(raw) > limit:
        return None, f"이미지가 너무 큽니다 ({limit // 1_000_000:.1f}MB 초과)"
    return b64, None


def extract_json(text: str):
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            return json.loads(m.group(0))
        raise


REFINE_COMMON = """You are a visual inspection assistant that helps people protect their privacy by \
spotting recording devices. You are given a CLOSE-UP CROP cut out of a larger video frame. \
An earlier coarse pass flagged this region as possibly containing a camera.

Rules:
- Never identify who the person is. Describe only equipment and objects.
- Be honest about uncertainty. Low resolution, blur or glare => lower confidence. Never claim certainty.
- Coordinates are normalized 0-1 relative to THIS CROP: x,y = top-left corner, w,h = size.
- Reply with ONLY a JSON object, no prose."""

REFINE_PROMPTS = {
    "glasses": REFINE_COMMON + """

Task: inspect the eyewear closely. Smart / AI glasses usually show some of: a small round camera \
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


def call_gemini(api_key: str, jpeg_b64: str, system: str = SYSTEM_PROMPT,
                prompt: str = "Analyze this frame. JSON only.", max_tokens: int = 900) -> dict:
    url = f"{API_BASE_URL}/{urllib.parse.quote(MODEL)}:generateContent?key={urllib.parse.quote(api_key)}"
    body = {
        "systemInstruction": {
            "parts": [{"text": system}]
        },
        "contents": [
            {
                "role": "user",
                "parts": [
                    {
                        "inlineData": {
                            "mimeType": "image/jpeg",
                            "data": jpeg_b64,
                        }
                    },
                    {"text": prompt},
                ],
            }
        ],
        "generationConfig": {
            "maxOutputTokens": max_tokens,
            "responseMimeType": "application/json",
        },
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"content-type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read())

    candidates = data.get("candidates") or []
    parts = ((candidates[0].get("content") or {}).get("parts")) if candidates else []
    text = "".join(p.get("text", "") for p in (parts or []) if isinstance(p.get("text"), str))
    return extract_json(text)


def mock_result(i: int) -> dict:
    rnd = random.Random(i)
    suspicious = rnd.random() > 0.4
    return {
        "person_visible": True,
        "glasses": {
            "worn": True,
            "kind": "smart_glasses_suspected" if suspicious else "regular",
            "confidence": round(rnd.uniform(0.45, 0.8), 2) if suspicious else 0.8,
            "bbox": {"x": 0.38, "y": 0.18, "w": 0.24, "h": 0.1},
            "evidence": "MOCK: 데모 데이터입니다 (API 키 없음).",
        },
        "camera_candidates": (
            [
                {
                    "where": "MOCK: 가슴 단추",
                    "kind": "pinhole",
                    "confidence": 0.4,
                    "bbox": {"x": 0.47, "y": 0.52, "w": 0.06, "h": 0.06},
                    "evidence": "MOCK 데모",
                }
            ]
            if suspicious
            else []
        ),
        "image_quality": "fair",
        "notes": "MOCK 모드",
    }


def mock_refine(kind: str, i: int) -> dict:
    rnd = random.Random(1000 + i)
    c = round(rnd.uniform(0.35, 0.8), 2)
    box = {"x": 0.35, "y": 0.35, "w": 0.3, "h": 0.3}
    if kind == "glasses":
        return {"kind": "smart_glasses_suspected", "confidence": c, "lens_found": True,
                "lens_bbox": box, "features_seen": ["MOCK: 모서리 렌즈"], "features_missing": [],
                "evidence": "MOCK: 데모 데이터입니다 (API 키 없음)."}
    if kind == "glint":
        return {"source": "camera_lens" if c > 0.6 else "metal_hardware",
                "is_camera": "possible" if c > 0.6 else "unlikely", "confidence": c,
                "object": "MOCK 물체", "lens_bbox": box, "evidence": "MOCK: 데모 데이터입니다."}
    return {"is_camera": "possible" if c > 0.6 else "unlikely", "confidence": c, "kind": "pinhole",
            "object": "MOCK 단추", "lens_bbox": box, "evidence": "MOCK: 데모 데이터입니다."}


def run_parallel(items, fn, workers=4):
    """items 순서를 유지하며 병렬로 fn(i, item)을 실행한다."""
    out = [None] * len(items)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fn, i, it): i for i, it in enumerate(items)}
        for fut in futs:
            out[futs[fut]] = fut.result()
    return out


@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.get("/sw.js")
def service_worker():
    # 서비스 워커는 사이트 전체(/)를 제어해야 하므로 루트 경로로 제공한다.
    resp = send_from_directory(app.static_folder, "sw.js", mimetype="application/javascript")
    resp.headers["Cache-Control"] = "no-cache"
    return resp


@app.get("/api/status")
def status():
    return jsonify({"mock": not bool(os.environ.get("GEMINI_API_KEY")), "model": MODEL})


@app.post("/api/analyze")
def analyze():
    payload = request.get_json(silent=True) or {}
    frames = payload.get("frames")
    if not isinstance(frames, list) or not frames:
        return jsonify({"error": "frames 배열이 필요합니다."}), 400
    if len(frames) > MAX_FRAMES:
        return jsonify({"error": f"한 번에 최대 {MAX_FRAMES}프레임까지 가능합니다."}), 400

    api_key = os.environ.get("GEMINI_API_KEY", "")

    def work(i, f):
        b64, err = decode_image(f)
        if err:
            return {"index": i, "error": err}
        try:
            r = call_gemini(api_key, b64) if api_key else mock_result(i)
            r["index"] = i
            return r
        except urllib.error.HTTPError as e:
            return {"index": i, "error": f"API 오류 {e.code}"}
        except Exception as e:  # noqa: BLE001
            return {"index": i, "error": f"분석 실패: {type(e).__name__}"}

    return jsonify({"mock": not bool(api_key), "results": run_parallel(frames, work)})


@app.post("/api/refine")
def refine():
    """잘라낸 영역(crop)을 종류별 정밀 프롬프트로 다시 분석한다.

    요청: {"crops": [{"image": "data:image/jpeg;base64,...", "kind": "glasses|camera|glint"}]}
    """
    payload = request.get_json(silent=True) or {}
    crops = payload.get("crops")
    if not isinstance(crops, list) or not crops:
        return jsonify({"error": "crops 배열이 필요합니다."}), 400
    if len(crops) > MAX_CROPS:
        return jsonify({"error": f"한 번에 최대 {MAX_CROPS}개 영역까지 가능합니다."}), 400

    api_key = os.environ.get("GEMINI_API_KEY", "")

    def work(i, c):
        kind = c.get("kind") if isinstance(c, dict) else None
        if kind not in REFINE_PROMPTS:
            return {"index": i, "error": "알 수 없는 kind"}
        b64, err = decode_image(c.get("image"), MAX_CROP_BYTES)
        if err:
            return {"index": i, "error": err}
        try:
            if api_key:
                r = call_gemini(api_key, b64, system=REFINE_PROMPTS[kind],
                                prompt="Analyze this close-up crop. JSON only.", max_tokens=600)
            else:
                r = mock_refine(kind, i)
            r["index"] = i
            r["kind_requested"] = kind
            return r
        except urllib.error.HTTPError as e:
            return {"index": i, "error": f"API 오류 {e.code}"}
        except Exception as e:  # noqa: BLE001
            return {"index": i, "error": f"분석 실패: {type(e).__name__}"}

    return jsonify({"mock": not bool(api_key), "results": run_parallel(crops, work)})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
