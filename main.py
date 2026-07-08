import asyncio
import hashlib
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

# Add NVIDIA library paths to DLL search path and system PATH for CUDA 12 on Windows
import site

for prefix in site.getsitepackages():
    nvidia_dir = Path(prefix) / "nvidia"
    if nvidia_dir.exists():
        for root, dirs, files in os.walk(nvidia_dir):
            if any(f.endswith(".dll") for f in files):
                try:
                    os.add_dll_directory(root)
                except Exception:
                    pass
                os.environ["PATH"] = f"{root};" + os.environ["PATH"]

import httpx
from faster_whisper import WhisperModel
import edge_tts

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

# ==========================================
# CONSTANTS & CONFIGURATION
# ==========================================
STT_MODEL_SIZE = os.getenv("STT_MODEL_SIZE", "base")
STT_DEVICE = os.getenv("STT_DEVICE", "auto").lower()
OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "").strip()
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY", "").strip()
OLLAMA_MODEL_NAME = os.getenv("OLLAMA_MODEL_NAME", "qwen2.5:3b")
STT_HOTWORDS = os.getenv(
    "STT_HOTWORDS",
    "a1 a2 a3 a4 a5 a6 a7 a8 b1 b2 b3 b4 b5 b6 b7 b8 c1 c2 c3 c4 c5 c6 c7 c8 d1 d2 d3 d4 d5 d6 d7 d8 e1 e2 e3 e4 e5 e6 e7 e8 f1 f2 f3 f4 f5 f6 f7 f8 g1 g2 g3 g4 g5 g6 g7 g8 h1 h2 h3 h4 h5 h6 h7 h8 D4 ke D5 d4 ke d5 ke to dari menuju pion kuda gajah benteng ratu raja",
)
STT_INITIAL_PROMPT = os.getenv(
    "STT_INITIAL_PROMPT",
    "Chess voice command. Use board coordinates exactly, for example d4 ke d5, e2 ke e4, g1 ke f3. Letters are a b c d e f g h and ranks are 1 2 3 4 5 6 7 8.",
)
CHESS_WORDS = {
    "ah": "a",
    "bee": "b",
    "be": "b",
    "bi": "b",
    "cee": "c",
    "sea": "c",
    "see": "c",
    "ce": "c",
    "si": "c",
    "dee": "d",
    "de": "d",
    "di": "d",
    "the": "d",
    "ee": "e",
    "eh": "e",
    "eff": "f",
    "ef": "f",
    "gee": "g",
    "ge": "g",
    "ji": "g",
    "aitch": "h",
    "ha": "h",
    "one": "1",
    "satu": "1",
    "two": "2",
    "dua": "2",
    "three": "3",
    "tiga": "3",
    "four": "4",
    "empat": "4",
    "five": "5",
    "lima": "5",
    "rima": "5",
    "delima": "5",
    "six": "6",
    "enam": "6",
    "seven": "7",
    "tujuh": "7",
    "eight": "8",
    "delapan": "8",
}
DEFAULT_VOICES = {
    "id": "id-ID-GadisNeural",
    "en": "en-US-JennyNeural",
}
SUPPORTED_STT_LANGUAGES = {
    "af",
    "am",
    "ar",
    "as",
    "az",
    "ba",
    "be",
    "bg",
    "bn",
    "bo",
    "br",
    "bs",
    "ca",
    "cs",
    "cy",
    "da",
    "de",
    "el",
    "en",
    "es",
    "et",
    "eu",
    "fa",
    "fi",
    "fo",
    "fr",
    "gl",
    "gu",
    "ha",
    "haw",
    "he",
    "hi",
    "hr",
    "ht",
    "hu",
    "hy",
    "id",
    "is",
    "it",
    "ja",
    "jw",
    "ka",
    "kk",
    "km",
    "kn",
    "ko",
    "la",
    "lb",
    "ln",
    "lo",
    "lt",
    "lv",
    "mg",
    "mi",
    "mk",
    "ml",
    "mn",
    "mr",
    "ms",
    "mt",
    "my",
    "ne",
    "nl",
    "nn",
    "no",
    "oc",
    "pa",
    "pl",
    "ps",
    "pt",
    "ro",
    "ru",
    "sa",
    "sd",
    "si",
    "sk",
    "sl",
    "sn",
    "so",
    "sq",
    "sr",
    "su",
    "sv",
    "sw",
    "ta",
    "te",
    "tg",
    "th",
    "tk",
    "tl",
    "tr",
    "tt",
    "uk",
    "ur",
    "uz",
    "vi",
    "yi",
    "yo",
    "zh",
    "yue",
}

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("voicechess.ai")


def log_event(event: str, **fields) -> None:
    payload = {"event": event, **fields}
    logger.info(json.dumps(payload, default=str, ensure_ascii=False))


def ollama_headers() -> dict[str, str]:
    if not OLLAMA_API_KEY:
        return {}
    return {"Authorization": f"Bearer {OLLAMA_API_KEY}"}


# ==========================================
# GATEWAY RESPONSE MODELS
# ==========================================
@dataclass
class VoiceRequestConfig:
    language: str
    tts_voice: str
    tts_rate: str
    enabled: bool
    context: str


class MoveData(BaseModel):
    # ponytail: flat move fields, expand to a nested object only if the BE needs more chess metadata
    uci: Optional[str] = None
    san: Optional[str] = None
    from_square: Optional[str] = None
    to_square: Optional[str] = None
    promotion: Optional[str] = None
    confidence: float = 0.0


class LatencyBreakdown(BaseModel):
    audio_ms: float
    stt_ms: float
    response_ms: float
    move_ms: float
    ai_ms: float
    tts_ms: float
    total_ms: float


class ProcessAudioResponse(BaseModel):
    text: str
    message: str
    audio_url: Optional[str] = None
    audioUrl: Optional[str] = None
    language: str = "en"
    confidence: float = 0.0
    response_source: str
    responseSource: str
    move: MoveData
    latency_ms: float
    latency: LatencyBreakdown


class TranscribeResponse(BaseModel):
    text: str
    language: str = "en"
    confidence: float = 0.0
    latency_ms: float


class TestTTSRequest(BaseModel):
    text: Optional[str] = None


# ==========================================
# VOICE PROCESSOR (WHISPER + OLLAMA API + EDGE-TTS)
# ==========================================
class VoiceProcessor:
    def __init__(self) -> None:
        self.language = os.getenv("VOICE_LANGUAGE", "en")
        self.tts_voice = os.getenv("VOICE_TTS_VOICE", "en-US-JennyNeural")
        self.tts_rate = os.getenv("VOICE_TTS_RATE", "-10%")
        self.audio_cache_dir = Path(os.getenv("VOICE_AUDIO_CACHE_DIR", "./audio_cache"))
        self.audio_cache_dir.mkdir(parents=True, exist_ok=True)

        self.stt_model = None
        self.stt_device = "cpu"

    def load(self) -> None:
        if STT_DEVICE in {"cuda", "gpu"}:
            log_event("stt_load_start", device="cuda", model=STT_MODEL_SIZE)
            self.stt_model = WhisperModel(
                STT_MODEL_SIZE, device="cuda", compute_type="float16"
            )
            self.stt_device = "cuda"
            log_event("stt_load_success", device="cuda", model=STT_MODEL_SIZE)
            return

        if STT_DEVICE == "cpu":
            log_event("stt_load_start", device="cpu", model=STT_MODEL_SIZE)
            self.stt_model = WhisperModel(
                STT_MODEL_SIZE, device="cpu", compute_type="int8", cpu_threads=4
            )
            self.stt_device = "cpu"
            log_event("stt_load_success", device="cpu", model=STT_MODEL_SIZE)
            return

        try:
            log_event("stt_load_start", device="cuda", model=STT_MODEL_SIZE)
            self.stt_model = WhisperModel(
                STT_MODEL_SIZE, device="cuda", compute_type="float16"
            )
            self.stt_device = "cuda"
            log_event("stt_load_success", device="cuda", model=STT_MODEL_SIZE)
        except Exception as e:
            log_event(
                "stt_load_failed", device="cuda", model=STT_MODEL_SIZE, error=str(e)
            )
            log_event("stt_load_start", device="cpu", model=STT_MODEL_SIZE)
            self.stt_model = WhisperModel(
                STT_MODEL_SIZE, device="cpu", compute_type="int8", cpu_threads=4
            )
            self.stt_device = "cpu"
            log_event("stt_load_success", device="cpu", model=STT_MODEL_SIZE)

    def config_from_request(self, request: Request) -> VoiceRequestConfig:
        return VoiceRequestConfig(
            language=self._normalize_language(
                request.headers.get("x-voice-language", self.language)
            ),
            tts_voice=request.headers.get("x-voice-tts-voice", ""),
            tts_rate=request.headers.get("x-voice-rate", self.tts_rate),
            enabled=request.headers.get("x-voice-enabled", "true").lower() != "false",
            context=request.headers.get("x-voice-context", "").strip(),
        )

    # ── Layer 0: STT (Local Faster-Whisper) ───────────────────────────────────
    async def transcribe(
        self, audio_bytes: bytes, target_language: Optional[str] = None
    ) -> tuple[str, str, float]:
        if self.stt_model is None:
            raise RuntimeError("STT model is not initialized")

        temp_file = self.audio_cache_dir / f"temp_{time.time()}_transcribe.wav"
        with open(temp_file, "wb") as f:
            f.write(audio_bytes)

        try:
            log_event(
                "stt_transcribe_started",
                bytes=len(audio_bytes),
                target_language=target_language,
                device=self.stt_device,
            )
            start = time.perf_counter()
            loop = asyncio.get_running_loop()

            def run_whisper():
                segments, info = self.stt_model.transcribe(
                    str(temp_file),
                    beam_size=5,
                    language=target_language,
                    initial_prompt=STT_INITIAL_PROMPT,
                    hotwords=STT_HOTWORDS,
                    condition_on_previous_text=False,
                    vad_filter=True,
                )
                transcript = " ".join([segment.text for segment in segments]).strip()
                return transcript, info.language, info.language_probability

            transcript, detected_lang, confidence = await loop.run_in_executor(
                None, run_whisper
            )
            raw_transcript = transcript
            transcript = self._normalize_chess_transcript(transcript)

            if confidence < 0.7:
                log_event(
                    "stt_low_confidence",
                    confidence=round(confidence, 4),
                    detected_language=detected_lang,
                )
                if detected_lang != "en":
                    detected_lang = "en"

            if detected_lang not in ["id", "en"]:
                detected_lang = "en"

            log_event(
                "stt_transcribe_completed",
                duration_ms=round((time.perf_counter() - start) * 1000, 2),
                detected_language=detected_lang,
                confidence=round(confidence, 4),
                text_length=len(transcript),
                transcript=transcript,
                raw_transcript=raw_transcript,
            )
            return transcript, detected_lang, confidence
        finally:
            if temp_file.exists():
                temp_file.unlink()

    # ── Layer 1: Conversational Response (Ollama API) ────────────────────────
    async def response_for(
        self, transcript: str, language: str, context: str = ""
    ) -> tuple[str, str]:
        text = self._normalize(transcript)
        if not text:
            return (
                "I'm sorry, I didn't catch that. Could you please say your move again?",
                "fallback",
            )

        if not OLLAMA_API_URL:
            return (
                "I couldn't reach the chess engine. Please try your move again.",
                "fallback",
            )

        try:
            context_instruction = f"Current board (FEN): {context}\n" if context else ""
            system_instruction = (
                "You are the voice companion for a chess app. "
                f"{context_instruction}"
                "The user speaks a chess move or asks about the game. "
                "Reply warmly and clearly in EXACTLY 1 short sentence confirming the move or answering. "
                "Do NOT explain chess theory unless asked."
            )

            payload = {
                "model": OLLAMA_MODEL_NAME,
                "messages": [
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": transcript},
                ],
                "options": {"temperature": 0.4, "num_predict": 50},
                "keep_alive": -1,
                "stream": False,
            }

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    OLLAMA_API_URL, json=payload, headers=ollama_headers(), timeout=20.0
                )
                response.raise_for_status()
                data = response.json()
                content = self._normalize_response(data["message"]["content"] or "")
                if content:
                    return content[:260], "ollama_qwen"

        except Exception as e:
            log_event(
                "ollama_response_failed",
                error=str(e),
                language=language,
                model=OLLAMA_MODEL_NAME,
            )

        return "Okay, I noted your move.", "fallback"

    # ── Layer 2: Move Extraction (Ollama JSON) ───────────────────────────────
    async def parse_move(
        self, transcript: str, language: str, context: str = ""
    ) -> MoveData:
        text = self._normalize(transcript)
        if not text or not OLLAMA_API_URL:
            return MoveData()

        try:
            prompt = (
                f'Board FEN: "{context or "unknown"}"\n'
                f'User speech: "{transcript}"\n\n'
                "Extract the chess move the user wants to play. "
                "Respond with JSON only (no markdown, no explanation):\n"
                '{"uci":"e2e4 or null","san":"e4 or null","from_square":"e2 or null",'
                '"to_square":"e4 or null","promotion":"q|r|b|n or null","confidence":0.0}'
            )
            payload = {
                "model": OLLAMA_MODEL_NAME,
                "messages": [
                    {
                        "role": "system",
                        "content": "You convert spoken chess moves into structured JSON. Respond strictly in JSON.",
                    },
                    {"role": "user", "content": prompt},
                ],
                "options": {"temperature": 0.1, "num_predict": 80},
                "keep_alive": -1,
                "stream": False,
            }
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    OLLAMA_API_URL, json=payload, headers=ollama_headers(), timeout=15.0
                )
                response.raise_for_status()
                data = response.json()
                result_text = data["message"]["content"].strip()
                match = re.search(r"\{.*?\}", result_text, re.DOTALL)
                if match:
                    parsed = json.loads(match.group())
                    return MoveData(
                        uci=self._none_if_null(parsed.get("uci")),
                        san=self._none_if_null(parsed.get("san")),
                        from_square=self._none_if_null(parsed.get("from_square")),
                        to_square=self._none_if_null(parsed.get("to_square")),
                        promotion=self._none_if_null(parsed.get("promotion")),
                        confidence=float(parsed.get("confidence", 0.5)),
                    )
        except Exception as e:
            log_event(
                "ollama_move_failed",
                error=str(e),
                language=language,
                model=OLLAMA_MODEL_NAME,
            )

        return MoveData()

    # ── Local TTS (Edge-TTS) ──────────────────────────────────────────────────
    async def generate_audio(
        self, text: str, cfg: Optional[VoiceRequestConfig] = None, language: str = "en"
    ) -> Optional[str]:
        if cfg and not cfg.enabled:
            return None

        clean_text = self._clean_for_tts(text)
        if not clean_text:
            return None

        header_voice = cfg.tts_voice if cfg else None
        configured_default = self.tts_voice or DEFAULT_VOICES.get(
            language, DEFAULT_VOICES["en"]
        )
        voice = header_voice or configured_default
        if not header_voice and language in DEFAULT_VOICES:
            voice = DEFAULT_VOICES[language]

        rate = cfg.tts_rate if cfg else self.tts_rate

        cache_key = f"{voice}_{rate}_{clean_text}"
        filename = f"tts_{hashlib.md5(cache_key.encode()).hexdigest()[:12]}.mp3"
        path = self.audio_cache_dir / filename

        if not path.exists():
            communicate = edge_tts.Communicate(clean_text, voice, rate=rate)
            await communicate.save(str(path))

        return f"/api/audio/{filename}"

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _none_if_null(self, value) -> Optional[str]:
        if value is None:
            return None
        s = str(value).strip()
        return None if s.lower() in {"", "null", "none"} else s

    def _normalize_language(self, language: str) -> str:
        value = (language or "en").strip().lower().split("-")[0]
        return value if value in SUPPORTED_STT_LANGUAGES else "en"

    def _normalize_chess_transcript(self, transcript: str) -> str:
        text = transcript.lower().strip()
        for source, target in CHESS_WORDS.items():
            text = re.sub(rf"\b{re.escape(source)}\b", target, text)
        text = re.sub(r"\bke\s+b\s+e\s*([1-8])\b", r"ke d\1", text)
        text = re.sub(r"\bke\s+b\s+e([1-8])\b", r"ke d\1", text)
        text = re.sub(r"\b(kadeh|kade|ke de|ke d)\b", "ke d", text)
        text = re.sub(r"\b([a-h])\s+([1-8])\b", r"\1\2", text)
        text = re.sub(
            r"\b([a-h][1-8])\s+(?:2|,|go|to|tu|ke|menuju|pindah ke)\s+([a-h])\s*([1-8])\b",
            r"\1 ke \2\3",
            text,
        )
        text = re.sub(
            r"\b([a-h])\s*([1-8])\s*,?\s+([a-h])\s*([1-8])\b", r"\1\2 ke \3\4", text
        )
        return text.strip()

    def _normalize(self, text: str) -> str:
        return re.sub(r"\s+", " ", text.lower()).strip()

    def _normalize_response(self, text: str) -> str:
        return re.sub(r"\s+", " ", text).strip().strip('"')

    def _clean_for_tts(self, text: str) -> str:
        return re.sub(r"[^\w\s.,!?;:'\-()À-ɏ]", "", text).strip()


# ==========================================
# FASTAPI GATEWAY ENGINE
# ==========================================
processor = VoiceProcessor()
app = FastAPI(title="VoiceChess Voice Service (Ollama Gateway)", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_logger(request: Request, call_next):
    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception as error:
        duration_ms = round((time.perf_counter() - start) * 1000, 2)
        log_event(
            "request_failed",
            method=request.method,
            path=request.url.path,
            duration_ms=duration_ms,
            error=str(error),
        )
        raise

    duration_ms = round((time.perf_counter() - start) * 1000, 2)
    log_event(
        "request_completed",
        method=request.method,
        path=request.url.path,
        status=response.status_code,
        duration_ms=duration_ms,
    )
    return response


@app.on_event("startup")
async def startup() -> None:
    log_event(
        "startup",
        stt_model=STT_MODEL_SIZE,
        stt_device=STT_DEVICE,
        ollama_configured=bool(OLLAMA_API_URL),
        ollama_api_key_configured=bool(OLLAMA_API_KEY),
    )

    # ponytail: load Whisper in a background thread so the server (and /health)
    # comes up immediately. Downloading "base" on CPU can take minutes; blocking
    # startup here makes Railway's healthcheck fail with "service unavailable".
    async def _load_stt() -> None:
        try:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, processor.load)
            log_event(
                "startup_ready",
                active_stt_device=processor.stt_device,
                default_language=processor.language,
                default_voice=processor.tts_voice,
            )
        except Exception as error:
            log_event("startup_stt_failed", error=str(error), model=STT_MODEL_SIZE)

    asyncio.create_task(_load_stt())


@app.get("/health")
async def health() -> dict:
    # ponytail: liveness only — must answer instantly so Railway's healthcheck
    # passes even while Whisper is still loading. Deep checks live in /health/deep.
    return {
        "status": "ok",
        "stt_ready": processor.stt_model is not None,
        "llm_model": f"Ollama ({OLLAMA_MODEL_NAME})",
    }


@app.get("/health/deep")
async def health_deep() -> dict:
    status = "not_configured"
    if OLLAMA_API_URL:
        try:
            async with httpx.AsyncClient() as client:
                res = await client.get(
                    OLLAMA_API_URL.split("/api")[0] + "/",
                    headers=ollama_headers(),
                    timeout=2.0,
                )
                status = (
                    "healthy"
                    if res.status_code == 200
                    else "unhealthy_ollama_not_responding"
                )
        except Exception as error:
            log_event("health_deep_ollama_failed", error=str(error))
            status = "ollama_not_reachable"

    return {
        "status": "healthy"
        if processor.stt_model is not None and status == "healthy"
        else "degraded",
        "ollama_status": status,
        "stt_model": f"Faster-Whisper (Local {processor.stt_device.upper()})",
        "llm_model": f"Ollama ({OLLAMA_MODEL_NAME})",
        "language": processor.language,
    }


@app.post("/api/transcribe", response_model=TranscribeResponse)
async def transcribe_audio(request: Request) -> TranscribeResponse:
    cfg = processor.config_from_request(request)
    start = time.perf_counter()
    audio_bytes = await request.body()

    if len(audio_bytes) < 1000:
        log_event(
            "voice_rejected",
            reason="audio_too_short",
            bytes=len(audio_bytes),
            language=cfg.language,
        )
        raise HTTPException(status_code=400, detail="Audio stream is too short")

    if processor.stt_model is None:
        log_event("voice_rejected", reason="stt_model_loading", language=cfg.language)
        raise HTTPException(status_code=503, detail="Speech model is still loading")

    try:
        text, language, confidence = await processor.transcribe(
            audio_bytes, cfg.language
        )
    except Exception as error:
        log_event("voice_transcribe_failed", error=str(error), language=cfg.language)
        raise HTTPException(
            status_code=500, detail="Voice transcription failed"
        ) from error

    total_ms = round((time.perf_counter() - start) * 1000, 2)
    log_event(
        "voice_transcribe_completed",
        total_ms=total_ms,
        language=language,
        confidence=confidence,
        text_length=len(text),
        transcript=text,
    )
    return TranscribeResponse(
        text=text, language=language, confidence=confidence, latency_ms=total_ms
    )


@app.post("/api/process-audio", response_model=ProcessAudioResponse)
async def process_audio(request: Request) -> ProcessAudioResponse:
    cfg = processor.config_from_request(request)
    start = time.perf_counter()
    audio_bytes = await request.body()
    audio_ms = round((time.perf_counter() - start) * 1000, 2)

    if len(audio_bytes) < 1000:
        log_event(
            "voice_rejected",
            reason="audio_too_short",
            bytes=len(audio_bytes),
            language=cfg.language,
        )
        raise HTTPException(status_code=400, detail="Audio stream is too short")

    log_event(
        "voice_processing_started",
        bytes=len(audio_bytes),
        language=cfg.language,
        voice=cfg.tts_voice or processor.tts_voice,
        rate=cfg.tts_rate,
        voice_enabled=cfg.enabled,
    )

    if processor.stt_model is None:
        log_event("voice_rejected", reason="stt_model_loading", language=cfg.language)
        raise HTTPException(status_code=503, detail="Speech model is still loading")

    try:
        # Layer 0: STT — must complete before move/response generation
        stt_start = time.perf_counter()
        transcript, language, confidence = await processor.transcribe(
            audio_bytes, cfg.language
        )
        stt_ms = round((time.perf_counter() - stt_start) * 1000, 2)

        # Layer 1 (spoken reply) + Layer 2 (structured move) run concurrently
        ai_start = time.perf_counter()

        async def timed_response():
            start_at = time.perf_counter()
            result = await processor.response_for(transcript, language, cfg.context)
            return result, round((time.perf_counter() - start_at) * 1000, 2)

        async def timed_move():
            start_at = time.perf_counter()
            result = await processor.parse_move(transcript, language, cfg.context)
            return result, round((time.perf_counter() - start_at) * 1000, 2)

        (
            ((message, response_source), response_ms),
            (move, move_ms),
        ) = await asyncio.gather(
            timed_response(),
            timed_move(),
        )
        ai_ms = round((time.perf_counter() - ai_start) * 1000, 2)

        # TTS Synthesis — use per-request settings and pass language for auto voice packs
        tts_start = time.perf_counter()
        audio_url = await processor.generate_audio(message, cfg, language)
        tts_ms = round((time.perf_counter() - tts_start) * 1000, 2)

    except Exception as error:
        log_event(
            "voice_processing_failed",
            error=str(error),
            language=cfg.language,
            voice=cfg.tts_voice or processor.tts_voice,
        )
        raise HTTPException(status_code=500, detail="Voice processing failed")

    total_ms = round((time.perf_counter() - start) * 1000, 2)
    latency = LatencyBreakdown(
        audio_ms=audio_ms,
        stt_ms=stt_ms,
        response_ms=response_ms,
        move_ms=move_ms,
        ai_ms=ai_ms,
        tts_ms=tts_ms,
        total_ms=total_ms,
    )
    log_event(
        "voice_processing_completed",
        stt_ms=stt_ms,
        response_ms=response_ms,
        move_ms=move_ms,
        ai_ms=ai_ms,
        tts_ms=tts_ms,
        total_ms=total_ms,
        response_source=response_source,
        move_uci=move.uci,
        move_confidence=move.confidence,
        language=language,
        confidence=confidence,
        voice=cfg.tts_voice or processor.tts_voice,
        audio_cached=bool(audio_url),
    )
    return ProcessAudioResponse(
        text=transcript,
        message=message,
        audio_url=audio_url,
        audioUrl=audio_url,
        language=language,
        confidence=confidence,
        response_source=response_source,
        responseSource=response_source,
        move=move,
        latency_ms=total_ms,
        latency=latency,
    )


@app.post("/api/test-tts")
async def test_tts(request: Request, body: TestTTSRequest) -> dict:
    cfg = processor.config_from_request(request)
    text = (
        body.text
        or "Hello! I am your VoiceChess companion. Say your move whenever you are ready."
    )
    audio_url = await processor.generate_audio(text, cfg, cfg.language)
    log_event(
        "tts_preview_generated",
        language=cfg.language,
        voice=cfg.tts_voice or processor.tts_voice,
        rate=cfg.tts_rate,
        audio_cached=bool(audio_url),
    )
    return {"audio_url": audio_url, "audioUrl": audio_url, "text": text}


@app.get("/api/audio/{filename}")
def get_audio(filename: str) -> FileResponse:
    # ponytail: reject path traversal; filename must be a bare tts_*.mp3 we generated
    if not re.fullmatch(r"tts_[a-f0-9]{12}\.mp3", filename):
        raise HTTPException(status_code=400, detail="Invalid filename")
    path = processor.audio_cache_dir / filename
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="Audio not found")
    return FileResponse(str(path), media_type="audio/mpeg", filename=filename)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
