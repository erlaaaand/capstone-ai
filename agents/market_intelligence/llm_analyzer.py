# agents/market_intelligence/llm_analyzer.py
"""
LLM Analyzer — ekstrak data harga dari teks mentah menggunakan Ollama.

Strategi:
  - httpx langsung ke Ollama /api/chat (tanpa LangChain)
  - Prompt spesifik untuk D197 (Musang King) dan D200 (Duri Hitam)
  - JSON-only output dengan retry + repair prompt jika LLM menghasilkan non-JSON
  - Truncate input agar tidak melebihi context window
"""

from __future__ import annotations

import json
import re
from typing import List, Optional, Tuple

import httpx

from core.logger import get_logger
from agents.market_intelligence.config import OLLAMA_CONFIG, OllamaConfig
from agents.market_intelligence.schemas import MarketPriceEntry, ScrapedPage

logger = get_logger("agent.llm_analyzer")


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
Kamu adalah analis harga komoditas pertanian yang sangat teliti dan terstruktur.
Tugasmu adalah mengekstrak informasi harga durian premium dari teks marketplace/forum Indonesia.

PENGETAHUAN DOMAIN — WAJIB DIINGAT:
- "Musang King", "Raja Kunyit", "Mao Shan Wang", "MSW", "MK", "D197" → variety_code = "D197"
- "Duri Hitam", "Black Thorn", "Ochee", "Ochee Durian", "D200"       → variety_code = "D200"
- ABAIKAN varietas lain: D13, D24, D2, monthong, petruk, bawor, dll.
- Harga SELALU IDR. Konversi: "350rb"→350000, "1.2jt"→1200000, "Rp 450.000"→450000.

FORMAT OUTPUT:
Kembalikan HANYA array JSON valid tanpa markdown, komentar, atau teks lain.
Jika tidak ada data relevan, kembalikan: []

SCHEMA TIAP ELEMEN:
{
  "variety_code": "D197" | "D200",
  "variety_alias": "<nama di teks>",
  "price_per_kg_min": <float|null>,
  "price_per_kg_max": <float|null>,
  "price_per_kg_avg": <float|null>,
  "price_per_unit_min": <float|null>,
  "price_per_unit_max": <float|null>,
  "location_hint": "<kota/region/'online'|null>",
  "seller_type": "<'kebun'|'reseller'|'importir'|'toko online'|null>",
  "confidence": <0.0–1.0>,
  "raw_text_snippet": "<kutipan sumber maks 200 karakter>"
}"""

_USER_PROMPT_TEMPLATE = """\
Teks dari {source_name} ({source_url}):

--- TEKS SUMBER ---
{raw_text}
--- AKHIR TEKS ---

Kembalikan HANYA array JSON. Tidak ada teks sebelum atau sesudah JSON."""

_REPAIR_PROMPT = """\
Output sebelumnya bukan JSON valid. Perbaiki dan kembalikan HANYA array JSON yang valid.
Jika tidak bisa diperbaiki, kembalikan [].

Output sebelumnya:
{previous_output}"""


# ---------------------------------------------------------------------------
# Ollama client
# ---------------------------------------------------------------------------

async def _call_ollama(messages: List[dict], config: OllamaConfig) -> str:
    payload = {
        "model":    config.model,
        "messages": messages,
        "stream":   False,
        "options":  {
            "temperature": config.temperature,
            "top_p":       config.top_p,
            "num_predict": 4096,
        },
    }
    async with httpx.AsyncClient(timeout=config.timeout_sec) as client:
        resp = await client.post(f"{config.base_url}/api/chat", json=payload)
        resp.raise_for_status()
        data = resp.json()

    content: str = data.get("message", {}).get("content", "")
    if not content:
        raise ValueError(f"Ollama mengembalikan respons kosong. Raw: {data}")
    return content


# ---------------------------------------------------------------------------
# JSON parsing
# ---------------------------------------------------------------------------

def _extract_json_array(text: str) -> Optional[list]:
    """Parse JSON dari output LLM; tangani markdown code fence jika ada."""
    cleaned = re.sub(r"```(?:json)?\s*", "", text).strip()
    cleaned = re.sub(r"```\s*$", "", cleaned).strip()

    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, list):
            return parsed
        # LLM kadang wrap dalam {"data": [...]}
        if isinstance(parsed, dict):
            for key in ("data", "results", "entries", "prices"):
                if isinstance(parsed.get(key), list):
                    return parsed[key]
    except json.JSONDecodeError:
        pass

    # Last resort: cari array dengan regex
    m = re.search(r"\[.*\]", cleaned, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass

    return None


def _validate_entries(raw_list: list, source_name: str) -> Tuple[List[MarketPriceEntry], int]:
    """Validasi tiap elemen list via Pydantic. Kembalikan (valid_entries, error_count)."""
    valid:  List[MarketPriceEntry] = []
    errors: int = 0

    for i, item in enumerate(raw_list):
        if not isinstance(item, dict):
            logger.warning(f"[LLM] Entry #{i} bukan dict, skip.")
            errors += 1
            continue
        try:
            valid.append(MarketPriceEntry.model_validate(item))
        except Exception as e:
            logger.warning(
                f"[LLM] Entry #{i} dari '{source_name}' gagal validasi: {e}. "
                f"Data: {str(item)[:200]}"
            )
            errors += 1

    return valid, errors


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

async def analyze_page(
    page:   ScrapedPage,
    config: Optional[OllamaConfig] = None,
) -> Tuple[List[MarketPriceEntry], int]:
    """Analisis satu ScrapedPage. Kembalikan (entries, parse_error_count)."""
    if config is None:
        config = OLLAMA_CONFIG

    if not page.success or not page.raw_text.strip():
        logger.info(f"[LLM] Skip '{page.source_name}' — halaman gagal di-scrape.")
        return [], 0

    raw_text = page.raw_text[: config.max_input_chars]
    if len(page.raw_text) > config.max_input_chars:
        logger.info(
            f"[LLM] Teks '{page.source_name}' truncated: "
            f"{len(page.raw_text)} → {config.max_input_chars} chars."
        )

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user",   "content": _USER_PROMPT_TEMPLATE.format(
            source_name = page.source_name,
            source_url  = page.source_url,
            raw_text    = raw_text,
        )},
    ]

    parse_errors = 0

    for attempt in range(1, config.max_parse_retries + 2):
        try:
            logger.info(
                f"[LLM] Memanggil Ollama: '{page.source_name}' "
                f"(model={config.model}, attempt={attempt})"
            )
            llm_output = await _call_ollama(messages, config)
            raw_list   = _extract_json_array(llm_output)

            if raw_list is None:
                logger.warning(f"[LLM] Output non-JSON (attempt={attempt}): {llm_output[:300]}")
                parse_errors += 1
                if attempt <= config.max_parse_retries:
                    messages += [
                        {"role": "assistant", "content": llm_output},
                        {"role": "user", "content": _REPAIR_PROMPT.format(
                            previous_output=llm_output[:500]
                        )},
                    ]
                    continue
                logger.error(f"[LLM] Semua attempt habis untuk '{page.source_name}'.")
                return [], parse_errors

            if not raw_list:
                logger.info(f"[LLM] Tidak ada entry harga di '{page.source_name}'.")
                return [], parse_errors

            entries, val_errors = _validate_entries(raw_list, page.source_name)
            parse_errors += val_errors
            logger.info(
                f"[LLM] '{page.source_name}': {len(entries)} entry valid, "
                f"{val_errors} validasi error."
            )
            return entries, parse_errors

        except httpx.ConnectError:
            logger.error(
                f"[LLM] Tidak bisa terhubung ke Ollama di {config.base_url}. "
                "Jalankan: `ollama serve`"
            )
            return [], parse_errors + 1
        except httpx.TimeoutException:
            logger.error(f"[LLM] Timeout ({config.timeout_sec}s) untuk '{page.source_name}'.")
            return [], parse_errors + 1
        except httpx.HTTPStatusError as e:
            logger.error(f"[LLM] HTTP error dari Ollama: {e.response.status_code}")
            return [], parse_errors + 1
        except Exception as e:
            logger.error(f"[LLM] Error tak terduga: {e}", exc_info=True)
            return [], parse_errors + 1

    return [], parse_errors


async def analyze_pages(
    pages:  List[ScrapedPage],
    config: Optional[OllamaConfig] = None,
) -> Tuple[List[MarketPriceEntry], int]:
    """Analisis semua halaman secara sequential."""
    all_entries:  List[MarketPriceEntry] = []
    total_errors: int = 0

    for page in pages:
        entries, errors = await analyze_page(page, config)
        all_entries.extend(entries)
        total_errors += errors

    logger.info(
        f"[LLM] Selesai: {len(all_entries)} total entry | {total_errors} total parse error."
    )
    return all_entries, total_errors