# agents/market_intelligence/llm_analyzer.py
"""
LLM Analyzer — mengekstrak data harga dari teks mentah menggunakan Ollama.

Strategi:
  - Komunikasi via httpx langsung ke Ollama REST API (tidak pakai LangChain
    untuk menghindari dependency berat; pattern mudah diupgrade ke LangChain)
  - Prompt engineering spesifik untuk D197 (Musang King) dan D200 (Duri Hitam)
  - Output terstruktur: LLM diminta kembalikan JSON saja
  - Retry dengan perbaikan prompt jika output bukan JSON valid
  - Truncate input agar tidak melebihi context window Ollama
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import List, Optional, Tuple

import httpx

from core.logger import get_logger
from agents.market_intelligence.config import OLLAMA_CONFIG, OllamaConfig
from agents.market_intelligence.schemas import (
    DurianVariety,
    MarketPriceEntry,
    ScrapedPage,
)

logger = get_logger("agent.llm_analyzer")


# ---------------------------------------------------------------------------
# Prompt Templates
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """Kamu adalah analis harga komoditas pertanian yang sangat teliti dan terstruktur.
Tugasmu adalah mengekstrak informasi harga durian premium dari teks marketplace/forum Indonesia.

PENGETAHUAN DOMAIN — WAJIB DIINGAT:
- "Musang King", "Raja Kunyit", "Mao Shan Wang", "MSW", "MK", "D197" → variety_code = "D197"
- "Duri Hitam", "Black Thorn", "Ochee", "Ochee Durian", "D200" → variety_code = "D200"
- ABAIKAN varietas lain seperti D13, D24, D2, monthong, petruk, bawor, dll.
- Harga SELALU dalam satuan IDR (Rupiah Indonesia). Jika tertulis "Rp", "ribu", "rb", "jt" → konversi ke angka penuh.
  Contoh: "350rb" → 350000, "1.2jt" → 1200000, "Rp 450.000" → 450000

FORMAT OUTPUT:
Kembalikan HANYA array JSON valid tanpa komentar, markdown, atau teks tambahan apapun.
Jika tidak menemukan data harga yang relevan, kembalikan array kosong: []

SCHEMA SETIAP ELEMEN:
{
  "variety_code": "D197" | "D200",
  "variety_alias": "<nama yang ditemukan di teks>",
  "price_per_kg_min": <angka float atau null>,
  "price_per_kg_max": <angka float atau null>,
  "price_per_kg_avg": <angka float atau null>,
  "price_per_unit_min": <angka float atau null>,
  "price_per_unit_max": <angka float atau null>,
  "location_hint": "<kota/region atau 'online' atau null>",
  "seller_type": "<'kebun'|'reseller'|'importir'|'toko online'|null>",
  "confidence": <0.0 hingga 1.0>,
  "raw_text_snippet": "<kutipan teks sumber maks 200 karakter>"
}"""

_USER_PROMPT_TEMPLATE = """Berikut adalah teks dari {source_name} ({source_url}).
Ekstrak semua data harga durian Musang King (D197) dan Duri Hitam (D200) dari teks ini.

--- TEKS SUMBER ---
{raw_text}
--- AKHIR TEKS ---

Ingat: kembalikan HANYA array JSON. Tidak ada teks sebelum atau sesudah JSON."""

_REPAIR_PROMPT = """Output sebelumnya bukan JSON valid. Perbaiki dan kembalikan HANYA array JSON yang valid.
Jika tidak bisa diperbaiki, kembalikan [].

Output sebelumnya:
{previous_output}"""


# ---------------------------------------------------------------------------
# Ollama HTTP client (fungsi internal)
# ---------------------------------------------------------------------------

async def _call_ollama(
    prompt_messages: List[dict],
    config: OllamaConfig,
) -> str:
    """
    Panggil Ollama /api/chat endpoint.
    Kembalikan string konten response model.
    """
    url = f"{config.base_url}/api/chat"
    payload = {
        "model":   config.model,
        "messages": prompt_messages,
        "stream":   False,
        "options": {
            "temperature": config.temperature,
            "top_p":       config.top_p,
            "num_predict": 4096,
        },
    }

    async with httpx.AsyncClient(timeout=config.timeout_sec) as client:
        response = await client.post(url, json=payload)
        response.raise_for_status()
        data = response.json()

    # Ollama /api/chat response structure
    content: str = data.get("message", {}).get("content", "")
    if not content:
        raise ValueError(f"Ollama mengembalikan respons kosong. Raw: {data}")

    return content


# ---------------------------------------------------------------------------
# JSON parsing dengan cleanup
# ---------------------------------------------------------------------------

def _extract_json_array(text: str) -> Optional[list]:
    """
    Coba parse JSON dari output LLM.
    LLM kadang membungkus JSON dalam markdown code fence — tangani itu.
    """
    # Bersihkan markdown fence jika ada
    cleaned = re.sub(r"```(?:json)?\s*", "", text).strip()
    cleaned = re.sub(r"```\s*$", "", cleaned).strip()

    # Coba parse langsung
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict):
            # Kadang LLM wrap dalam {"data": [...]}
            for key in ("data", "results", "entries", "prices"):
                if isinstance(parsed.get(key), list):
                    return parsed[key]
    except json.JSONDecodeError:
        pass

    # Coba extract array dengan regex sebagai last resort
    array_match = re.search(r"\[.*\]", cleaned, re.DOTALL)
    if array_match:
        try:
            return json.loads(array_match.group())
        except json.JSONDecodeError:
            pass

    return None


# ---------------------------------------------------------------------------
# Validasi dan konversi ke Pydantic models
# ---------------------------------------------------------------------------

def _parse_llm_entries(raw_list: list, source_name: str) -> Tuple[List[MarketPriceEntry], int]:
    """
    Validasi setiap elemen dari output LLM menggunakan Pydantic.
    Kembalikan (valid_entries, error_count).
    """
    valid: List[MarketPriceEntry] = []
    errors = 0

    for i, item in enumerate(raw_list):
        if not isinstance(item, dict):
            logger.warning(f"[LLM] Entry #{i} bukan dict, skip.")
            errors += 1
            continue
        try:
            entry = MarketPriceEntry.model_validate(item)
            valid.append(entry)
        except Exception as e:
            logger.warning(
                f"[LLM] Entry #{i} dari '{source_name}' gagal validasi: {e}. "
                f"Data: {str(item)[:200]}"
            )
            errors += 1

    return valid, errors


# ---------------------------------------------------------------------------
# Public async interface
# ---------------------------------------------------------------------------

async def analyze_page(
    page: ScrapedPage,
    config: Optional[OllamaConfig] = None,
) -> Tuple[List[MarketPriceEntry], int]:
    """
    Analisis satu ScrapedPage dan kembalikan (entries, parse_error_count).

    Menerapkan retry dengan prompt perbaikan jika output LLM bukan JSON valid.
    """
    if config is None:
        config = OLLAMA_CONFIG

    if not page.success or not page.raw_text.strip():
        logger.info(f"[LLM] Skip '{page.source_name}' — halaman gagal di-scrape.")
        return [], 0

    # Truncate teks agar tidak melebihi context window
    raw_text = page.raw_text[: config.max_input_chars]
    if len(page.raw_text) > config.max_input_chars:
        logger.info(
            f"[LLM] Teks '{page.source_name}' truncated dari "
            f"{len(page.raw_text)} → {config.max_input_chars} chars."
        )

    user_message = _USER_PROMPT_TEMPLATE.format(
        source_name=page.source_name,
        source_url=page.source_url,
        raw_text=raw_text,
    )

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user",   "content": user_message},
    ]

    last_output: str = ""
    parse_errors: int = 0

    for attempt in range(1, config.max_parse_retries + 2):
        try:
            logger.info(
                f"[LLM] Memanggil Ollama untuk '{page.source_name}' "
                f"(model={config.model}, attempt={attempt})..."
            )
            llm_output = await _call_ollama(messages, config)
            last_output = llm_output

            raw_list = _extract_json_array(llm_output)
            if raw_list is None:
                logger.warning(
                    f"[LLM] Output bukan JSON valid (attempt={attempt}). "
                    f"Preview: {llm_output[:300]}"
                )
                parse_errors += 1
                if attempt <= config.max_parse_retries:
                    # Kirim repair prompt
                    messages.append({"role": "assistant", "content": llm_output})
                    messages.append({
                        "role":    "user",
                        "content": _REPAIR_PROMPT.format(previous_output=llm_output[:500]),
                    })
                    continue
                else:
                    logger.error(
                        f"[LLM] Semua attempt habis untuk '{page.source_name}'. "
                        "Tidak ada data yang bisa di-parse."
                    )
                    return [], parse_errors

            if not raw_list:
                logger.info(f"[LLM] Tidak ada entry harga relevan di '{page.source_name}'.")
                return [], parse_errors

            entries, validation_errors = _parse_llm_entries(raw_list, page.source_name)
            parse_errors += validation_errors

            logger.info(
                f"[LLM] '{page.source_name}': {len(entries)} entry valid "
                f"| {validation_errors} validasi error."
            )
            return entries, parse_errors

        except httpx.ConnectError:
            logger.error(
                f"[LLM] Tidak bisa terhubung ke Ollama di {config.base_url}. "
                "Pastikan Ollama daemon berjalan: `ollama serve`"
            )
            return [], parse_errors + 1
        except httpx.TimeoutException:
            logger.error(
                f"[LLM] Timeout ({config.timeout_sec}s) saat memanggil Ollama "
                f"untuk '{page.source_name}'."
            )
            return [], parse_errors + 1
        except httpx.HTTPStatusError as e:
            logger.error(
                f"[LLM] HTTP error dari Ollama: {e.response.status_code} "
                f"— {e.response.text[:200]}"
            )
            return [], parse_errors + 1
        except Exception as e:
            logger.error(
                f"[LLM] Error tidak terduga untuk '{page.source_name}': {e}",
                exc_info=True,
            )
            return [], parse_errors + 1

    return [], parse_errors


async def analyze_pages(
    pages: List[ScrapedPage],
    config: Optional[OllamaConfig] = None,
) -> Tuple[List[MarketPriceEntry], int]:
    """
    Analisis semua halaman secara sequential.
    Kembalikan (all_entries, total_parse_errors).
    """
    all_entries: List[MarketPriceEntry] = []
    total_errors: int = 0

    for page in pages:
        entries, errors = await analyze_page(page, config)
        all_entries.extend(entries)
        total_errors += errors

    logger.info(
        f"[LLM] Analisis selesai: {len(all_entries)} total entry "
        f"| {total_errors} total parse error."
    )
    return all_entries, total_errors