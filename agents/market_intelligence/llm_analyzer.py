# agents/market_intelligence/llm_analyzer.py

from __future__ import annotations

import asyncio
import json
import re
from typing import List, Optional, Tuple

import httpx

from core.logger import get_logger
from agents.market_intelligence.config import OLLAMA_CONFIG, OllamaConfig
from agents.market_intelligence.schemas import MarketPriceEntry, ScrapedPage

logger = get_logger("agent.llm_analyzer")

_SYSTEM_PROMPT = """\
Kamu adalah analis data harga komoditas pertanian yang SANGAT KETAT dan TERSTRUKTUR.
Tugasmu adalah mengekstrak data harga durian premium dari JSON intercept API e-commerce Indonesia.

====================================================================
BAGIAN 1 — PENGETAHUAN DOMAIN (WAJIB HAFAL)
====================================================================

PEMETAAN VARIETAS (5 varietas resmi yang dipantau):
- "Musang King", "Raja Kunyit", "Mao Shan Wang", "MSW", "MK", "D197" → variety_code = "D197"
- "Duri Hitam", "Black Thorn", "Ochee", "D200"                       → variety_code = "D200"
- "Golden Bun", "D13"                                                 → variety_code = "D13"
- "Sultan", "Bukit Merah", "D24"                                      → variety_code = "D24"
- "Dato Nina", "D2"                                                   → variety_code = "D2"

Semua kode harga dalam IDR. Konversi: "350rb"→350000, "1.2jt"→1200000, "Rp 450.000"→450000.

====================================================================
BAGIAN 2 — ATURAN REJECT MUTLAK (WAJIB DIPATUHI TANPA PENGECUALIAN)
====================================================================

ABAIKAN SEPENUHNYA (is_whole_fruit = false, atau skip dari output) produk yang
nama/deskripsinya mengandung satu saja dari kata kunci berikut (case-insensitive):

  DAFTAR KATA KUNCI REJECT:
  ┌─────────────────────────────────────────────────────────┐
  │ KUPAS    │ DIKUPAS  │ FLESH    │ PULP     │ DAGING     │
  │ FROZEN   │ BEKU     │ FREEZER  │ COLD     │ ES         │
  │ BOX      │ KOTAK    │ PACK     │ VAKUM    │ VACUUM     │
  │ 100GR    │ 200GR    │ 250GR    │ 400GR    │ 500GR      │
  │ OLAHAN   │ PANCAKE  │ BISKUIT  │ KUE      │ CAKE       │
  │ BIBIT    │ BENIH    │ SEEDLING │ TANAM    │ POHON      │
  │ EXTRACT  │ SARI     │ MINUMAN  │ JELLY    │ PUDDING    │
  └─────────────────────────────────────────────────────────┘

  WAJIB DIPROSES: Hanya produk "Durian Utuh", "Durian Montong", "Durian Berkulit",
  "Durian Segar", atau produk yang secara eksplisit menyebutkan berat per-buah
  (misal "per biji 2-3 kg") TANPA kata kunci reject di atas.

====================================================================
BAGIAN 3 — ATURAN NORMALISASI HARGA (WAJIB CoT DI FIELD "notes")
====================================================================

Semua harga di output WAJIB dalam satuan IDR per-1-Kg.

CONTOH CHAIN OF THOUGHT (CoT) yang wajib kamu tiru di field "notes":

CONTOH A — Harga per buah:
  Input JSON: {"name": "Durian Musang King Utuh", "price": 800000, "weight": "2 kg/buah"}
  Kalkulasi: 800000 / 2 = 400000 per kg
  Output:
    price_per_kg_avg: 400000
    weight_reference: "per buah 2 kg"
    notes: "Harga listing Rp800.000/buah dibagi estimasi berat 2kg → 800000÷2=400000/kg"

CONTOH B — Harga sudah per-kg:
  Input JSON: {"name": "Durian Black Thorn Segar", "price": 550000, "unit": "per kg"}
  Kalkulasi: sudah per-kg, tidak perlu konversi
  Output:
    price_per_kg_avg: 550000
    weight_reference: "per kg"
    notes: null

CONTOH C — Range harga:
  Input JSON: {"name": "Durian Golden Bun Whole", "price_min": 300000, "price_max": 400000,
               "weight_desc": "per buah 1.5-2.5 kg"}
  Kalkulasi: min → 300000/2.5=120000/kg, max → 400000/1.5=266667/kg
  Output:
    price_per_kg_min: 120000
    price_per_kg_max: 266667
    weight_reference: "per buah 1.5-2.5 kg"
    notes: "Range harga Rp300k-400k per buah, dibagi range berat 2.5-1.5kg → 120000-266667/kg"

CONTOH D — REJECT (kupas/frozen):
  Input JSON: {"name": "Durian Musang King KUPAS FROZEN 400gr", "price": 150000}
  → Abaikan sepenuhnya. JANGAN masukkan ke output array.

====================================================================
BAGIAN 4 — FORMAT OUTPUT (WAJIB DIIKUTI TEPAT)
====================================================================

Kembalikan HANYA array JSON valid tanpa markdown, komentar, atau teks lain.
Jika tidak ada data yang lolos filter, kembalikan: []

SCHEMA TIAP ELEMEN (SEMUA FIELD WAJIB ADA):
{
  "variety_code":       "D13" | "D197" | "D2" | "D200" | "D24",
  "variety_alias":      "<nama produk di JSON sumber, maks 100 karakter>",
  "is_whole_fruit":     true,
  "weight_reference":   "<berat asli dari listing, misal 'per buah 2kg' atau 'per kg'>",
  "notes":              "<CoT kalkulasi normalisasi, atau null jika sudah per-kg>",
  "price_per_kg_min":   <float|null>,
  "price_per_kg_max":   <float|null>,
  "price_per_kg_avg":   <float|null>,
  "price_per_unit_min": <float|null>,
  "price_per_unit_max": <float|null>,
  "location_hint":      "<kota/region/'online'|null>",
  "seller_type":        "<'kebun'|'reseller'|'importir'|'toko online'|null>",
  "confidence":         <0.0–1.0>,
  "raw_text_snippet":   "<field name+price dari JSON sumber, maks 200 karakter>"
}

PERHATIAN TERAKHIR:
- is_whole_fruit SELALU true pada output array (yang false langsung dibuang/skip).
- Jika ragu apakah produk utuh atau tidak → buang (conservative reject).
- Jangan menciptakan data. Jika harga tidak ada di JSON → skip produk tersebut."""

_USER_PROMPT_TEMPLATE = """\
Berikut adalah JSON intercept dari API e-commerce {source_name} ({source_url}).
JSON ini berisi daftar produk dari hasil pencarian "durian premium".

=== JSON INTERCEPT ===
{raw_json}
=== AKHIR JSON ===

Terapkan semua aturan di system prompt. Kembalikan HANYA array JSON.
Tidak ada teks, markdown, atau penjelasan sebelum maupun sesudah array JSON."""

_REPAIR_PROMPT = """\
Output sebelumnya bukan JSON valid atau bukan array. Perbaiki dan kembalikan \
HANYA array JSON yang valid sesuai schema.
Jika tidak bisa diperbaiki atau tidak ada data valid, kembalikan: []

Output sebelumnya (untuk referensi perbaikan):
{previous_output}"""

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


def _extract_json_array(text: str) -> Optional[list]:
    cleaned = re.sub(r"```(?:json)?\s*", "", text).strip()
    cleaned = re.sub(r"```\s*$", "", cleaned).strip()

    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict):
            for key in ("data", "results", "entries", "prices", "products"):
                if isinstance(parsed.get(key), list):
                    logger.debug(f"[LLM] JSON di-unwrap dari key '{key}'.")
                    return parsed[key]
    except json.JSONDecodeError:
        pass

    m = re.search(r"\[.*\]", cleaned, re.DOTALL)
    if m:
        try:
            candidate = json.loads(m.group())
            if isinstance(candidate, list):
                logger.debug("[LLM] JSON array di-extract via regex fallback.")
                return candidate
        except json.JSONDecodeError:
            pass

    return None

def _validate_and_filter_entries(
    raw_list:    list,
    source_name: str,
) -> Tuple[List[MarketPriceEntry], int, int]:
    valid_entries:  List[MarketPriceEntry] = []
    pydantic_errors: int = 0
    non_whole_count: int = 0

    for i, item in enumerate(raw_list):
        if not isinstance(item, dict):
            logger.warning(
                f"[LLM] Entry #{i} dari '{source_name}' bukan dict "
                f"(type={type(item).__name__}), skip."
            )
            pydantic_errors += 1
            continue

        try:
            entry = MarketPriceEntry.model_validate(item)

            if not entry.is_whole_fruit:
                logger.warning(
                    f"[LLM] Entry #{i} '{source_name}' ditandai is_whole_fruit=False "
                    f"(alias='{entry.variety_alias}'). "
                    "Entry DIBUANG. [DATA LEAKAGE PREVENTED - LLM LAYER]"
                )
                non_whole_count += 1
                continue

            valid_entries.append(entry)

        except Exception as exc:
            logger.warning(
                f"[LLM] Entry #{i} dari '{source_name}' gagal validasi Pydantic: {exc}. "
                f"Preview: {str(item)[:200]}"
            )
            pydantic_errors += 1

    return valid_entries, pydantic_errors, non_whole_count

async def analyze_page(
    page:   ScrapedPage,
    config: Optional[OllamaConfig] = None,
) -> Tuple[List[MarketPriceEntry], int, int]:

    if config is None:
        config = OLLAMA_CONFIG

    if not page.success or not page.raw_json.strip():
        logger.info(f"[LLM] Skip '{page.source_name}' — halaman gagal di-intercept.")
        return [], 0, 0

    raw_json = page.raw_json[: config.max_input_chars]
    if len(page.raw_json) > config.max_input_chars:
        logger.info(
            f"[LLM] JSON '{page.source_name}' truncated: "
            f"{len(page.raw_json)} → {config.max_input_chars} chars."
        )

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {
            "role": "user",
            "content": _USER_PROMPT_TEMPLATE.format(
                source_name=page.source_name,
                source_url=page.source_url,
                raw_json=raw_json,
            ),
        },
    ]

    parse_errors    = 0
    total_discarded = 0

    for attempt in range(1, config.max_parse_retries + 2):
        try:
            logger.info(
                f"[LLM] Memanggil Ollama: '{page.source_name}' "
                f"(model={config.model}, attempt={attempt})"
            )
            llm_output = await _call_ollama(messages, config)
            raw_list   = _extract_json_array(llm_output)

            if raw_list is None:
                logger.warning(
                    f"[LLM] Output non-JSON dari '{page.source_name}' "
                    f"(attempt={attempt}): {llm_output[:300]}"
                )
                parse_errors += 1
                if attempt <= config.max_parse_retries:
                    messages += [
                        {"role": "assistant", "content": llm_output},
                        {
                            "role": "user",
                            "content": _REPAIR_PROMPT.format(
                                previous_output=llm_output[:500]
                            ),
                        },
                    ]
                    continue
                logger.error(
                    f"[LLM] Semua {config.max_parse_retries + 1} attempt habis "
                    f"untuk '{page.source_name}'."
                )
                return [], parse_errors, total_discarded

            if not raw_list:
                logger.info(
                    f"[LLM] Tidak ada entry harga yang lolos filter "
                    f"di '{page.source_name}' (LLM mengembalikan [])."
                )
                return [], parse_errors, total_discarded

            entries, val_errors, discarded = _validate_and_filter_entries(
                raw_list, page.source_name
            )
            parse_errors    += val_errors
            total_discarded += discarded

            logger.info(
                f"[LLM] '{page.source_name}': "
                f"{len(entries)} entry valid | "
                f"{discarded} entry buang (non-whole) | "
                f"{val_errors} error validasi schema."
            )
            return entries, parse_errors, total_discarded

        except httpx.ConnectError:
            logger.error(
                f"[LLM] Tidak bisa terhubung ke Ollama di {config.base_url}. "
                "Pastikan Ollama berjalan: `ollama serve`"
            )
            return [], parse_errors + 1, total_discarded
        except httpx.TimeoutException:
            logger.error(
                f"[LLM] Timeout ({config.timeout_sec}s) saat analisis '{page.source_name}'."
            )
            return [], parse_errors + 1, total_discarded
        except httpx.HTTPStatusError as exc:
            logger.error(
                f"[LLM] HTTP error dari Ollama: {exc.response.status_code} "
                f"untuk '{page.source_name}'."
            )
            return [], parse_errors + 1, total_discarded
        except Exception as exc:
            logger.error(
                f"[LLM] Error tak terduga saat analisis '{page.source_name}': {exc}",
                exc_info=True,
            )
            return [], parse_errors + 1, total_discarded

    return [], parse_errors, total_discarded


async def analyze_pages(
    pages:  List[ScrapedPage],
    config: Optional[OllamaConfig] = None,
) -> Tuple[List[MarketPriceEntry], int, int]:

    if not pages:
        return [], 0, 0

    tasks = [analyze_page(page, config) for page in pages]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    all_entries:     List[MarketPriceEntry] = []
    total_errors:    int = 0
    total_discarded: int = 0

    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logger.error(
                f"[LLM] Exception dari gather task #{i} "
                f"('{pages[i].source_name}'): {result}",
                exc_info=result,
            )
            total_errors += 1
            continue
        entries, errors, discarded = result
        all_entries.extend(entries)
        total_errors    += errors
        total_discarded += discarded

    logger.info(
        f"[LLM] Analisis selesai (paralel, {len(pages)} page): "
        f"{len(all_entries)} entry valid | "
        f"{total_discarded} entry dibuang | "
        f"{total_errors} parse error."
    )
    return all_entries, total_errors, total_discarded
