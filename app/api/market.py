# app/api/market.py

from __future__ import annotations

import asyncio
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.core_dependencies import AuthResult, require_scope, verify_api_key
from core.config import get_display_name
from core.logger import get_logger
from core.security import KeyScope
from agents.market_intelligence.store import get_market_store
from agents.market_intelligence.task import run_once
from schemas.response import (
    MarketPricesResponse,
    MarketReportResponse,
    VarietyPriceSummary,
)

logger = get_logger("api.market")

router = APIRouter(prefix="/market", tags=["Market Intelligence"])

_VALID_VARIETY_CODES = frozenset({"D13", "D197", "D2", "D200", "D24"})


@router.get(
    "/prices",
    response_model = MarketPricesResponse,
    summary        = "Harga Pasar Durian Premium",
    description    = (
        "Ringkasan harga terkini (IDR/kg) untuk semua varietas durian premium "
        "berdasarkan data dari marketplace Indonesia (Tokopedia & Shopee).\n\n"
        "Data diperbarui otomatis setiap hari oleh Market Intelligence Agent.\n\n"
        "Field `data_fresh=false` berarti data lebih dari 25 jam — "
        "tampilkan indikator 'data mungkin tidak terkini' di UI.\n\n"
        "**Memerlukan API key.**"
    ),
)
async def get_market_prices(
    variety_code: Optional[str] = Query(
        default=None,
        description=(
            "Filter by variety code. "
            "Nilai valid: D13, D197, D2, D200, D24. "
            "Kosong = semua varietas."
        ),
        examples=["D197"],
        max_length=10,
    ),
    auth: AuthResult = Depends(verify_api_key),
) -> MarketPricesResponse:

    # Validasi variety_code sebelum query ke store
    if variety_code is not None:
        code_upper = variety_code.upper().strip()
        if code_upper not in _VALID_VARIETY_CODES:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    f"variety_code '{variety_code}' tidak valid. "
                    f"Kode yang didukung: {', '.join(sorted(_VALID_VARIETY_CODES))}."
                ),
            )
    else:
        code_upper = None

    store = get_market_store()

    has_data = await store.has_data()
    if not has_data:
        logger.info(
            f"[Market API] GET /prices — belum ada data | key={auth.key_prefix}"
        )
        return MarketPricesResponse(
            success       = True,
            data_fresh    = False,
            scraped_at    = None,
            prices        = [],
            variety_count = 0,
        )

    is_stale      = await store.is_stale()
    price_summary = await store.get_price_summary()
    latest_report = await store.get_latest_report()

    # Filter per varietas jika diminta
    if code_upper is not None:
        price_summary = {k: v for k, v in price_summary.items() if k == code_upper}
        if not price_summary:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=(
                    f"Tidak ada data harga untuk variety_code='{variety_code}'. "
                    "Data mungkin belum tersedia untuk varietas ini."
                ),
            )

    prices = [
        VarietyPriceSummary(
            variety_code  = code,
            variety_name  = get_display_name(code),
            price_min_idr = summary.price_min_idr,
            price_max_idr = summary.price_max_idr,
            price_avg_idr = summary.price_avg_idr,
            sample_count  = summary.sample_count,
            scraped_at    = summary.scraped_at,
        )
        for code, summary in sorted(price_summary.items())
    ]

    logger.info(
        f"[Market API] GET /prices: "
        f"{len(prices)} varietas | "
        f"stale={is_stale} | "
        f"filter={variety_code or 'all'} | "
        f"key={auth.key_prefix}"
    )

    return MarketPricesResponse(
        success       = True,
        data_fresh    = not is_stale,
        scraped_at    = latest_report.run_ended_at if latest_report else None,
        prices        = prices,
        variety_count = len(prices),
    )


@router.get(
    "/report",
    response_model = MarketReportResponse,
    summary        = "Full Market Intelligence Report",
    description    = (
        "Report lengkap dari run agent terakhir, termasuk metadata scraping dan LLM.\n\n"
        "Gunakan `?include_entries=true` untuk menyertakan seluruh daftar entry harga "
        "(berguna untuk NestJS upsert ke database).\n\n"
        "**Memerlukan API key.**"
    ),
)
async def get_market_report(
    include_entries: bool = Query(
        default=False,
        description=(
            "Sertakan seluruh daftar MarketPriceEntry di response body. "
            "Hanya aktifkan jika diperlukan — dapat menghasilkan response besar."
        ),
    ),
    auth: AuthResult = Depends(verify_api_key),
) -> MarketReportResponse:
    store         = get_market_store()
    latest_report = await store.get_latest_report()
    is_stale      = await store.is_stale()

    if latest_report is None:
        logger.info(
            f"[Market API] GET /report — belum ada data | key={auth.key_prefix}"
        )
        return MarketReportResponse(
            success    = True,
            data_fresh = False,
        )

    entries_payload = None
    if include_entries:
        entries_payload = [
            e.model_dump(mode="json") for e in latest_report.entries
        ]

    logger.info(
        f"[Market API] GET /report: "
        f"run_id={latest_report.run_id} | "
        f"entries={latest_report.entry_count} | "
        f"include_entries={include_entries} | "
        f"key={auth.key_prefix}"
    )

    return MarketReportResponse(
        success           = True,
        run_id            = latest_report.run_id,
        agent_version     = latest_report.agent_version,
        status            = latest_report.status.value,
        run_started_at    = latest_report.run_started_at,
        run_ended_at      = latest_report.run_ended_at,
        sources_scraped   = latest_report.sources_scraped,
        sources_failed    = latest_report.sources_failed,
        entry_count       = latest_report.entry_count,
        entries_discarded = latest_report.entries_discarded,
        llm_parse_errors  = latest_report.llm_parse_errors,
        data_fresh        = not is_stale,
        error_details     = latest_report.error_details,
        entries           = entries_payload,
    )


@router.post(
    "/trigger-run",
    summary     = "Trigger Manual Run Agent (Admin)",
    description = (
        "Jalankan Market Intelligence Agent sekarang tanpa menunggu jadwal cron.\n\n"
        "Berguna untuk development, debugging, atau refresh data manual.\n\n"
        "Request langsung return `202 Accepted` — agent berjalan di background.\n\n"
        "**Memerlukan scope `admin`.**"
    ),
    status_code = status.HTTP_202_ACCEPTED,
    tags        = ["Market Intelligence", "Admin"],
)
async def trigger_market_run(
    auth: AuthResult = Depends(require_scope(KeyScope.ADMIN)),
):
    logger.info(
        f"[Market API] Manual trigger oleh key={auth.key_prefix}"
    )

    # Fire-and-forget — tidak tunggu selesai
    asyncio.create_task(run_once(), name="manual_market_run_api")

    return {
        "success": True,
        "message": (
            "Market Intelligence Agent dimulai di background. "
            "Data akan tersedia di GET /api/v1/market/prices setelah run selesai."
        ),
        "triggered_by": auth.key_prefix,
    }


@router.get(
    "/diagnostics",
    summary     = "Diagnostik Market Data Store (Admin)",
    description = (
        "Kembalikan info diagnostik internal store: usia data, entry count, status CB, dll.\n\n"
        "**Memerlukan scope `admin`.**"
    ),
    tags        = ["Market Intelligence", "Admin"],
)
async def get_market_diagnostics(
    auth: AuthResult = Depends(require_scope(KeyScope.ADMIN)),
):
    store = get_market_store()
    diagnostics = await store.get_diagnostics()

    logger.info(
        f"[Market API] GET /diagnostics oleh key={auth.key_prefix}"
    )

    return {
        "success":     True,
        "diagnostics": diagnostics,
    }