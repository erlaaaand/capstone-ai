# agents/market_intelligence/store.py

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Dict, List, Optional

from agents.market_intelligence.schemas import (
    AgentRunStatus,
    DurianVariety,
    MarketPriceEntry,
    MarketReportPayload,
)
from core.logger import get_logger

logger = get_logger("agent.store")


class MarketDataStore:
    """
    Singleton in-memory store untuk data pasar durian terbaru.
    Diakses oleh agent (write) dan API endpoint (read).
    """

    _instance: Optional["MarketDataStore"] = None
    _lock: asyncio.Lock

    def __new__(cls) -> "MarketDataStore":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._data_lock   = asyncio.Lock()
            cls._instance._latest:     Optional[MarketReportPayload] = None
            cls._instance._by_variety: Dict[str, List[MarketPriceEntry]] = {}
        return cls._instance

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    async def save(self, payload: MarketReportPayload) -> None:
        """Simpan hasil run terbaru dan bangun index per-varietas."""
        async with self._data_lock:
            self._latest = payload

            # Rebuild variety index
            by_variety: Dict[str, List[MarketPriceEntry]] = {}
            for entry in payload.entries:
                code = entry.variety_code.value
                by_variety.setdefault(code, []).append(entry)
            self._by_variety = by_variety

        logger.info(
            f"[Store] Data pasar tersimpan: "
            f"run_id={payload.run_id} | "
            f"entries={payload.entry_count} | "
            f"varieties={list(by_variety.keys())}"
        )

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    async def get_latest_report(self) -> Optional[MarketReportPayload]:
        """Kembalikan payload run terbaru, atau None jika belum ada data."""
        async with self._data_lock:
            return self._latest

    async def get_entries_by_variety(
        self,
        variety_code: str,
    ) -> List[MarketPriceEntry]:
        """Kembalikan semua entry untuk satu varietas tertentu."""
        async with self._data_lock:
            return list(self._by_variety.get(variety_code.upper(), []))

    async def get_price_summary(self) -> Dict[str, "VarietySummary"]:
        """
        Kembalikan ringkasan harga min/max/avg per varietas.
        Berguna untuk endpoint GET /market/prices — tidak perlu kirim
        seluruh payload besar ke Flutter.
        """
        async with self._data_lock:
            summary: Dict[str, VarietySummary] = {}
            for code, entries in self._by_variety.items():
                kg_prices = [
                    e.price_per_kg_avg
                    for e in entries
                    if e.price_per_kg_avg is not None
                ]
                if not kg_prices:
                    # Fallback ke mid-range jika hanya ada min/max
                    kg_prices = []
                    for e in entries:
                        if e.price_per_kg_min is not None and e.price_per_kg_max is not None:
                            kg_prices.append((e.price_per_kg_min + e.price_per_kg_max) / 2)
                        elif e.price_per_kg_min is not None:
                            kg_prices.append(e.price_per_kg_min)
                        elif e.price_per_kg_max is not None:
                            kg_prices.append(e.price_per_kg_max)

                if kg_prices:
                    summary[code] = VarietySummary(
                        variety_code  = code,
                        price_min_idr = round(min(kg_prices)),
                        price_max_idr = round(max(kg_prices)),
                        price_avg_idr = round(sum(kg_prices) / len(kg_prices)),
                        sample_count  = len(entries),
                        scraped_at    = (
                            self._latest.run_ended_at if self._latest else datetime.now(timezone.utc)
                        ),
                    )
            return summary

    async def is_stale(self, max_age_hours: int = 25) -> bool:
        """
        True jika data lebih tua dari max_age_hours atau belum ada.
        Digunakan oleh endpoint untuk memberi tahu client tentang freshness.
        """
        async with self._data_lock:
            if self._latest is None:
                return True
            age = (datetime.now(timezone.utc) - self._latest.run_ended_at).total_seconds()
            return age > (max_age_hours * 3600)

    async def has_data(self) -> bool:
        async with self._data_lock:
            return self._latest is not None and bool(self._latest.entries)


# ---------------------------------------------------------------------------
# Value Object: Ringkasan harga per varietas
# ---------------------------------------------------------------------------

from dataclasses import dataclass

@dataclass
class VarietySummary:
    variety_code:  str
    price_min_idr: int      # IDR per kg
    price_max_idr: int      # IDR per kg
    price_avg_idr: int      # IDR per kg
    sample_count:  int      # Jumlah listing yang dijadikan basis
    scraped_at:    datetime # Waktu data diambil


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------

_store: Optional[MarketDataStore] = None


def get_market_store() -> MarketDataStore:
    global _store
    if _store is None:
        _store = MarketDataStore()
    return _store
