# agents/market_intelligence/store.py

from __future__ import annotations

import asyncio
from dataclasses import dataclass
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

# Threshold staleness default (jam)
_DEFAULT_MAX_AGE_HOURS: int = 25


@dataclass
class VarietySummary:
    variety_code:  str
    price_min_idr: int      # IDR per kg
    price_max_idr: int      # IDR per kg
    price_avg_idr: int      # IDR per kg
    sample_count:  int      # Jumlah listing yang dijadikan basis
    scraped_at:    datetime # Waktu data diambil


class MarketDataStore:
    """
    Singleton in-memory store untuk data pasar durian terbaru.
    Thread-safe menggunakan asyncio.Lock.
    Diakses oleh agent (write) dan API endpoint (read).
    """

    _instance: Optional["MarketDataStore"] = None

    def __new__(cls) -> "MarketDataStore":
        if cls._instance is None:
            inst = super().__new__(cls)
            inst._data_lock = asyncio.Lock()
            inst._latest = None
            inst._by_variety = {}
            cls._instance = inst
        return cls._instance

    # ──────────────────────────────────────────────────────────────────────
    # Write
    # ──────────────────────────────────────────────────────────────────────

    async def save(self, payload: MarketReportPayload) -> None:
        """Simpan hasil run terbaru dan rebuild index per-varietas."""
        async with self._data_lock:
            self._latest = payload

            by_variety: Dict[str, List[MarketPriceEntry]] = {}
            for entry in payload.entries:
                code = entry.variety_code.value
                by_variety.setdefault(code, []).append(entry)
            self._by_variety = by_variety

        logger.info(
            f"[Store] Data pasar tersimpan: "
            f"run_id={payload.run_id} | "
            f"entries={payload.entry_count} | "
            f"varieties={sorted(by_variety.keys())} | "
            f"status={payload.status.value} | "
            f"duration={payload.run_duration_sec:.1f}s"
        )

    # ──────────────────────────────────────────────────────────────────────
    # Read
    # ──────────────────────────────────────────────────────────────────────

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

    async def get_price_summary(self) -> Dict[str, VarietySummary]:
        """
        Kembalikan ringkasan harga min/max/avg per varietas.
        Berguna untuk endpoint GET /market/prices.
        """
        async with self._data_lock:
            return self._build_price_summary()

    def _build_price_summary(self) -> Dict[str, VarietySummary]:
        """Build price summary. HARUS dipanggil saat _data_lock dipegang."""
        summary: Dict[str, VarietySummary] = {}

        for code, entries in self._by_variety.items():
            kg_prices = self._collect_kg_prices(entries)

            if not kg_prices:
                logger.debug(
                    f"[Store] Tidak ada data harga per-kg untuk varietas '{code}', skip."
                )
                continue

            summary[code] = VarietySummary(
                variety_code  = code,
                price_min_idr = round(min(kg_prices)),
                price_max_idr = round(max(kg_prices)),
                price_avg_idr = round(sum(kg_prices) / len(kg_prices)),
                sample_count  = len(entries),
                scraped_at    = (
                    self._latest.run_ended_at
                    if self._latest else datetime.now(timezone.utc)
                ),
            )

        return summary

    @staticmethod
    def _collect_kg_prices(entries: List[MarketPriceEntry]) -> List[float]:
        """
        Kumpulkan semua harga per-kg dari entries.
        Prioritas: price_per_kg_avg → mid-range (min+max)/2 → min → max.
        """
        prices: List[float] = []

        for e in entries:
            if e.price_per_kg_avg is not None:
                prices.append(e.price_per_kg_avg)
            elif e.price_per_kg_min is not None and e.price_per_kg_max is not None:
                prices.append((e.price_per_kg_min + e.price_per_kg_max) / 2)
            elif e.price_per_kg_min is not None:
                prices.append(e.price_per_kg_min)
            elif e.price_per_kg_max is not None:
                prices.append(e.price_per_kg_max)

        return prices

    async def is_stale(self, max_age_hours: int = _DEFAULT_MAX_AGE_HOURS) -> bool:
        """
        True jika data lebih tua dari max_age_hours atau belum ada.
        Digunakan oleh endpoint untuk memberi tahu client tentang freshness.
        """
        if max_age_hours <= 0:
            raise ValueError("max_age_hours harus > 0.")

        async with self._data_lock:
            if self._latest is None:
                return True
            age_sec = (datetime.now(timezone.utc) - self._latest.run_ended_at).total_seconds()
            return age_sec > (max_age_hours * 3600)

    async def has_data(self) -> bool:
        """True jika ada setidaknya satu entry harga yang tersimpan."""
        async with self._data_lock:
            return self._latest is not None and bool(self._latest.entries)

    async def clear(self) -> None:
        """Hapus semua data (berguna untuk testing)."""
        async with self._data_lock:
            self._latest    = None
            self._by_variety = {}
        logger.warning("[Store] Data pasar dihapus (clear() dipanggil).")

    async def get_diagnostics(self) -> dict:
        """Kembalikan info diagnostik store untuk health check."""
        async with self._data_lock:
            if self._latest is None:
                return {"has_data": False}

            age_sec = (datetime.now(timezone.utc) - self._latest.run_ended_at).total_seconds()
            return {
                "has_data":        True,
                "run_id":          self._latest.run_id,
                "status":          self._latest.status.value,
                "entry_count":     self._latest.entry_count,
                "variety_count":   len(self._by_variety),
                "age_hours":       round(age_sec / 3600, 2),
                "is_stale":        age_sec > (_DEFAULT_MAX_AGE_HOURS * 3600),
                "run_ended_at":    self._latest.run_ended_at.isoformat(),
                "duration_sec":    round(self._latest.run_duration_sec, 1),
            }


# ──────────────────────────────────────────────────────────────────────────
# Singleton accessor
# ──────────────────────────────────────────────────────────────────────────

_store: Optional[MarketDataStore] = None


def get_market_store() -> MarketDataStore:
    global _store
    if _store is None:
        _store = MarketDataStore()
    return _store