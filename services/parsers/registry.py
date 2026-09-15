import io
import os
import glob
import json
import time
import logging
from typing import List, Dict, Any, Optional
from pypdf import PdfReader
import config

from .base import BaseParser, ExtractionResult, ExtractionMetadata
from .bank_statement import BankStatementParser
from .mutual_fund_cas import MutualFundCASParser
from .fixed_deposit import FixedDepositParser
from .csv_statement import CSVStatementParser
from .gemini_extractor import GeminiExtractor

logger = logging.getLogger(__name__)

class ParserRegistry:
    def __init__(self):
        self.config = config.config
        self.parsers: List[BaseParser] = [
            MutualFundCASParser(),
            FixedDepositParser(),
            BankStatementParser(),
            CSVStatementParser()
        ]
        self.gemini_extractor = GeminiExtractor()
        self.template_store_path = self.config.TEMPLATE_STORE_PATH
        os.makedirs(self.template_store_path, exist_ok=True)
        self._seed_default_templates()

    def _seed_default_templates(self):
        """Seed default built-in templates into template store if absent."""
        defaults = [
            {
                "id": "chase_checking_v1",
                "bank": "chase",
                "account_type": "checking",
                "doc_type": "bank_statement",
                "version": 1,
                "fingerprint": {
                    "header_keywords": ["JPMorgan Chase Bank", "CHECKING SUMMARY"],
                    "confidence_threshold": 0.8
                },
                "parser_config": {
                    "parser_class": "BankStatementParser",
                    "account_name": "Chase Private Client Checking"
                },
                "stats": {"auto_generated": False, "usage_count": 0, "accuracy_score": 1.0}
            },
            {
                "id": "bofa_checking_v1",
                "bank": "bofa",
                "account_type": "checking",
                "doc_type": "bank_statement",
                "version": 1,
                "fingerprint": {
                    "header_keywords": ["Bank of America", "Checking"],
                    "confidence_threshold": 0.8
                },
                "parser_config": {
                    "parser_class": "BankStatementParser",
                    "account_name": "Bank of America Checking"
                },
                "stats": {"auto_generated": False, "usage_count": 0, "accuracy_score": 1.0}
            },
            {
                "id": "mf_cas_v1",
                "bank": "cams_kfintech",
                "account_type": "mutual_funds",
                "doc_type": "mutual_fund_cas",
                "version": 1,
                "fingerprint": {
                    "header_keywords": ["Consolidated Account Statement", "Folio No:"],
                    "confidence_threshold": 0.85
                },
                "parser_config": {
                    "parser_class": "MutualFundCASParser"
                },
                "stats": {"auto_generated": False, "usage_count": 0, "accuracy_score": 1.0}
            },
            {
                "id": "hdfc_fd_v1",
                "bank": "hdfc",
                "account_type": "fixed_deposit",
                "doc_type": "fixed_deposit",
                "version": 1,
                "fingerprint": {
                    "header_keywords": ["Fixed Deposit Summary"],
                    "confidence_threshold": 0.85
                },
                "parser_config": {
                    "parser_class": "FixedDepositParser"
                },
                "stats": {"auto_generated": False, "usage_count": 0, "accuracy_score": 1.0}
            },
            {
                "id": "citi_csv_v1",
                "bank": "citi",
                "account_type": "credit_card",
                "doc_type": "csv_statement",
                "version": 1,
                "fingerprint": {
                    "header_keywords": ["Date", "Description", "Debit", "Credit"],
                    "confidence_threshold": 0.85
                },
                "parser_config": {
                    "parser_class": "CSVStatementParser"
                },
                "stats": {"auto_generated": False, "usage_count": 0, "accuracy_score": 1.0}
            }
        ]
        for tpl in defaults:
            p = os.path.join(self.template_store_path, f"{tpl['id']}.json")
            if not os.path.exists(p):
                try:
                    with open(p, "w", encoding="utf-8") as f:
                        json.dump(tpl, f, indent=2)
                except Exception as e:
                    logger.warning(f"Could not seed template {tpl['id']}: {e}")

    def list_templates(self) -> List[Dict[str, Any]]:
        templates = []
        pattern = os.path.join(self.template_store_path, "*.json")
        for f in glob.glob(pattern):
            try:
                with open(f, "r", encoding="utf-8") as fh:
                    templates.append(json.load(fh))
            except Exception:
                pass
        return templates

    def get_template(self, template_id: str) -> Optional[Dict[str, Any]]:
        p = os.path.join(self.template_store_path, f"{template_id}.json")
        if os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8") as fh:
                    return json.load(fh)
            except Exception:
                pass
        return None

    def _quick_extract_text(self, file_bytes: bytes, filename: str, options: Dict[str, Any]) -> str:
        if filename.lower().endswith(".csv"):
            return file_bytes[:4000].decode("utf-8", errors="ignore")
        if filename.lower().endswith(".pdf"):
            password = options.get("password") or "30031981"
            try:
                reader = PdfReader(io.BytesIO(file_bytes))
                if reader.is_encrypted:
                    try:
                        reader.decrypt(password)
                    except Exception:
                        pass
                text = ""
                for page in reader.pages[:3]:  # sample first 3 pages
                    t = page.extract_text()
                    if t:
                        text += t + "\n"
                return text
            except Exception:
                return ""
        return ""

    def extract(self, file_bytes: bytes, filename: str = "", doc_type: str = "unknown", hints: Optional[Dict[str, Any]] = None) -> ExtractionResult:
        hints = hints or {}
        if doc_type != "unknown":
            hints["doc_type"] = doc_type

        # 1. Quick text sampling for parser routing
        sampled_text = self._quick_extract_text(file_bytes, filename, hints)

        # 2. Evaluate deterministic fast-path parsers
        best_parser = None
        best_score = 0.0

        for parser in self.parsers:
            score = parser.can_handle(sampled_text, filename, hints)
            if score > best_score:
                best_score = score
                best_parser = parser

        # If high-confidence match found (>= 0.70), execute fast path
        if best_parser and best_score >= 0.70:
            logger.info(f"Fast-path matched parser '{best_parser.parser_id}' with score {best_score:.2f}")
            res = best_parser.extract(file_bytes, filename, hints)
            
            # If extracted successfully with data, return immediately
            if res.success and (res.transactions or res.assets or res.mutual_fund_folios):
                # Update usage stats on matched template if any
                if res.template_id:
                    self._increment_template_usage(res.template_id)
                return res

            logger.info(f"Fast-path parser '{best_parser.parser_id}' yielded 0 items, falling back to Gemini Vision.")

        # 3. Fallback to Gemini Vision / Auto-Template Extractor
        logger.info(f"Invoking Gemini Vision fallback for file {filename}")
        res = self.gemini_extractor.extract(file_bytes, filename, hints)
        return res

    def _increment_template_usage(self, template_id: str):
        tpl = self.get_template(template_id)
        if tpl:
            stats = tpl.get("stats", {})
            stats["usage_count"] = stats.get("usage_count", 0) + 1
            stats["last_used"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            tpl["stats"] = stats
            p = os.path.join(self.template_store_path, f"{template_id}.json")
            try:
                with open(p, "w", encoding="utf-8") as f:
                    json.dump(tpl, f, indent=2)
            except Exception:
                pass
