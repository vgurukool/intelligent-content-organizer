import os
import io
import re
import json
import time
import base64
import logging
from typing import Optional, Dict, Any, List
import openai
from pypdf import PdfReader
import config
from .base import BaseParser, ExtractionResult, ExtractionMetadata, TransactionData, AssetData, FolioData

logger = logging.getLogger(__name__)

class GeminiExtractor(BaseParser):
    parser_id = "gemini_vision"
    supported_types = ["pdf", "csv", "png", "jpg", "jpeg"]

    def __init__(self):
        self.config = config.config
        self.api_key = getattr(self.config, "OPENAI_API_KEY", None) or os.getenv("OPENAI_API_KEY") or "sk-litellm-vgurukool-master-2026"
        self.base_url = getattr(self.config, "OPENAI_BASE_URL", "http://litellm.litellm.svc.cluster.local:4000/v1")
        self.model_name = getattr(self.config, "OPENAI_MODEL", "gemini-2.5-flash")
        
        self.client = None
        if self.api_key:
            try:
                self.client = openai.OpenAI(api_key=self.api_key, base_url=self.base_url)
                logger.info(f"GeminiExtractor initialized with model '{self.model_name}' at {self.base_url}")
            except Exception as e:
                logger.warning(f"Failed to initialize OpenAI/LiteLLM client in GeminiExtractor: {e}")

    def can_handle(self, text: str, filename: str, hints: Optional[Dict[str, Any]] = None) -> float:
        # Gemini can handle any document as fallback
        return 0.3

    def _extract_text_pdf(self, file_bytes: bytes, password: str = "30031981") -> str:
        text = ""
        try:
            reader = PdfReader(io.BytesIO(file_bytes))
            if reader.is_encrypted:
                try:
                    reader.decrypt(password)
                except Exception:
                    pass
            for page in reader.pages[:10]: # First 10 pages max
                t = page.extract_text()
                if t:
                    text += t + "\n"
        except Exception:
            pass
        return text

    def extract(self, file_bytes: bytes, filename: str = "", options: Optional[Dict[str, Any]] = None) -> ExtractionResult:
        start_time = time.time()
        options = options or {}
        password = options.get("password") or "30031981"
        is_pdf = filename.lower().endswith(".pdf")
        is_csv = filename.lower().endswith(".csv")

        extracted_text = ""
        if is_pdf:
            extracted_text = self._extract_text_pdf(file_bytes, password)
        elif is_csv:
            extracted_text = file_bytes.decode("utf-8", errors="ignore")

        if not self.client:
            return ExtractionResult(
                success=False,
                doc_type="unknown",
                extraction_method="gemini_vision",
                error="LLM Client not initialized for Gemini extraction"
            )

        prompt = """You are a financial document processing expert. Analyze this document and extract all structured data into a single strict JSON object.

Output JSON format:
{
  "doc_type": "bank_statement | mutual_fund_cas | fixed_deposit | csv_statement | other",
  "account_name": "Name of institution or account (e.g. Chase Checking ...1234)",
  "ending_balance": 1234.56 or null,
  "currency": "USD" or "INR",
  "statement_period": "YYYY-MM-DD to YYYY-MM-DD" or null,
  "transactions": [
    {
      "date": "YYYY-MM-DD",
      "merchant": "Clean vendor or payee name",
      "amount": 123.45,
      "type": "expense" or "income",
      "category": "Groceries | Dining | Utilities | Housing | Transportation | Income | Credit Card Payment | Transfer | Needs review"
    }
  ],
  "assets": [
    {
      "name": "Asset name",
      "type": "Fixed Deposit (FD) | Mutual Funds | Cash / Bank Account",
      "value": 12345.67,
      "currency": "USD" or "INR"
    }
  ],
  "folios": [
    {
      "scheme": "Scheme name",
      "folio": "Folio number",
      "marketValue": 12345.67,
      "currency": "INR"
    }
  ],
  "template_hints": {
    "header_keywords": ["keyword1", "keyword2"],
    "suggested_template_id": "suggested_unique_name_v1"
  }
}

Important Rules:
1. Extract ALL individual transaction rows if present.
2. Amounts must be positive numbers. Type must be 'expense' or 'income'.
3. Output ONLY the JSON block, no markdown commentary.
"""

        messages: List[Dict[str, Any]] = []

        # If we have extracted text, pass as text prompt. If image/scanned PDF, send as base64 data URI
        if extracted_text and len(extracted_text.strip()) > 50:
            content_text = f"{prompt}\n\nDocument Content:\n{extracted_text[:12000]}"
            messages.append({"role": "user", "content": content_text})
        else:
            # Fallback to vision
            b64_str = base64.b64encode(file_bytes).decode("utf-8")
            mime = "application/pdf" if is_pdf else "image/png"
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64_str}"}}
                ]
            })

        try:
            resp = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.1,
                max_tokens=4096
            )
            raw_content = resp.choices[0].message.content or ""
            clean_json = raw_content.strip()
            if clean_json.startswith("```json"):
                clean_json = clean_json[7:]
            if clean_json.startswith("```"):
                clean_json = clean_json[3:]
            if clean_json.endswith("```"):
                clean_json = clean_json[:-3]
            clean_json = clean_json.strip()

            parsed_data = json.loads(clean_json)

            transactions = [
                TransactionData(
                    date=t.get("date") or "2026-08-01",
                    merchant=t.get("merchant") or "Unknown Merchant",
                    amount=abs(float(t.get("amount") or 0.0)),
                    type=t.get("type", "expense"),
                    category=t.get("category", "Needs review"),
                    account=parsed_data.get("account_name") or "Imported account",
                    tags=["Gemini Vision"],
                    receipt=True,
                    source="document"
                )
                for t in parsed_data.get("transactions", [])
                if float(t.get("amount") or 0.0) > 0
            ]

            assets = [
                AssetData(
                    name=a.get("name") or "Asset",
                    type=a.get("type") or "Cash / Bank Account",
                    value=float(a.get("value") or 0.0),
                    currency=a.get("currency") or parsed_data.get("currency", "USD"),
                    hideFromDashboard=False
                )
                for a in parsed_data.get("assets", [])
            ]

            folios = [
                FolioData(
                    scheme=f.get("scheme") or "Scheme",
                    folio=f.get("folio") or "0000",
                    marketValue=float(f.get("marketValue") or 0.0),
                    currency=f.get("currency") or "INR"
                )
                for f in parsed_data.get("folios", [])
            ]

            # Also synthesize and save template if template_hints present
            template_created = False
            template_id = "gemini_generated"
            template_hints = parsed_data.get("template_hints")
            if template_hints and template_hints.get("header_keywords"):
                suggested_id = template_hints.get("suggested_template_id") or f"auto_{int(time.time())}"
                template_id = suggested_id
                template_created = self._save_synthesized_template(suggested_id, parsed_data, template_hints)

            elapsed_ms = int((time.time() - start_time) * 1000)

            return ExtractionResult(
                success=True,
                doc_type=parsed_data.get("doc_type", "bank_statement"),
                extraction_method="gemini_vision",
                template_id=template_id,
                transactions=transactions,
                assets=assets,
                mutual_fund_folios=folios,
                metadata=ExtractionMetadata(
                    accountName=parsed_data.get("account_name"),
                    endingBalance=parsed_data.get("ending_balance"),
                    statementPeriod=parsed_data.get("statement_period"),
                    currency=parsed_data.get("currency", "USD"),
                    textLength=len(extracted_text),
                    extractedCount=len(transactions) + len(assets) + len(folios),
                    rawSummary=f"Processed with {self.model_name}"
                ),
                confidence=0.90,
                processing_time_ms=elapsed_ms,
                template_created=template_created
            )

        except Exception as e:
            logger.error(f"Gemini extraction failed: {str(e)}")
            return ExtractionResult(
                success=False,
                doc_type="unknown",
                extraction_method="gemini_vision",
                error=f"Gemini extraction error: {str(e)}"
            )

    def _save_synthesized_template(self, template_id: str, parsed_data: Dict[str, Any], hints: Dict[str, Any]) -> bool:
        try:
            store_dir = self.config.TEMPLATE_STORE_PATH
            os.makedirs(store_dir, exist_ok=True)
            tpl_path = os.path.join(store_dir, f"{template_id}.json")
            
            template_doc = {
                "id": template_id,
                "doc_type": parsed_data.get("doc_type", "bank_statement"),
                "account_name": parsed_data.get("account_name"),
                "currency": parsed_data.get("currency", "USD"),
                "fingerprint": {
                    "header_keywords": hints.get("header_keywords", []),
                    "confidence_threshold": 0.8
                },
                "parser_config": {
                    "account_name": parsed_data.get("account_name"),
                    "sample_transaction_count": len(parsed_data.get("transactions", []))
                },
                "stats": {
                    "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "last_used": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "usage_count": 1,
                    "accuracy_score": 0.95,
                    "auto_generated": True
                }
            }
            with open(tpl_path, "w", encoding="utf-8") as f:
                json.dump(template_doc, f, indent=2)
            logger.info(f"Synthesized template saved to {tpl_path}")
            return True
        except Exception as err:
            logger.warning(f"Could not save synthesized template: {err}")
            return False
