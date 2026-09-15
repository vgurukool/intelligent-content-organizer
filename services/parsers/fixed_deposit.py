import io
import re
import time
from typing import Optional, Dict, Any
from pypdf import PdfReader
from .base import BaseParser, ExtractionResult, ExtractionMetadata, AssetData

class FixedDepositParser(BaseParser):
    parser_id = "fixed_deposit"
    supported_types = ["pdf"]
    
    def can_handle(self, text: str, filename: str, hints: Optional[Dict[str, Any]] = None) -> float:
        if hints and hints.get("doc_type") == "fixed_deposit":
            return 1.0
            
        if "Fixed Deposit Summary" in text:
            return 0.95
        if "Maturity Date" in text and "Rate of Interest" in text and ("Fixed Deposit" in text or "FD" in text or "HDFC" in text):
            return 0.90
        return 0.0

    def extract(self, file_bytes: bytes, filename: str = "", options: Optional[Dict[str, Any]] = None) -> ExtractionResult:
        start_time = time.time()
        options = options or {}
        password = options.get("password") or "30031981"
        
        text = ""
        try:
            reader = PdfReader(io.BytesIO(file_bytes))
            if reader.is_encrypted:
                try:
                    reader.decrypt(password)
                except Exception:
                    pass
            for page in reader.pages:
                t = page.extract_text()
                if t:
                    text += t + "\n"
        except Exception as e:
            return ExtractionResult(
                success=False,
                doc_type="fixed_deposit",
                extraction_method="template",
                template_id="hdfc_fd_v1",
                error=f"PDF Decryption or read error: {str(e)}"
            )

        total_match = re.search(r'Total\s+INR\s+([\d,]+\.\d{2})\s+([\d,]+\.\d{2})', text, re.IGNORECASE) or re.search(r'Total[^\d]+([\d,]+\.\d{2})', text, re.IGNORECASE)
        principal_val = 0.0
        maturity_val = 0.0
        if total_match:
            try:
                principal_val = float(total_match.group(1).replace(',', ''))
            except Exception:
                pass
            if total_match.lastindex and total_match.lastindex >= 2 and total_match.group(2):
                try:
                    maturity_val = float(total_match.group(2).replace(',', ''))
                except Exception:
                    pass

        fd_matches = re.findall(r' 5030\d{10} ', text)
        fd_count = len(fd_matches) if fd_matches else 1
        
        assets = []
        if principal_val > 0:
            assets.append(AssetData(
                name=f"HDFC Fixed Deposits ({fd_count} FDs)",
                type="Fixed Deposit (FD)",
                currency="INR",
                value=principal_val,
                hideFromDashboard=False
            ))

        elapsed_ms = int((time.time() - start_time) * 1000)

        return ExtractionResult(
            success=True,
            doc_type="fixed_deposit",
            extraction_method="template",
            template_id="hdfc_fd_v1",
            assets=assets,
            metadata=ExtractionMetadata(
                accountName="HDFC Fixed Deposits",
                endingBalance=principal_val,
                currency="INR",
                textLength=len(text),
                extractedCount=len(assets),
                rawSummary=f"Principal: INR {principal_val:,.2f}, Maturity: INR {maturity_val:,.2f}, FDs: {fd_count}"
            ),
            confidence=0.95 if principal_val > 0 else 0.6,
            processing_time_ms=elapsed_ms
        )
