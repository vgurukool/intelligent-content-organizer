import io
import re
import time
from typing import Optional, Dict, Any
from pypdf import PdfReader
from .base import BaseParser, ExtractionResult, ExtractionMetadata, FolioData, AssetData

class MutualFundCASParser(BaseParser):
    parser_id = "mutual_fund_cas"
    supported_types = ["pdf"]
    
    def can_handle(self, text: str, filename: str, hints: Optional[Dict[str, Any]] = None) -> float:
        if hints and hints.get("doc_type") == "mutual_fund_cas":
            return 1.0
        
        matches = 0
        if "Consolidated Account Statement" in text:
            matches += 2
        if "PORTFOLIO SUMMARY" in text:
            matches += 1
        if "myCAMS" in text or "CAMS" in text or "KFintech" in text:
            matches += 1
        if "Folio No:" in text:
            matches += 1
            
        if matches >= 2:
            return 0.95
        elif matches == 1:
            return 0.5
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
                doc_type="mutual_fund_cas",
                extraction_method="template",
                template_id="mf_cas_v1",
                error=f"PDF Decryption or read error: {str(e)}"
            )

        folios = []
        assets = []
        lines = text.splitlines()
        current_scheme = ""
        current_folio = ""

        for i, line in enumerate(lines):
            line_str = line.strip()
            if "Folio No:" in line_str:
                f_match = re.search(r'Folio No:\s*([0-9/]+)', line_str, re.IGNORECASE)
                if f_match:
                    current_folio = f_match.group(1)
                for j in range(i - 1, max(-1, i - 5), -1):
                    if "ISIN:" in lines[j] or "Fund" in lines[j] or "Plan" in lines[j]:
                        current_scheme = lines[j].strip()
                        break
            if "Market Value on" in line_str or "Market Value" in line_str:
                mv_match = re.search(r'Market Value[^\d]*INR\s*([\d,]+\.\d{2})', line_str, re.IGNORECASE)
                if mv_match and current_folio:
                    mv = float(mv_match.group(1).replace(',', ''))
                    if mv > 0:
                        folio_obj = FolioData(
                            scheme=current_scheme or "Mutual Fund Scheme",
                            folio=current_folio,
                            marketValue=mv,
                            currency="INR"
                        )
                        folios.append(folio_obj)
                        
                        clean_scheme = re.sub(r'^[A-Z0-9]+-', '', current_scheme).split('- ISIN')[0].strip() or "Mutual Fund"
                        assets.append(AssetData(
                            name=f"{clean_scheme} (Folio: {current_folio})",
                            type="Mutual Funds",
                            currency="INR",
                            value=mv,
                            hideFromDashboard=False
                        ))

        total_mv = sum(f.marketValue for f in folios)
        elapsed_ms = int((time.time() - start_time) * 1000)

        return ExtractionResult(
            success=True,
            doc_type="mutual_fund_cas",
            extraction_method="template",
            template_id="mf_cas_v1",
            mutual_fund_folios=folios,
            assets=assets,
            metadata=ExtractionMetadata(
                accountName="Mutual Fund Portfolio",
                endingBalance=total_mv,
                currency="INR",
                textLength=len(text),
                extractedCount=len(folios)
            ),
            confidence=0.98 if folios else 0.5,
            processing_time_ms=elapsed_ms
        )
