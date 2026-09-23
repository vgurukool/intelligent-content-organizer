import io
import re
import time
from datetime import datetime
from typing import Optional, Dict, Any, List
from pypdf import PdfReader
from .base import BaseParser, ExtractionResult, ExtractionMetadata, TransactionData

def extract_ordered_lines(page) -> List[List[tuple]]:
    """Extract text fragments from a PDF page sorted in visual reading order (y desc, x asc)."""
    parts = []
    def visitor(text, cm, tm, font_dict, font_size):
        if text.strip():
            parts.append((tm[4], tm[5], text.strip()))
    try:
        page.extract_text(visitor_text=visitor)
    except Exception:
        return [[(0.0, l.strip())] for l in (page.extract_text() or "").splitlines() if l.strip()]
    if not parts:
        return [[(0.0, l.strip())] for l in (page.extract_text() or "").splitlines() if l.strip()]

    parts.sort(key=lambda item: (-round(item[1], 1), round(item[0], 1)))
    lines = []
    curr_y = None
    curr_line = []
    for x, y, t in parts:
        if curr_y is None or abs(y - curr_y) > 3.5:
            if curr_line:
                lines.append(curr_line)
            curr_y = y
            curr_line = [(x, t)]
        else:
            curr_line.append((x, t))
    if curr_line:
        lines.append(curr_line)
    return lines


class BankStatementParser(BaseParser):
    parser_id = "bank_statement"
    supported_types = ["pdf"]
    
    def can_handle(self, text: str, filename: str, hints: Optional[Dict[str, Any]] = None) -> float:
        if hints and hints.get("doc_type") in ["bank_statement", "credit_card"]:
            return 0.95
        
        # Don't handle if it's mutual funds or fixed deposit
        if "Consolidated Account Statement" in text or "myCAMS" in text:
            return 0.0
        if "Fixed Deposit Summary" in text:
            return 0.0
            
        lower = text.lower()
        score = 0.0
        if any(k in lower for k in ["chase", "bank of america", "bofa", "wells fargo", "citibank", "citi", "capital one", "checking", "savings", "credit card", "payment due date"]):
            score += 0.5
        if any(k in lower for k in ["beginning balance", "ending balance", "new balance", "account number", "statement period"]):
            score += 0.3
        if re.search(r'\d{1,2}[-/.]\d{1,2}(?:[-/.]\d{2,4})?', text):
            score += 0.2
            
        return min(1.0, score)

    def extract(self, file_bytes: bytes, filename: str = "", options: Optional[Dict[str, Any]] = None) -> ExtractionResult:
        start_time = time.time()
        options = options or {}
        password = options.get("password") or "30031981"

        text = ""
        reader = None
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
                doc_type="bank_statement",
                extraction_method="template",
                error=f"Password required or invalid PDF: {str(e)}"
            )

        full_header = text[:2000].lower()
        last4 = ""
        acct_match = re.search(r'Account\s*Number:?\s*(?:X+\s*)*([0-9A-Z]{4,16})', text, re.IGNORECASE) or re.search(r'Card\s*(\d{4})', text, re.IGNORECASE)
        if acct_match and acct_match.group(1):
            digits = re.sub(r'[^0-9]', '', acct_match.group(1))
            if len(digits) >= 4:
                last4 = digits[-4:]

        # Detect account type and institution
        extracted_account_name = "Imported account"
        template_id = "generic_bank_v1"
        is_credit_card = False

        if "marriott" in full_header or ("chase" in full_header and any(k in full_header for k in ["credit card", "payment due date", "minimum payment", "autopay is on"])):
            is_credit_card = True
            if "marriott" in full_header:
                extracted_account_name = f"Chase Marriott Bonvoy Card (...{last4})" if last4 else "Chase Marriott Bonvoy Card"
                template_id = "chase_marriott_v1"
            else:
                extracted_account_name = f"Chase Credit Card (...{last4})" if last4 else "Chase Credit Card"
                template_id = "chase_credit_card_v1"
        elif "chase private client checking" in full_header or ("chase" in full_header and "checking" in full_header):
            extracted_account_name = f"Chase Private Client Checking (...{last4})" if last4 else "Chase Private Client Checking"
            template_id = "chase_checking_v1"
        elif "chase private client savings" in full_header or ("chase" in full_header and "savings" in full_header):
            extracted_account_name = f"Chase Private Client Savings (...{last4})" if last4 else "Chase Private Client Savings"
            template_id = "chase_savings_v1"
        elif 'bank of america' in full_header or 'bofa' in full_header:
            if 'checking' in full_header:
                extracted_account_name = f"Bank of America Checking (...{last4})" if last4 else "Bank of America Checking"
                template_id = "bofa_checking_v1"
            elif 'savings' in full_header:
                extracted_account_name = f"Bank of America Savings (...{last4})" if last4 else "Bank of America Savings"
                template_id = "bofa_savings_v1"
            else:
                extracted_account_name = f"Bank of America Card (...{last4})" if last4 else "Bank of America Account"
                template_id = "bofa_card_v1"
                is_credit_card = True
        elif any(k in full_header for k in ['payment due date', 'minimum payment due', 'credit card']):
            extracted_account_name = f"Credit Card (...{last4})" if last4 else "Credit Card"
            template_id = "credit_card_generic_v1"
            is_credit_card = True
        elif 'checking' in full_header:
            extracted_account_name = f"Checking Account (...{last4})" if last4 else "Checking Account"
            template_id = "generic_checking_v1"
        elif 'savings' in full_header:
            extracted_account_name = f"Savings Account (...{last4})" if last4 else "Savings Account"
            template_id = "generic_savings_v1"

        # Determine ending/new balance
        ending_balance = None
        if is_credit_card:
            new_bal_m = re.search(r"New Balance:?\s*\$?([0-9,]+\.\d{2})", text, re.IGNORECASE)
            if new_bal_m:
                try:
                    ending_balance = float(new_bal_m.group(1).replace(",", ""))
                except Exception:
                    pass
        else:
            end_bal_m = re.search(r"(?:Ending|Total|Closing)\s*(?:Account|Savings|Checking)?\s*Balance[^\n\d]*\$?\s*([0-9,]+\.\d{2})", text, re.IGNORECASE)
            if end_bal_m:
                try:
                    ending_balance = float(end_bal_m.group(1).replace(",", ""))
                except Exception:
                    pass

        # Statement year detection
        year_m = re.search(r"(?:through|Statement Date:?|Period:?)[^\n]*?(202[4-9]|203[0-9])", text) or re.search(r" (202[4-9]|203[0-9]) ", text)
        statement_year = year_m.group(1) if year_m else str(datetime.now().year)

        date_regex = re.compile(r"^(\d{1,2}/\d{1,2}(?:/\d{2,4})?)")
        extracted_transactions: List[TransactionData] = []

        # Iterate over ordered lines from each page
        if reader and reader.pages:
            for page in reader.pages:
                ordered_groups = extract_ordered_lines(page)
                for group in ordered_groups:
                    line_str = " ".join([t for _, t in group]).strip()
                    d_match = date_regex.match(line_str)
                    if not d_match:
                        continue

                    raw_date = d_match.group(1)
                    rest = line_str[len(raw_date):].strip()

                    lower_rest = rest.lower()
                    if any(ign in lower_rest for ign in [
                        "beginning balance", "ending balance", "new balance",
                        "payment due date", "payment due", "minimum payment due",
                        "customer service", "total fees charged", "total interest charged"
                    ]):
                        continue

                    # ISO date resolution
                    date_parts = raw_date.split("/")
                    if len(date_parts) == 2:
                        iso_date = f"{statement_year}-{date_parts[0].zfill(2)}-{date_parts[1].zfill(2)}"
                    elif len(date_parts) == 3:
                        yr = date_parts[2]
                        if len(yr) == 2: yr = "20" + yr
                        iso_date = f"{yr}-{date_parts[0].zfill(2)}-{date_parts[1].zfill(2)}"
                    else:
                        iso_date = f"{statement_year}-01-01"

                    # Numeric amount and merchant parsing
                    neg_m = re.search(r"-\s*([\d,]+\.\d{2})\s+([\d,]+\.\d{2})$", rest)
                    pos_two_m = re.search(r"([\d,]+\.\d{2})\s+([\d,]+\.\d{2})$", rest)
                    single_neg_m = re.search(r"-\s*([\d,]+\.\d{2})$", rest)
                    single_pos_m = re.search(r"\+?\s*([\d,]+\.\d{2})$", rest)

                    amount = 0.0
                    merchant = rest
                    ttype = "expense"

                    if neg_m:
                        amount = float(neg_m.group(1).replace(",", ""))
                        merchant = rest[:neg_m.start()].strip()
                        ttype = "expense"
                    elif pos_two_m:
                        v1 = float(pos_two_m.group(1).replace(",", ""))
                        v2 = float(pos_two_m.group(2).replace(",", ""))
                        merchant = rest[:pos_two_m.start()].strip()
                        ttype = "income"
                        if v1 > v2 and v1 > 5000 and v2 < 5000:
                            amount = v2
                        else:
                            amount = v1
                    elif single_neg_m:
                        amount = float(single_neg_m.group(1).replace(",", ""))
                        merchant = rest[:single_neg_m.start()].strip()
                        ttype = "income" if is_credit_card else "expense"
                    elif single_pos_m:
                        amount = float(single_pos_m.group(1).replace(",", ""))
                        merchant = rest[:single_pos_m.start()].strip()
                        ttype = "expense" if is_credit_card else "income"
                    else:
                        continue

                    if amount == 0.0 or amount > 1000000:
                        continue

                    # Clean merchant text
                    merchant = re.sub(r"PPD ID:\s*\S+", "", merchant)
                    merchant = re.sub(r"Web ID:\s*\S+", "", merchant)
                    merchant = re.sub(r"Transaction\s*#:\s*\S+", "", merchant)
                    merchant = re.sub(r"^\d{2}/\d{2}\s+", "", merchant)
                    merchant = re.sub(r"\s+", " ", merchant).strip()

                    if len(re.findall(r'[a-zA-Z]', merchant)) < 2:
                        continue

                    # Categorization
                    category = "Needs review"
                    low_m = merchant.lower()
                    if any(k in low_m for k in ["payroll", "interest payment", "zelle payment from", "direct deposit"]):
                        category = "Income"
                        ttype = "income"
                    elif any(k in low_m for k in ["automatic payment", "autopay", "thank you", "citi autopay", "payment to card"]):
                        category = "Credit Card Payment"
                    elif any(k in low_m for k in ["online transfer", "transfer to", "transfer from", "xfer"]):
                        category = "Transfer"
                    elif any(k in low_m for k in ["grocers", "wal-mart", "kroger", "trader joe", "parivar", "subhlaxmi"]):
                        category = "Groceries"
                    elif any(k in low_m for k in ["chipotle", "subway", "domino", "biryani", "sweets", "food", "baguette", "starbucks", "deli"]):
                        category = "Dining & Food"
                    elif any(k in low_m for k in ["energy", "cpenergy", "tmobile", "att*", "verizon", "electric"]):
                        category = "Utilities & Bills"
                    elif any(k in low_m for k in ["insurance", "aaa tx"]):
                        category = "Insurance"
                    elif any(k in low_m for k in ["home depot", "office depot", "michaels", "dollar tree"]):
                        category = "Shopping"

                    extracted_transactions.append(TransactionData(
                        date=iso_date,
                        merchant=merchant,
                        amount=round(amount, 2),
                        type=ttype,
                        category=category,
                        account=extracted_account_name,
                        tags=["PDF Import"],
                        receipt=True,
                        source="document"
                    ))

        elapsed_ms = int((time.time() - start_time) * 1000)

        return ExtractionResult(
            success=True,
            doc_type="credit_card" if is_credit_card else "bank_statement",
            extraction_method="template",
            template_id=template_id,
            transactions=extracted_transactions,
            metadata=ExtractionMetadata(
                accountName=extracted_account_name,
                endingBalance=ending_balance,
                currency="USD",
                textLength=len(text),
                extractedCount=len(extracted_transactions),
                reviewNeeded=False
            ),
            confidence=0.95 if extracted_transactions else 0.4,
            processing_time_ms=elapsed_ms
        )
