from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime

class TransactionData(BaseModel):
    date: str
    merchant: str
    amount: float
    type: str = "expense"  # "expense" | "income"
    category: str = "Needs review"
    account: str = "Imported account"
    tags: List[str] = Field(default_factory=lambda: ["Imported"])
    receipt: bool = True
    source: str = "document"

class AssetData(BaseModel):
    name: str
    type: str  # "Mutual Funds" | "Fixed Deposit (FD)" | "Cash / Bank Account"
    value: float
    currency: str = "USD"
    hideFromDashboard: bool = False

class FolioData(BaseModel):
    scheme: str
    folio: str
    marketValue: float
    currency: str = "INR"

class ExtractionMetadata(BaseModel):
    accountName: Optional[str] = None
    endingBalance: Optional[float] = None
    statementPeriod: Optional[str] = None
    currency: str = "USD"
    textLength: int = 0
    extractedCount: int = 0
    reviewNeeded: bool = False
    rawSummary: Optional[str] = None

class ExtractionResult(BaseModel):
    success: bool = True
    doc_type: str = "unknown"  # "bank_statement" | "mutual_fund_cas" | "fixed_deposit" | "csv_statement"
    extraction_method: str = "template"  # "template" | "gemini_vision" | "hybrid"
    template_id: Optional[str] = None
    transactions: List[TransactionData] = Field(default_factory=list)
    assets: List[AssetData] = Field(default_factory=list)
    mutual_fund_folios: List[FolioData] = Field(default_factory=list)
    metadata: ExtractionMetadata = Field(default_factory=ExtractionMetadata)
    confidence: float = 1.0
    processing_time_ms: int = 0
    template_created: bool = False
    error: Optional[str] = None

class BaseParser(ABC):
    """Base class for all document parsers."""
    
    parser_id: str = "base"
    supported_types: List[str] = ["pdf"]
    
    @abstractmethod
    def can_handle(self, text: str, filename: str, hints: Optional[Dict[str, Any]] = None) -> float:
        """
        Return confidence score 0.0-1.0 that this parser can handle the document.
        0.0 means cannot handle, 1.0 means exact match.
        """
        pass
        
    @abstractmethod
    def extract(self, file_bytes: bytes, filename: str = "", options: Optional[Dict[str, Any]] = None) -> ExtractionResult:
        """Extract structured data from the document."""
        pass
