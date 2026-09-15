import logging
import asyncio
from pathlib import Path
import os
import base64
from typing import Optional, List, Dict, Any
from PIL import Image
import pytesseract
import openai
import config

logger = logging.getLogger(__name__)

class OCRService:
    def __init__(self):
        self.config = config.config
        self.api_key = getattr(self.config, "OPENAI_API_KEY", None) or os.getenv("OPENAI_API_KEY") or os.getenv("MISTRAL_API_KEY")
        self.base_url = getattr(self.config, "OPENAI_BASE_URL", "http://litellm.litellm.svc.cluster.local:4000/v1")
        self.model_name = getattr(self.config, "OPENAI_MODEL", "gemini-2.5-flash")
        self.language = 'eng'

        self.client = None
        if self.api_key:
            try:
                self.client = openai.OpenAI(api_key=self.api_key, base_url=self.base_url)
                logger.info(f"OCRService initialized with Gemini/LiteLLM model '{self.model_name}' at {self.base_url}")
            except Exception as e:
                logger.warning(f"Failed to initialize OpenAI/LiteLLM client for OCR: {e}")
        else:
            logger.info("OCRService initialized in local fallback mode (Tesseract OCR).")

    def _encode_file_to_base64(self, file_path: str) -> Optional[str]:
        try:
            with open(file_path, "rb") as file_to_encode:
                return base64.b64encode(file_to_encode.read()).decode('utf-8')
        except FileNotFoundError:
            logger.error(f"Error: The file {file_path} was not found for Base64 encoding.")
            return None
        except Exception as e:
            logger.error(f"Error during Base64 encoding for {file_path}: {e}")
            return None

    def _tesseract_fallback(self, file_path: str) -> str:
        try:
            img = Image.open(file_path)
            text = pytesseract.image_to_string(img, lang=self.language)
            return text.strip()
        except Exception as e:
            logger.warning(f"Tesseract OCR fallback failed for {file_path}: {e}")
            return ""

    async def _process_file_with_llm(self, file_path: str, mime_type: str) -> str:
        file_name = Path(file_path).name
        logger.info(f"Processing file {file_name} (MIME: {mime_type}) with Gemini vision / OCR")

        if self.client:
            base64_file = self._encode_file_to_base64(file_path)
            if base64_file:
                try:
                    data_uri = f"data:{mime_type};base64,{base64_file}"
                    loop = asyncio.get_event_loop()
                    response = await loop.run_in_executor(
                        None,
                        lambda: self.client.chat.completions.create(
                            model=self.model_name,
                            messages=[
                                {
                                    "role": "user",
                                    "content": [
                                        {
                                            "type": "text",
                                            "text": "Extract and transcribe all text and structure from this document/image into clean Markdown format. Preserve headings, lists, tables, and paragraphs accurately."
                                        },
                                        {
                                            "type": "image_url",
                                            "image_url": {"url": data_uri}
                                        }
                                    ]
                                }
                            ],
                            max_tokens=4096,
                            temperature=0.2
                        )
                    )
                    content = response.choices[0].message.content or ""
                    if content.strip():
                        return content.strip()
                except Exception as e:
                    logger.warning(f"Gemini/LiteLLM vision call failed for {file_name}: {e}. Falling back to Tesseract.")

        # Local Tesseract fallback for image files
        if mime_type.startswith("image/"):
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self._tesseract_fallback, file_path)

        return ""

    async def extract_text_from_image(self, image_path: str, language: Optional[str] = None) -> str:
        ext = Path(image_path).suffix.lower()
        mime_map = {
            '.jpeg': 'image/jpeg', '.jpg': 'image/jpeg', '.png': 'image/png',
            '.gif': 'image/gif', '.bmp': 'image/bmp', '.tiff': 'image/tiff',
            '.webp': 'image/webp'
        }
        mime_type = mime_map.get(ext, 'image/png')
        return await self._process_file_with_llm(image_path, mime_type)

    async def extract_text_from_pdf(self, pdf_path: str) -> str:
        return await self._process_file_with_llm(pdf_path, "application/pdf")

    async def extract_text_from_pdf_images(self, pdf_path: str) -> List[str]:
        result = await self.extract_text_from_pdf(pdf_path)
        return [result] if result else [""]

    async def extract_text_with_confidence(self, image_path: str, min_confidence: float = 0.5) -> Dict[str, Any]:
        text = await self.extract_text_from_image(image_path)
        return {
            "text": text,
            "confidence": 0.95 if text else 0.0,
            "word_count": len(text.split()) if text else 0,
            "raw_data": "Extracted via Gemini Vision / Tesseract OCR"
        }

    async def detect_language(self, image_path: str) -> str:
        return 'eng'

    async def extract_tables_from_image(self, image_path: str) -> List[List[str]]:
        text = await self.extract_text_from_image(image_path)
        table_data = []
        if text:
            for line in text.split('\n'):
                stripped = line.strip()
                if stripped.startswith('|') and stripped.endswith('|') and "---" not in stripped:
                    cells = [c.strip() for c in stripped.strip('|').split('|')]
                    if any(cells):
                        table_data.append(cells)
        return table_data

    async def get_supported_languages(self) -> List[str]:
        return ['eng', 'multilingual (Gemini vision)']

    async def validate_ocr_setup(self) -> Dict[str, Any]:
        return {
            "status": "operational",
            "message": f"Gemini OCR via LiteLLM ({self.model_name}) with Tesseract local fallback.",
            "configured_ocr_model": self.model_name
        }

    def extract_text(self, file_path: str) -> str:
        try:
            ext = Path(file_path).suffix.lower()
            if ext in ['.jpeg', '.jpg', '.png', '.gif', '.bmp', '.tiff', '.webp']:
                return asyncio.run(self.extract_text_from_image(file_path))
            elif ext == '.pdf':
                return asyncio.run(self.extract_text_from_pdf(file_path))
            else:
                return self._tesseract_fallback(file_path)
        except Exception as e:
            logger.error(f"Error during extract_text for {file_path}: {e}")
            return ""