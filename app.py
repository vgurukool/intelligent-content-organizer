import gradio as gr
import os
import asyncio
import json
import logging
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
import nest_asyncio

# Apply nest_asyncio to handle nested event loops in Gradio
nest_asyncio.apply()

# Import our custom modules
from mcp_tools.ingestion_tool import IngestionTool
from mcp_tools.search_tool import SearchTool
from mcp_tools.generative_tool import GenerativeTool
from services.vector_store_service import VectorStoreService
from services.document_store_service import DocumentStoreService
from services.embedding_service import EmbeddingService
from services.llm_service import LLMService
from services.ocr_service import OCRService
from core.models import SearchResult, Document
import config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ContentOrganizerMCPServer:
    def __init__(self):
        # Initialize services
        logger.info("Initializing Content Organizer MCP Server...")
        self.vector_store = VectorStoreService()
        self.document_store = DocumentStoreService()
        self.embedding_service = EmbeddingService()
        self.llm_service = LLMService()
        self.ocr_service = OCRService()
        
        # Initialize tools
        self.ingestion_tool = IngestionTool(
            vector_store=self.vector_store,
            document_store=self.document_store,
            embedding_service=self.embedding_service,
            ocr_service=self.ocr_service
        )
        self.search_tool = SearchTool(
            vector_store=self.vector_store,
            embedding_service=self.embedding_service,
            document_store=self.document_store
        )
        self.generative_tool = GenerativeTool(
            llm_service=self.llm_service,
            search_tool=self.search_tool
        )

        # Track processing status
        self.processing_status = {}
        
        # Document cache for quick access
        self.document_cache = {}
        logger.info("Content Organizer MCP Server initialized successfully!")

    def run_async(self, coro):
        """Helper to run async functions in Gradio"""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        if loop.is_running():
            # If loop is already running, create a task
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, coro)
                return future.result()
        else:
            return loop.run_until_complete(coro)

    async def ingest_document_async(self, file_path: str, file_type: str) -> Dict[str, Any]:
        """MCP Tool: Ingest and process a document"""
        try:
            task_id = str(uuid.uuid4())
            self.processing_status[task_id] = {"status": "processing", "progress": 0}
            result = await self.ingestion_tool.process_document(file_path, file_type, task_id)
            if result.get("success"):
                self.processing_status[task_id] = {"status": "completed", "progress": 100}
                doc_id = result.get("document_id")
                if doc_id:
                    doc = await self.document_store.get_document(doc_id)
                    if doc:
                        self.document_cache[doc_id] = doc
                return result
            else:
                self.processing_status[task_id] = {"status": "failed", "error": result.get("error")}
                return result
        except Exception as e:
            logger.error(f"Document ingestion failed: {str(e)}")
            return {"success": False, "error": str(e), "message": "Failed to process document"}

    async def get_document_content_async(self, document_id: str) -> Optional[str]:
        """Get document content by ID"""
        try:
            # Check cache first
            if document_id in self.document_cache:
                return self.document_cache[document_id].content
            
            # Get from store
            doc = await self.document_store.get_document(document_id)
            if doc:
                self.document_cache[document_id] = doc
                return doc.content
            return None
        except Exception as e:
            logger.error(f"Error getting document content: {str(e)}")
            return None

    async def semantic_search_async(self, query: str, top_k: int = 5, filters: Optional[Dict] = None) -> Dict[str, Any]:
        """MCP Tool: Perform semantic search"""
        try:
            results = await self.search_tool.search(query, top_k, filters)
            return {"success": True, "query": query, "results": [result.to_dict() for result in results], "total_results": len(results)}
        except Exception as e:
            logger.error(f"Semantic search failed: {str(e)}")
            return {"success": False, "error": str(e), "query": query, "results": []}

    async def summarize_content_async(self, content: str = None, document_id: str = None, style: str = "concise") -> Dict[str, Any]:
        try:
            if document_id and document_id != "none":
                content = await self.get_document_content_async(document_id)
                if not content:
                    return {"success": False, "error": f"Document {document_id} not found"}
            if not content or not content.strip():
                return {"success": False, "error": "No content provided for summarization"}
            max_content_length = 4000
            if len(content) > max_content_length:
                content = content[:max_content_length] + "..."
            summary = await self.generative_tool.summarize(content, style)
            return {"success": True, "summary": summary, "original_length": len(content), "summary_length": len(summary), "style": style, "document_id": document_id}
        except Exception as e:
            logger.error(f"Summarization failed: {str(e)}")
            return {"success": False, "error": str(e)}

    async def generate_tags_async(self, content: str = None, document_id: str = None, max_tags: int = 5) -> Dict[str, Any]:
        """MCP Tool: Generate tags for content"""
        try:
            if document_id and document_id != "none":
                content = await self.get_document_content_async(document_id)
                if not content:
                    return {"success": False, "error": f"Document {document_id} not found"}
            if not content or not content.strip():
                return {"success": False, "error": "No content provided for tag generation"}
            tags = await self.generative_tool.generate_tags(content, max_tags)
            if document_id and document_id != "none" and tags:
                await self.document_store.update_document_metadata(document_id, {"tags": tags})
            return {"success": True, "tags": tags, "content_length": len(content), "document_id": document_id}
        except Exception as e:
            logger.error(f"Tag generation failed: {str(e)}")
            return {"success": False, "error": str(e)}

    async def answer_question_async(self, question: str, context_filter: Optional[Dict] = None) -> Dict[str, Any]:
        try:
            search_results = await self.search_tool.search(question, top_k=5, filters=context_filter)
            if not search_results:
                return {"success": False, "error": "No relevant context found in your documents. Please make sure you have uploaded relevant documents.", "question": question}
            answer = await self.generative_tool.answer_question(question, search_results)
            return {"success": True, "question": question, "answer": answer, "sources": [result.to_dict() for result in search_results], "confidence": "high" if len(search_results) >= 3 else "medium"}
        except Exception as e:
            logger.error(f"Question answering failed: {str(e)}")
            return {"success": False, "error": str(e), "question": question}

    def list_documents_sync(self, limit: int = 100, offset: int = 0) -> Dict[str, Any]:
        try:
            documents = self.run_async(self.document_store.list_documents(limit, offset))
            return {"success": True, "documents": [doc.to_dict() for doc in documents], "total": len(documents)}
        except Exception as e:
            return {"success": False, "error": str(e)}

mcp_server = ContentOrganizerMCPServer()

def get_document_list():
    try:
        result = mcp_server.list_documents_sync(limit=100)
        if result["success"]:
            if result["documents"]:
                doc_list_str = "📚 Documents in Library:\n\n"
                for i, doc_item in enumerate(result["documents"], 1):
                    doc_list_str += f"{i}. {doc_item['filename']} (ID: {doc_item['id'][:8]}...)\n"
                    doc_list_str += f"   Type: {doc_item['doc_type']}, Size: {doc_item['file_size']} bytes\n"
                    if doc_item.get('tags'):
                        doc_list_str += f"   Tags: {', '.join(doc_item['tags'])}\n"
                    doc_list_str += f"   Created: {doc_item['created_at'][:10]}\n\n"
                return doc_list_str
            else:
                return "No documents in library yet. Upload some documents to get started!"
        else:
            return f"Error loading documents: {result['error']}"
    except Exception as e:
        return f"Error: {str(e)}"

def get_document_choices():
    try:
        result = mcp_server.list_documents_sync(limit=100)
        if result["success"] and result["documents"]:
            choices = [(f"{doc['filename']} ({doc['id'][:8]}...)", doc['id']) for doc in result["documents"]]
            logger.info(f"Generated {len(choices)} document choices")
            return choices
        return []
    except Exception as e:
        logger.error(f"Error getting document choices: {str(e)}")
        return []

def refresh_library():
    doc_list_refreshed = get_document_list()
    doc_choices_refreshed = get_document_choices()
    logger.info(f"Refreshing library. Found {len(doc_choices_refreshed)} choices.")
    return (
        doc_list_refreshed,
        gr.update(choices=doc_choices_refreshed),
        gr.update(choices=doc_choices_refreshed),
        gr.update(choices=doc_choices_refreshed)
    )

def upload_and_process_file(file):
    if file is None:
        doc_list_initial = get_document_list()
        doc_choices_initial = get_document_choices()
        return (
            "No file uploaded", "", doc_list_initial,
            gr.update(choices=doc_choices_initial),
            gr.update(choices=doc_choices_initial),
            gr.update(choices=doc_choices_initial)
        )
    file_path = file.name if hasattr(file, 'name') else str(file)
    filename = Path(file_path).name
    file_type = Path(file_path).suffix.lower().strip('.') # Ensure suffix is clean
    
    from services.task_service import task_service
    task = task_service.create_task(
        task_type="document_ingestion",
        source="UI Upload",
        name=filename,
        summary=f"Ingesting and indexing {file_type} document"
    )

    try:
        logger.info(f"Processing file: {file_path}, type: {file_type}")
        result = mcp_server.run_async(mcp_server.ingest_document_async(file_path, file_type))
        
        doc_list_updated = get_document_list()
        doc_choices_updated = get_document_choices()

        if result["success"]:
            chunks_cnt = result.get('chunks_created', 0)
            doc_id_val = result.get('document_id', '')
            task_service.complete_task(
                task_id=task.id,
                summary=f"Successfully indexed: {chunks_cnt} chunks created (ID: {doc_id_val[:8]}...)",
                output=result
            )
            return (
                f"✅ Success: {result['message']}\nDocument ID: {result['document_id']}\nChunks created: {result['chunks_created']}",
                result["document_id"],
                doc_list_updated,
                gr.update(choices=doc_choices_updated),
                gr.update(choices=doc_choices_updated),
                gr.update(choices=doc_choices_updated)
            )
        else:
            task_service.fail_task(
                task_id=task.id,
                error=result.get('error', 'Unknown error')
            )
            return (
                f"❌ Error: {result.get('error', 'Unknown error')}", "",
                doc_list_updated,
                gr.update(choices=doc_choices_updated),
                gr.update(choices=doc_choices_updated),
                gr.update(choices=doc_choices_updated)
            )
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        task_service.fail_task(task.id, error=str(e))
        doc_list_error = get_document_list()
        doc_choices_error = get_document_choices()
        return (
            f"❌ Error: {str(e)}", "",
            doc_list_error,
            gr.update(choices=doc_choices_error),
            gr.update(choices=doc_choices_error),
            gr.update(choices=doc_choices_error)
        )

def perform_search(query, top_k):
    if not query.strip():
        return "Please enter a search query"
    try:
        result = mcp_server.run_async(mcp_server.semantic_search_async(query, int(top_k)))
        if result["success"]:
            if result["results"]:
                output_str = f"🔍 Found {result['total_results']} results for: '{query}'\n\n"
                for i, res_item in enumerate(result["results"], 1):
                    output_str += f"Result {i}:\n"
                    output_str += f"📊 Relevance Score: {res_item['score']:.3f}\n"
                    output_str += f"📄 Content: {res_item['content'][:300]}...\n"
                    if 'document_filename' in res_item.get('metadata', {}):
                        output_str += f"📁 Source: {res_item['metadata']['document_filename']}\n"
                    output_str += f"🔗 Document ID: {res_item.get('document_id', 'Unknown')}\n"
                    output_str += "-" * 80 + "\n\n"
                return output_str
            else:
                return f"No results found for: '{query}'\n\nMake sure you have uploaded relevant documents first."
        else:
            return f"❌ Search failed: {result['error']}"
    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        return f"❌ Error: {str(e)}"

def summarize_document(doc_choice, custom_text, style):
    try:
        logger.info(f"Summarize called with doc_choice: {doc_choice}, type: {type(doc_choice)}")
        document_id = doc_choice if doc_choice and doc_choice != "none" and doc_choice != "" else None
        
        if custom_text and custom_text.strip():
            logger.info("Using custom text for summarization")
            result = mcp_server.run_async(mcp_server.summarize_content_async(content=custom_text, style=style))
        elif document_id:
            logger.info(f"Summarizing document: {document_id}")
            result = mcp_server.run_async(mcp_server.summarize_content_async(document_id=document_id, style=style))
        else:
            return "Please select a document from the dropdown or enter text to summarize"
        
        if result["success"]:
            output_str = f"📝 Summary ({style} style):\n\n{result['summary']}\n\n"
            output_str += f"📊 Statistics:\n"
            output_str += f"- Original length: {result['original_length']} characters\n"
            output_str += f"- Summary length: {result['summary_length']} characters\n"
            output_str += f"- Compression ratio: {(1 - result['summary_length']/max(1,result['original_length']))*100:.1f}%\n" # Avoid division by zero
            if result.get('document_id'):
                output_str += f"- Document ID: {result['document_id']}\n"
            return output_str
        else:
            return f"❌ Summarization failed: {result['error']}"
    except Exception as e:
        logger.error(f"Summarization error: {str(e)}")
        return f"❌ Error: {str(e)}"

def generate_tags_for_document(doc_choice, custom_text, max_tags):
    try:
        logger.info(f"Generate tags called with doc_choice: {doc_choice}, type: {type(doc_choice)}")
        document_id = doc_choice if doc_choice and doc_choice != "none" and doc_choice != "" else None

        if custom_text and custom_text.strip():
            logger.info("Using custom text for tag generation")
            result = mcp_server.run_async(mcp_server.generate_tags_async(content=custom_text, max_tags=int(max_tags)))
        elif document_id:
            logger.info(f"Generating tags for document: {document_id}")
            result = mcp_server.run_async(mcp_server.generate_tags_async(document_id=document_id, max_tags=int(max_tags)))
        else:
            return "Please select a document from the dropdown or enter text to generate tags"
        
        if result["success"]:
            tags_str = ", ".join(result["tags"])
            output_str = f"🏷️ Generated Tags:\n\n{tags_str}\n\n"
            output_str += f"📊 Statistics:\n"
            output_str += f"- Content length: {result['content_length']} characters\n"
            output_str += f"- Number of tags: {len(result['tags'])}\n"
            if result.get('document_id'):
                output_str += f"- Document ID: {result['document_id']}\n"
                output_str += f"\n✅ Tags have been saved to the document."
            return output_str
        else:
            return f"❌ Tag generation failed: {result['error']}"
    except Exception as e:
        logger.error(f"Tag generation error: {str(e)}")
        return f"❌ Error: {str(e)}"

def ask_question(question):
    if not question.strip():
        return "Please enter a question"
    try:
        result = mcp_server.run_async(mcp_server.answer_question_async(question))
        if result["success"]:
            output_str = f"❓ Question: {result['question']}\n\n"
            output_str += f"💡 Answer:\n{result['answer']}\n\n"
            output_str += f"🎯 Confidence: {result['confidence']}\n\n"
            output_str += f"📚 Sources Used ({len(result['sources'])}):\n"
            for i, source_item in enumerate(result['sources'], 1):
                filename = source_item.get('metadata', {}).get('document_filename', 'Unknown')
                output_str += f"\n{i}. 📄 {filename}\n"
                output_str += f"   📝 Excerpt: {source_item['content'][:150]}...\n"
                output_str += f"   📊 Relevance: {source_item['score']:.3f}\n"
            return output_str
        else:
            return f"❌ {result.get('error', 'Failed to answer question')}"
    except Exception as e:
        return f"❌ Error: {str(e)}"

def delete_document_from_library(document_id):
    if not document_id:
        doc_list_current = get_document_list()
        doc_choices_current = get_document_choices()
        return (
            "No document selected to delete.",
            doc_list_current,
            gr.update(choices=doc_choices_current),
            gr.update(choices=doc_choices_current),
            gr.update(choices=doc_choices_current)
        )
    try:
        delete_doc_store_result = mcp_server.run_async(mcp_server.document_store.delete_document(document_id))
        delete_vec_store_result = mcp_server.run_async(mcp_server.vector_store.delete_document(document_id))

        msg = ""
        if delete_doc_store_result:
            msg += f"🗑️ Document {document_id[:8]}... deleted from document store. "
        else:
            msg += f"❌ Failed to delete document {document_id[:8]}... from document store. "
        
        if delete_vec_store_result:
             msg += "Embeddings deleted from vector store."
        else:
             msg += "Failed to delete embeddings from vector store (or no embeddings existed)."


        doc_list_updated = get_document_list()
        doc_choices_updated = get_document_choices()
        return (
            msg,
            doc_list_updated,
            gr.update(choices=doc_choices_updated),
            gr.update(choices=doc_choices_updated),
            gr.update(choices=doc_choices_updated)
        )
    except Exception as e:
        logger.error(f"Error deleting document: {str(e)}")
        doc_list_error = get_document_list()
        doc_choices_error = get_document_choices()
        return (
            f"❌ Error deleting document: {str(e)}",
            doc_list_error,
            gr.update(choices=doc_choices_error),
            gr.update(choices=doc_choices_error),
            gr.update(choices=doc_choices_error)
        )

def extract_structured_data(file, doc_type="unknown", hints="{}"):
    """
    MCP Tool & UI: Extract structured financial data (transactions, assets, folios, balances)
    using deterministic bank parsers with Gemini Vision fallback and template synthesis.
    """
    if file is None:
        return json.dumps({"success": False, "error": "No file uploaded"}, indent=2)
        
    file_path = file.name if hasattr(file, 'name') else str(file)
    filename = Path(file_path).name

    from services.task_service import task_service
    task = task_service.create_task(
        task_type="financial_extraction",
        source="UI Extraction / MCP",
        name=filename,
        summary=f"Extracting financial statements (hint: {doc_type})"
    )

    try:
        with open(file_path, "rb") as f:
            file_bytes = f.read()
            
        parsed_hints = {}
        if hints:
            try:
                parsed_hints = json.loads(hints) if isinstance(hints, str) else hints
            except Exception:
                parsed_hints = {"raw_hints": str(hints)}

        from services.parsers import ParserRegistry
        registry = ParserRegistry()
        result = registry.extract(
            file_bytes=file_bytes,
            filename=filename,
            doc_type=doc_type or "unknown",
            hints=parsed_hints
        )

        if result.success:
            summary_msg = f"Extracted {len(result.transactions)} transactions, {len(result.assets)} assets ({result.template_id})"
            task_service.complete_task(
                task_id=task.id,
                summary=summary_msg,
                output=result.model_dump(),
                duration_ms=result.processing_time_ms
            )
        else:
            task_service.fail_task(
                task_id=task.id,
                error=result.error or "Extraction failed",
                duration_ms=result.processing_time_ms
            )

        return json.dumps(result.model_dump(), indent=2)
    except Exception as e:
        logger.error(f"Structured extraction failed: {str(e)}")
        task_service.fail_task(task.id, error=str(e))
        return json.dumps({"success": False, "error": str(e)}, indent=2)

def extract_structured_data_ui(file, doc_type="unknown", hints="{}"):
    """
    UI handler for structured extraction: parses statement, populates transactions table, and outputs JSON.
    """
    if file is None:
        return "⚠️ **No file selected.** Please upload a PDF or CSV statement to begin.", [], "{}"
    
    raw_json = extract_structured_data(file, doc_type, hints)
    try:
        data = json.loads(raw_json)
        if not data.get("success"):
            err = data.get("error", "Extraction failed")
            return f"❌ **Extraction Failed**: {err}", [], raw_json
        
        txs = data.get("transactions", [])
        tx_rows = []
        for t in txs:
            amt = t.get("amount")
            amt_str = f"${float(amt):,.2f}" if amt is not None else "-"
            bal = t.get("balance")
            bal_str = f"${float(bal):,.2f}" if bal is not None else "-"
            tx_rows.append([
                t.get("date", "-"),
                t.get("description", "-"),
                amt_str,
                t.get("transaction_type", "debit"),
                t.get("category", "General"),
                bal_str
            ])
        
        template = data.get("template_id", "unknown")
        duration = data.get("processing_time_ms", 0)
        is_fast = "gemini" not in template.lower()
        engine_label = "🟢 Fast-Path Deterministic" if is_fast else "🟣 Gemini AI Vision Fallback"
        status_md = f"### ✅ Successfully Extracted **{len(txs)} Transactions**\n**Engine:** `{template}` ({engine_label}) | **Latency:** `{duration}ms` | **Account:** `{data.get('account_number', 'N/A')}` | **Assets:** `{len(data.get('assets', []))}`"
        return status_md, tx_rows, raw_json
    except Exception as e:
        return f"❌ **Parsing Error**: {str(e)}", [], raw_json

def get_templates_rows():
    try:
        from services.parsers import ParserRegistry
        registry = ParserRegistry()
        templates = registry.list_templates()
        rows = []
        for t in templates:
            ptype = t.get("type", "deterministic")
            badge = "🟢 Fast-Path Regex (20-50ms)" if ptype == "deterministic" else "🟣 AI Vision Synthesizer"
            rows.append([
                t.get("template_id", ""),
                t.get("name", ""),
                t.get("category", ""),
                ptype,
                t.get("version", "1.0"),
                badge
            ])
        return rows
    except Exception as e:
        logger.error(f"Error loading templates: {e}")
        return []

def get_task_rows(status_filter="All", source_filter="All"):
    from services.task_service import task_service
    tasks = task_service.list_tasks(limit=100, status_filter=status_filter, source_filter=source_filter)
    rows = []
    for t in tasks:
        if t.status == "completed":
            status_badge = "🟢 Completed"
        elif t.status == "failed":
            status_badge = "🔴 Failed"
        else:
            status_badge = "🟡 Processing"
            
        dur_str = f"{t.duration_ms}ms" if t.duration_ms is not None else "-"
        created_display = t.created_at[:19].replace("T", " ") if t.created_at else "-"
        
        type_icon = {
            "financial_extraction": "💳 Extraction",
            "document_ingestion": "📄 Ingestion",
            "semantic_search": "🔍 Search",
            "summarization": "📝 Summary",
            "qa": "❓ Q&A"
        }.get(t.task_type, t.task_type)
        
        rows.append([
            created_display,
            t.id,
            type_icon,
            t.source,
            t.name,
            status_badge,
            dur_str,
            t.summary
        ])
    return rows

def get_task_choices_list():
    from services.task_service import task_service
    tasks = task_service.list_tasks(limit=100)
    choices = []
    for t in tasks:
        created_display = t.created_at[:19].replace("T", " ") if t.created_at else ""
        icon = "🟢" if t.status == "completed" else ("🔴" if t.status == "failed" else "🟡")
        label = f"{icon} [{created_display}] {t.name} ({t.source})"
        choices.append((label, t.id))
    return choices

def view_task_details(task_id):
    if not task_id:
        return "Select a task above to inspect details and outputs."
    from services.task_service import task_service
    task = task_service.get_task(task_id)
    if not task:
        return f"Task '{task_id}' not found."
    return json.dumps(task.model_dump(), indent=2)

def refresh_tasks_ui(status_filter="All", source_filter="All"):
    table_data = get_task_rows(status_filter, source_filter)
    choices = get_task_choices_list()
    first_choice_id = choices[0][1] if choices else None
    first_detail = view_task_details(first_choice_id) if first_choice_id else "No tasks recorded yet."
    return (
        table_data,
        gr.update(choices=choices, value=first_choice_id),
        first_detail
    )

HEADER_HTML = """
<div style="background: linear-gradient(135deg, #090d16 0%, #111827 100%); border: 1px solid #1e293b; border-radius: 12px; padding: 18px 24px; margin-bottom: 20px; display: flex; justify-content: space-between; align-items: center; box-shadow: 0 4px 20px rgba(0,0,0,0.4);">
  <div style="display: flex; align-items: center; gap: 16px;">
    <div style="width: 46px; height: 46px; border-radius: 12px; background: linear-gradient(135deg, #3b82f6 0%, #6366f1 100%); display: flex; align-items: center; justify-content: center; font-size: 24px; box-shadow: 0 0 15px rgba(59, 130, 246, 0.4);">
      🧠
    </div>
    <div>
      <div style="display: flex; align-items: center; gap: 10px;">
        <h1 style="color: #f8fafc; font-size: 22px; font-weight: 700; margin: 0; letter-spacing: -0.02em;">Vgurukool Intelligent Content Organizer</h1>
        <span style="background: #1e293b; color: #94a3b8; font-size: 11px; padding: 2px 8px; border-radius: 9999px; border: 1px solid #334155; font-family: monospace;">v2.2.0</span>
      </div>
      <p style="color: #94a3b8; font-size: 13px; margin: 4px 0 0 0;">Unified RAG Knowledge Engine & Deterministic Financial Extraction Studio</p>
    </div>
  </div>
  <div style="display: flex; align-items: center; gap: 12px;">
    <div style="background: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 8px; padding: 6px 12px; font-size: 12px; color: #34d399; display: flex; align-items: center; gap: 6px;">
      <span style="width: 8px; height: 8px; border-radius: 50%; background: #10b981; display: inline-block; box-shadow: 0 0 8px #10b981;"></span>
      <span>cnoe-ref-impl (us-east-2)</span>
    </div>
    <div style="background: rgba(99, 102, 241, 0.1); border: 1px solid rgba(99, 102, 241, 0.3); border-radius: 8px; padding: 6px 12px; font-size: 12px; color: #818cf8; display: flex; align-items: center; gap: 6px;">
      <span>⚡ Fast-Path: 20-50ms</span>
    </div>
    <div id="auth-status-container">
      <a id="sso-auth-btn" href="/login" style="background: linear-gradient(135deg, #4f46e5 0%, #3b82f6 100%); color: white; text-decoration: none; padding: 8px 16px; border-radius: 8px; font-size: 13px; font-weight: 600; display: flex; align-items: center; gap: 8px; box-shadow: 0 2px 10px rgba(79, 70, 229, 0.3);">
        <span>🔐</span>
        <span>Sign In with Keycloak SSO</span>
      </a>
    </div>
  </div>
</div>
<script>
(function() {
  fetch('/api/v1/auth/me')
    .then(r => r.json())
    .then(data => {
      const container = document.getElementById('auth-status-container');
      if (container && data.authenticated) {
        const u = data.user;
        const name = u.name || u.preferred_username || u.email;
        container.innerHTML = `
          <div style="display: flex; align-items: center; gap: 8px; background: #1e293b; border: 1px solid #334155; padding: 6px 14px; border-radius: 8px; font-size: 13px; color: #f1f5f9;">
            <span>👤</span>
            <span style="font-weight: 600;">${name}</span>
            <span style="color: #64748b; font-size: 11px;">(cnoe)</span>
            <a href="/logout" style="margin-left: 6px; color: #f87171; text-decoration: none; font-size: 12px; font-weight: 500;">Logout</a>
          </div>
        `;
      }
    })
    .catch(() => {});
})();
</script>
"""

OVERVIEW_HTML = """
<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 16px; margin-bottom: 24px;">
  <div style="background: #1e293b; border: 1px solid #334155; border-radius: 12px; padding: 20px;">
    <div style="color: #94a3b8; font-size: 12px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em;">Fast-Path Latency</div>
    <div style="font-size: 28px; font-weight: 800; color: #38bdf8; margin-top: 6px;">20 - 50 <span style="font-size: 16px; font-weight: 500; color: #94a3b8;">ms</span></div>
    <div style="color: #64748b; font-size: 12px; margin-top: 4px;">Deterministic regex parsers</div>
  </div>
  <div style="background: #1e293b; border: 1px solid #334155; border-radius: 12px; padding: 20px;">
    <div style="color: #94a3b8; font-size: 12px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em;">Supported Formats</div>
    <div style="font-size: 28px; font-weight: 800; color: #34d399; margin-top: 6px;">5+ <span style="font-size: 16px; font-weight: 500; color: #94a3b8;">Templates</span></div>
    <div style="color: #64748b; font-size: 12px; margin-top: 4px;">Chase, BofA, HDFC, CAS, Citi</div>
  </div>
  <div style="background: #1e293b; border: 1px solid #334155; border-radius: 12px; padding: 20px;">
    <div style="color: #94a3b8; font-size: 12px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em;">AI Fallback Engine</div>
    <div style="font-size: 28px; font-weight: 800; color: #c084fc; margin-top: 6px;">Gemini 2.5 <span style="font-size: 16px; font-weight: 500; color: #94a3b8;">Flash</span></div>
    <div style="color: #64748b; font-size: 12px; margin-top: 4px;">With automatic template synthesizer</div>
  </div>
  <div style="background: #1e293b; border: 1px solid #334155; border-radius: 12px; padding: 20px;">
    <div style="color: #94a3b8; font-size: 12px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em;">Identity & Auth</div>
    <div style="font-size: 28px; font-weight: 800; color: #fbbf24; margin-top: 6px;">Keycloak <span style="font-size: 16px; font-weight: 500; color: #94a3b8;">SSO</span></div>
    <div style="color: #64748b; font-size: 12px; margin-top: 4px;">Realm cnoe | Client vgurukool-apps</div>
  </div>
</div>

<div style="background: #182234; border: 1px solid #1e293b; border-radius: 12px; padding: 20px; margin-bottom: 20px;">
  <h3 style="color: #f8fafc; font-size: 16px; font-weight: 700; margin: 0 0 12px 0;">⚡ Dual-Engine Architectural Pipeline</h3>
  <div style="display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: 12px; background: #0f172a; border: 1px solid #1e293b; border-radius: 8px; padding: 16px;">
    <div style="text-align: center; flex: 1; min-width: 140px;">
      <div style="font-size: 20px;">📄</div>
      <div style="font-size: 13px; font-weight: 600; color: #f1f5f9; margin-top: 4px;">Document Ingestion</div>
      <div style="font-size: 11px; color: #94a3b8;">PDF / CSV / Scans</div>
    </div>
    <div style="color: #64748b; font-size: 18px;">➔</div>
    <div style="text-align: center; flex: 1; min-width: 140px;">
      <div style="font-size: 20px;">🎯</div>
      <div style="font-size: 13px; font-weight: 600; color: #38bdf8; margin-top: 4px;">Signature Matcher</div>
      <div style="font-size: 11px; color: #94a3b8;">Inspects headers & text</div>
    </div>
    <div style="color: #64748b; font-size: 18px;">➔</div>
    <div style="text-align: center; flex: 1.2; min-width: 160px; background: rgba(52, 211, 153, 0.1); border: 1px dashed rgba(52, 211, 153, 0.4); border-radius: 6px; padding: 8px;">
      <div style="font-size: 13px; font-weight: 700; color: #34d399;">Fast-Path (20-50ms)</div>
      <div style="font-size: 11px; color: #a7f3d0;">Deterministic Bank Parsers</div>
    </div>
    <div style="color: #64748b; font-size: 18px;">OR</div>
    <div style="text-align: center; flex: 1.2; min-width: 160px; background: rgba(192, 132, 252, 0.1); border: 1px dashed rgba(192, 132, 252, 0.4); border-radius: 6px; padding: 8px;">
      <div style="font-size: 13px; font-weight: 700; color: #c084fc;">AI Fallback (2-4s)</div>
      <div style="font-size: 11px; color: #e9d5ff;">Gemini Vision + Synthesizer</div>
    </div>
    <div style="color: #64748b; font-size: 18px;">➔</div>
    <div style="text-align: center; flex: 1; min-width: 140px;">
      <div style="font-size: 20px;">📊</div>
      <div style="font-size: 13px; font-weight: 600; color: #f1f5f9; margin-top: 4px;">Task Storage</div>
      <div style="font-size: 11px; color: #94a3b8;">/app/data/tasks/ (PVC)</div>
    </div>
  </div>
</div>
"""

CUSTOM_CSS = """
.gradio-container {
    max-width: 1400px !important;
    margin: auto !important;
}
.tab-nav button {
    font-weight: 600 !important;
    font-size: 14px !important;
}
"""

def create_gradio_interface():
    with gr.Blocks(title="🧠 Vgurukool Intelligent Content Organizer", theme=gr.themes.Soft(primary_hue="indigo", neutral_hue="slate"), css=CUSTOM_CSS) as interface:
        gr.HTML(HEADER_HTML)
        
        with gr.Tabs():
            # Tab 1: Overview & Metrics
            with gr.Tab("📊 Overview & Architecture"):
                gr.HTML(OVERVIEW_HTML)
                gr.Markdown("""### 🔌 Protocol & Machine-to-Machine Endpoints
- **REST Extraction**: `POST /api/v1/extract` (Exempt from auth redirects, optimized for Dhana Lakshmi & in-cluster services)
- **Task Query**: `GET /api/v1/tasks` & `GET /api/v1/tasks/{task_id}`
- **Template Query**: `GET /api/v1/templates`
- **MCP Endpoints**: `/gradio_api/mcp/sse` & `/gradio_api/mcp/messages/`
- **SSO Authentication**: `/login` (Keycloak PKCE), `/callback`, `/logout`, `/api/v1/auth/me`
                """)

            # Tab 2: Financial Studio
            with gr.Tab("💳 Financial Extraction Studio"):
                gr.Markdown("""### 🏦 High-Performance Financial Document Extraction
Extract structured statements, transactions, accounts, and investment folios with deterministic microsecond fast-path.
                """)
                with gr.Row():
                    with gr.Column(scale=1):
                        extract_file_input = gr.File(label="Upload Statement (PDF or CSV)", file_types=[".pdf", ".csv"])
                        extract_doctype_dropdown = gr.Dropdown(
                            choices=["unknown", "bank_statement", "mutual_fund_cas", "fixed_deposit", "csv_statement"],
                            value="unknown",
                            label="Document Type Hint"
                        )
                        extract_hints_input = gr.Textbox(
                            label="Hints / Parameters (JSON)",
                            value='{"bank": "auto", "password": ""}',
                            lines=2
                        )
                        extract_btn_action = gr.Button("⚡ Run Structured Extraction", variant="primary", size="lg")
                    
                    with gr.Column(scale=2):
                        extract_status_md = gr.Markdown("### 📄 Ready for Statement Extraction\nUpload a statement on the left and click **Run Structured Extraction**.")
                        with gr.Tabs():
                            with gr.Tab("📑 Extracted Transactions"):
                                extract_tx_table = gr.Dataframe(
                                    headers=["Date", "Description", "Amount", "Type", "Category", "Balance"],
                                    datatype=["str", "str", "str", "str", "str", "str"],
                                    value=[],
                                    interactive=False,
                                    wrap=True
                                )
                            with gr.Tab("💻 Structured JSON Output"):
                                extract_json_display = gr.Code(label="Full JSON Schema Payload", language="json", lines=22)

            # Tab 3: Activity & Task Monitor
            with gr.Tab("📋 Activity & Tasks"):
                gr.Markdown("""### 📊 Centralized Task & Request Monitor
Real-time audit log of all document operations: browser uploads, external REST API calls from Dhana Lakshmi, and MCP requests.
                """)
                with gr.Row():
                    filter_status = gr.Dropdown(
                        choices=["All", "completed", "processing", "failed"],
                        value="All",
                        label="Status Filter",
                        scale=1
                    )
                    filter_source = gr.Dropdown(
                        choices=["All", "UI Upload", "REST API", "UI Extraction / MCP", "Google Drive", "S3 Inbox"],
                        value="All",
                        label="Source Filter",
                        scale=1
                    )
                    task_refresh_btn = gr.Button("🔄 Refresh Activity", variant="secondary", scale=1)

                init_tasks = get_task_rows()
                init_choices = get_task_choices_list()
                first_choice_id = init_choices[0][1] if init_choices else None
                init_detail = view_task_details(first_choice_id) if first_choice_id else "No tasks found."

                tasks_table = gr.Dataframe(
                    headers=["Time (UTC)", "Task ID", "Type", "Source", "Document / Target", "Status", "Duration", "Summary"],
                    datatype=["str", "str", "str", "str", "str", "str", "str", "str"],
                    value=init_tasks,
                    interactive=False,
                    wrap=True
                )

                gr.Markdown("---")
                gr.Markdown("### 🔍 Task Inspector & Detailed Output")
                with gr.Row():
                    task_selector_dropdown = gr.Dropdown(
                        choices=init_choices,
                        label="Select Task to Inspect",
                        value=first_choice_id,
                        scale=3
                    )
                task_detail_output = gr.Code(
                    label="Task Details & Payload Output (JSON)",
                    language="json",
                    lines=20,
                    value=init_detail
                )

            # Tab 4: Document Intelligence & Q&A
            with gr.Tab("🧠 Document Intelligence & Q&A"):
                with gr.Tabs():
                    with gr.Tab("🔍 Semantic Search & RAG Q&A"):
                        with gr.Row():
                            with gr.Column():
                                gr.Markdown("#### 🔍 Vector Semantic Search")
                                search_query_input = gr.Textbox(label="Search Query", placeholder="Search document embeddings...", lines=2)
                                search_top_k_slider = gr.Slider(label="Max Results", minimum=1, maximum=10, value=5, step=1)
                                search_btn_action = gr.Button("🔍 Search Knowledge Base", variant="primary")
                                search_output_display = gr.Textbox(label="Search Results", lines=12, placeholder="Search results will appear here...")
                            with gr.Column():
                                gr.Markdown("#### ❓ Generative Q&A with Gemini")
                                qa_question_input = gr.Textbox(label="Ask Questions About Your Knowledge Base", placeholder="Ask anything about indexed documents...", lines=2)
                                qa_btn_action = gr.Button("💡 Get Answer with Citations", variant="primary")
                                qa_output_display = gr.Textbox(label="AI Answer & Sources", lines=12, placeholder="Answer will appear here with citations...")

                    with gr.Tab("📚 Document Library & Ingestion"):
                        with gr.Row():
                            with gr.Column(scale=1):
                                gr.Markdown("#### 📤 Ingest & Index New Document")
                                file_input_upload = gr.File(label="Select Document", file_types=[".txt", ".pdf", ".docx", ".csv", ".json", ".md"])
                                upload_btn_process = gr.Button("📤 Process and Index", variant="primary")
                                upload_output_display = gr.Textbox(label="Ingestion Result", lines=4)
                                doc_id_output_display = gr.Textbox(label="Generated Document ID", interactive=False)
                            with gr.Column(scale=2):
                                gr.Markdown("#### 📂 Indexed Documents in Repository")
                                document_list_display = gr.Textbox(label="Documents in Vector Store", value=get_document_list(), lines=12, interactive=False)
                                with gr.Row():
                                    refresh_btn_library = gr.Button("🔄 Refresh Library", variant="secondary")
                                    delete_doc_dropdown_visible = gr.Dropdown(choices=get_document_choices(), label="Select Document to Delete", value=None)
                                    delete_btn = gr.Button("🗑️ Delete", variant="stop")
                                delete_output_display = gr.Textbox(label="Status", lines=2, interactive=False)

                    with gr.Tab("📝 Summarize & Auto-Tag"):
                        with gr.Row():
                            with gr.Column():
                                gr.Markdown("#### 📝 Document Summarization")
                                doc_dropdown_sum_visible = gr.Dropdown(choices=get_document_choices(), label="Select Document to Summarize", value=None)
                                summary_text_input = gr.Textbox(label="Or Paste Raw Text", placeholder="Paste any text to summarize...", lines=4)
                                summary_style_dropdown = gr.Dropdown(choices=["concise", "detailed", "bullet_points", "executive"], value="concise", label="Summary Style")
                                summarize_btn_action = gr.Button("📝 Summarize", variant="primary")
                                summary_output_display = gr.Textbox(label="Summary", lines=10)
                            with gr.Column():
                                gr.Markdown("#### 🏷️ Automated Tag Generator")
                                doc_dropdown_tag_visible = gr.Dropdown(choices=get_document_choices(), label="Select Document to Tag", value=None)
                                tag_text_input = gr.Textbox(label="Or Paste Raw Text", placeholder="Paste text to extract tags...", lines=4)
                                max_tags_slider = gr.Slider(label="Number of Tags", minimum=3, maximum=15, value=5, step=1)
                                tag_btn_action = gr.Button("🏷️ Generate Tags", variant="primary")
                                tag_output_display = gr.Textbox(label="Extracted Tags", lines=6)

            # Tab 5: Template Registry
            with gr.Tab("⚡ Template Registry"):
                gr.Markdown("""### ⚡ Bank & Financial Statement Parsers
High-speed deterministic pattern engines execute directly in-memory without GPU or LLM inference.
Unrecognized documents automatically activate the Gemini Vision fallback and self-learning synthesizer.
                """)
                templates_table = gr.Dataframe(
                    headers=["Template ID", "Name", "Category", "Type", "Version", "Engine"],
                    datatype=["str", "str", "str", "str", "str", "str"],
                    value=get_templates_rows(),
                    interactive=False,
                    wrap=True
                )

        # Wire all reactive events
        all_dropdowns_to_update = [delete_doc_dropdown_visible, doc_dropdown_sum_visible, doc_dropdown_tag_visible]
        refresh_outputs = [document_list_display] + [dd for dd in all_dropdowns_to_update]
        refresh_btn_library.click(fn=refresh_library, outputs=refresh_outputs)
        
        upload_outputs = [upload_output_display, doc_id_output_display, document_list_display] + [dd for dd in all_dropdowns_to_update]
        upload_btn_process.click(upload_and_process_file, inputs=[file_input_upload], outputs=upload_outputs)

        delete_outputs = [delete_output_display, document_list_display] + [dd for dd in all_dropdowns_to_update]
        delete_btn.click(delete_document_from_library, inputs=[delete_doc_dropdown_visible], outputs=delete_outputs)
        
        search_btn_action.click(perform_search, inputs=[search_query_input, search_top_k_slider], outputs=[search_output_display])
        summarize_btn_action.click(summarize_document, inputs=[doc_dropdown_sum_visible, summary_text_input, summary_style_dropdown], outputs=[summary_output_display])
        tag_btn_action.click(generate_tags_for_document, inputs=[doc_dropdown_tag_visible, tag_text_input, max_tags_slider], outputs=[tag_output_display])
        qa_btn_action.click(ask_question, inputs=[qa_question_input], outputs=[qa_output_display])

        # Financial Studio event
        extract_btn_action.click(
            extract_structured_data_ui,
            inputs=[extract_file_input, extract_doctype_dropdown, extract_hints_input],
            outputs=[extract_status_md, extract_tx_table, extract_json_display]
        )

        # Task monitor events
        task_refresh_btn.click(
            refresh_tasks_ui,
            inputs=[filter_status, filter_source],
            outputs=[tasks_table, task_selector_dropdown, task_detail_output]
        )
        filter_status.change(
            refresh_tasks_ui,
            inputs=[filter_status, filter_source],
            outputs=[tasks_table, task_selector_dropdown, task_detail_output]
        )
        filter_source.change(
            refresh_tasks_ui,
            inputs=[filter_status, filter_source],
            outputs=[tasks_table, task_selector_dropdown, task_detail_output]
        )
        task_selector_dropdown.change(
            view_task_details,
            inputs=[task_selector_dropdown],
            outputs=[task_detail_output]
        )

        interface.load(fn=refresh_library, outputs=refresh_outputs)
        return interface           

if __name__ == "__main__":
    import uvicorn
    from fastapi import FastAPI
    from api.auth_api import router as auth_router
    from api.extraction_api import router as extraction_router

    gradio_interface = create_gradio_interface()
    server_name = getattr(config.config, "SERVER_NAME", "0.0.0.0")
    server_port = getattr(config.config, "SERVER_PORT", 7860)

    fastapi_app = FastAPI(
        title="Intelligent Content Organizer & Extraction Engine",
        description="Unified Enterprise RAG knowledge organizer and deterministic financial extraction studio with Keycloak SSO.",
        version="2.2.0"
    )
    fastapi_app.include_router(auth_router)
    fastapi_app.include_router(extraction_router)

    app = gr.mount_gradio_app(fastapi_app, gradio_interface, path="/", mcp_server=True)

    logger.info(f"Starting Unified incorg server v2.2.0 on {server_name}:{server_port}...")
    uvicorn.run(app, host=server_name, port=server_port)