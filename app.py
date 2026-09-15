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
    try:
        file_path = file.name if hasattr(file, 'name') else str(file)
        file_type = Path(file_path).suffix.lower().strip('.') # Ensure suffix is clean
        logger.info(f"Processing file: {file_path}, type: {file_type}")
        result = mcp_server.run_async(mcp_server.ingest_document_async(file_path, file_type))
        
        doc_list_updated = get_document_list()
        doc_choices_updated = get_document_choices()

        if result["success"]:
            return (
                f"✅ Success: {result['message']}\nDocument ID: {result['document_id']}\nChunks created: {result['chunks_created']}",
                result["document_id"],
                doc_list_updated,
                gr.update(choices=doc_choices_updated),
                gr.update(choices=doc_choices_updated),
                gr.update(choices=doc_choices_updated)
            )
        else:
            return (
                f"❌ Error: {result.get('error', 'Unknown error')}", "",
                doc_list_updated,
                gr.update(choices=doc_choices_updated),
                gr.update(choices=doc_choices_updated),
                gr.update(choices=doc_choices_updated)
            )
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
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

def create_gradio_interface():
    with gr.Blocks(title="🧠 Intelligent Content Organizer MCP Agent", theme=gr.themes.Soft()) as interface:
        gr.Markdown("""
        # 🧠 Intelligent Content Organizer MCP Agent
        A powerful MCP (Model Context Protocol) server for intelligent content management with semantic search, 
        summarization, and Q&A capabilities.

        👉 Read the full article here:  
        <a href="https://huggingface.co/blog/Nihal2000/intelligent-content-organizer#empowering-your-data-building-an-intelligent-content-organizer-with-mistral-ai-and-the-model-context-protocol" target="_blank">Building an Intelligent Content Organizer</a>

        ## 🚀 Quick Start:
        1. **Documents in Library** → View your uploaded documents in the "📚 Document Library" tab  
        2. **Upload Documents** → Go to "📄 Upload Documents" tab  
        3. **Search Your Content** → Use "🔍 Search Documents" to find information  
        4. **Get Summaries** → Select any document in "📝 Summarize" tab  
        5. **Generate Tags** → Auto-generate tags for your documents in "🏷️ Generate Tags" tab  
        6. **Ask Questions** → Get answers from your documents in "❓ Ask Questions" tab  
        7. **Delete Documents** → Remove documents from your library in "📚 Document Library" tab  
        8. **Refresh Library** → Click the 🔄 button to refresh the document list  

        ---
        🔗 For using MCP tools in Claude or other MCP clients, use this endpoint in the config file:  
         https://agents-mcp-hackathon-intelligent-content-organizer.hf.space/gradio_api/mcp/sse
        """)


        with gr.Tabs():
            with gr.Tab("📚 Document Library"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Your Document Collection")
                        document_list_display = gr.Textbox(label="Documents in Library", value=get_document_list(), lines=20, interactive=False)
                        refresh_btn_library = gr.Button("🔄 Refresh Library", variant="secondary")
                        delete_doc_dropdown_visible = gr.Dropdown(label="Select Document to Delete", choices=get_document_choices(), value=None, interactive=True, allow_custom_value=False)
                        delete_btn = gr.Button("🗑️ Delete Selected Document", variant="stop")
                        delete_output_display = gr.Textbox(label="Delete Status", visible=True)
            
            with gr.Tab("📄 Upload Documents"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Add Documents to Your Library")
                        file_input_upload = gr.File(label="Select Document to Upload", file_types=[".pdf", ".txt", ".docx", ".png", ".jpg", ".jpeg"], type="filepath")
                        upload_btn_process = gr.Button("🚀 Process & Add to Library", variant="primary", size="lg")
                    with gr.Column():
                        upload_output_display = gr.Textbox(label="Processing Result", lines=6, placeholder="Upload a document to see processing results...")
                        doc_id_output_display = gr.Textbox(label="Document ID", placeholder="Document ID will appear here after processing...")

            with gr.Tab("🔍 Search Documents"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### Search Your Document Library")
                        search_query_input = gr.Textbox(label="What are you looking for?", placeholder="Enter your search query...", lines=2)
                        search_top_k_slider = gr.Slider(label="Number of Results", minimum=1, maximum=20, value=5, step=1)
                        search_btn_action = gr.Button("🔍 Search Library", variant="primary", size="lg")
                    with gr.Column(scale=2):
                        search_output_display = gr.Textbox(label="Search Results", lines=20, placeholder="Search results will appear here...")
            
            with gr.Tab("📝 Summarize"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Generate Document Summaries")
                        doc_dropdown_sum_visible = gr.Dropdown(label="Select Document to Summarize", choices=get_document_choices(), value=None, interactive=True, allow_custom_value=False)
                        summary_text_input = gr.Textbox(label="Or Paste Text to Summarize", placeholder="Paste any text here to summarize...", lines=8)
                        summary_style_dropdown = gr.Dropdown(label="Summary Style", choices=["concise", "detailed", "bullet_points", "executive"], value="concise", info="Choose how you want the summary formatted")
                        summarize_btn_action = gr.Button("📝 Generate Summary", variant="primary", size="lg")
                    with gr.Column():
                        summary_output_display = gr.Textbox(label="Generated Summary", lines=20, placeholder="Summary will appear here...")

            with gr.Tab("🏷️ Generate Tags"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Generate Document Tags")
                        doc_dropdown_tag_visible = gr.Dropdown(label="Select Document to Tag", choices=get_document_choices(), value=None, interactive=True, allow_custom_value=False)
                        tag_text_input = gr.Textbox(label="Or Paste Text to Generate Tags", placeholder="Paste any text here to generate tags...", lines=8)
                        max_tags_slider = gr.Slider(label="Number of Tags", minimum=3, maximum=15, value=5, step=1)
                        tag_btn_action = gr.Button("🏷️ Generate Tags", variant="primary", size="lg")
                    with gr.Column():
                        tag_output_display = gr.Textbox(label="Generated Tags", lines=10, placeholder="Tags will appear here...")

            with gr.Tab("❓ Ask Questions"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("""### Ask Questions About Your Documents
                        The AI will search through all your uploaded documents to find relevant information 
                        and provide comprehensive answers with sources.""")
                        qa_question_input = gr.Textbox(label="Your Question", placeholder="Ask anything about your documents...", lines=3)
                        qa_btn_action = gr.Button("❓ Get Answer", variant="primary", size="lg")
                    with gr.Column():
                        qa_output_display = gr.Textbox(label="AI Answer", lines=20, placeholder="Answer will appear here with sources...")

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

        interface.load(fn=refresh_library, outputs=refresh_outputs)
        return interface           

if __name__ == "__main__":
    gradio_interface = create_gradio_interface()
    server_name = getattr(config.config, "SERVER_NAME", "0.0.0.0")
    server_port = getattr(config.config, "SERVER_PORT", 7860)
    logger.info(f"Starting Gradio MCP server on {server_name}:{server_port}...")
    gradio_interface.launch(
        server_name=server_name,
        server_port=server_port,
        mcp_server=True
    )