"""Real-time RAG (Retrieval Augmented Generation) engine using Pathway."""

import pathway as pw
from typing import Dict, List, Any, Optional
import chromadb
from chromadb.utils import embedding_functions
import openai
from datetime import datetime
import json
import numpy as np
from sentence_transformers import SentenceTransformer

from config.settings import Settings
from utils.logger import setup_logger

logger = setup_logger(__name__)


class RAGEngine:
    """Real-time RAG engine for financial market analysis."""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.openai_client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize ChromaDB for vector storage
        self.chroma_client = chromadb.PersistentClient(path="./data/chroma_db")
        
        # Create collections for different data types
        self.news_collection = self._get_or_create_collection("financial_news")
        self.market_collection = self._get_or_create_collection("market_data")
        self.analysis_collection = self._get_or_create_collection("ai_analysis")
        
        # Document counter for unique IDs
        self.doc_counter = 0
    
    def _get_or_create_collection(self, name: str):
        """Get or create a ChromaDB collection."""
        try:
            return self.chroma_client.get_collection(name)
        except:
            return self.chroma_client.create_collection(
                name=name,
                embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name="all-MiniLM-L6-v2"
                )
            )
    
    def index_documents(self, news_stream: pw.Table) -> pw.Table:
        """Index news documents in real-time for RAG."""
        
        def index_news_document(data: pw.Pointer) -> dict:
            """Index a single news document."""
            try:
                # Create document text for embedding
                doc_text = f"{data.title}\n\n{data.content}"
                
                # Generate unique ID
                doc_id = f"news_{self.doc_counter}_{int(datetime.now().timestamp())}"
                self.doc_counter += 1
                
                # Create metadata
                metadata = {
                    "timestamp": data.timestamp,
                    "source": data.source,
                    "url": data.url,
                    "sentiment_score": float(data.sentiment_score),
                    "relevance_score": float(data.relevance_score),
                    "symbols_mentioned": data.symbols_mentioned,
                    "type": "news"
                }
                
                # Add to ChromaDB
                self.news_collection.add(
                    documents=[doc_text],
                    ids=[doc_id],
                    metadatas=[metadata]
                )
                
                logger.debug(f"📚 Indexed news document: {doc_id}")
                
                return {
                    "document_id": doc_id,
                    "indexed_at": datetime.now().isoformat(),
                    "status": "success"
                }
                
            except Exception as e:
                logger.error(f"Failed to index news document: {str(e)}")
                return {
                    "status": "error",
                    "error": str(e)
                }
        
        # Apply indexing to news stream
        return news_stream.select(
            **pw.this,
            indexing_result=index_news_document(pw.this)
        )
    
    def index_market_data(self, market_stream: pw.Table) -> pw.Table:
        """Index market data for contextual retrieval."""
        
        def index_market_document(data: pw.Pointer) -> dict:
            """Index market data as a document."""
            try:
                processed_data = data.processed_data
                indicators = processed_data.get('indicators', {})
                
                # Create document text for market data
                doc_text = f"""
                Market Data for {data.symbol} at {data.timestamp}
                Price: ${data.price}
                Volume: {data.volume:,}
                Price Change: {indicators.get('price_change', 0):.2f}%
                RSI: {indicators.get('rsi', 50):.2f}
                Volatility: {indicators.get('volatility', 0):.4f}
                Volume Ratio: {indicators.get('volume_ratio', 1):.2f}
                """
                
                doc_id = f"market_{data.symbol}_{int(datetime.now().timestamp())}"
                
                metadata = {
                    "symbol": data.symbol,
                    "timestamp": data.timestamp,
                    "price": float(data.price),
                    "volume": int(data.volume),
                    "type": "market_data",
                    **{f"indicator_{k}": float(v) for k, v in indicators.items()}
                }
                
                self.market_collection.add(
                    documents=[doc_text],
                    ids=[doc_id],
                    metadatas=[metadata]
                )
                
                return {"status": "success", "document_id": doc_id}
                
            except Exception as e:
                logger.error(f"Failed to index market data: {str(e)}")
                return {"status": "error", "error": str(e)}
        
        return market_stream.select(
            **pw.this,
            indexing_result=index_market_document(pw.this)
        )
    
    def explain_anomaly(self, anomaly_data: pw.Pointer) -> str:
        """Generate AI explanation for detected anomaly."""
        try:
            symbol = anomaly_data.symbol
            anomaly_score = anomaly_data.anomaly_score
            processed_data = anomaly_data.processed_data
            
            # Retrieve relevant context
            context = self.retrieve_context(symbol, f"anomaly {symbol} price movement")
            
            # Create explanation prompt
            prompt = self._create_explanation_prompt(
                symbol, anomaly_score, processed_data, context
            )
            
            # Generate explanation using OpenAI
            response = self.openai_client.chat.completions.create(
                model=self.settings.LLM_MODEL,
                messages=[
                    {"role": "system", "content": "You are a financial analyst AI that explains market anomalies clearly and concisely."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.settings.MAX_TOKENS,
                temperature=self.settings.TEMPERATURE
            )
            
            explanation = response.choices[0].message.content.strip()
            
            # Store the analysis for future reference
            self._store_analysis(symbol, anomaly_score, explanation)
            
            return explanation
            
        except Exception as e:
            logger.error(f"Failed to generate anomaly explanation: {str(e)}")
            return f"Unable to generate explanation for {anomaly_data.symbol} anomaly. Please check logs for details."
    
    def retrieve_context(self, symbol: str, query: str, max_results: int = 5) -> Dict[str, List[Dict]]:
        """Retrieve relevant context from indexed documents."""
        try:
            context = {"news": [], "market": []}
            
            # Search news collection
            news_results = self.news_collection.query(
                query_texts=[query],
                n_results=max_results,
                where={"symbols_mentioned": {"$contains": symbol}} if symbol else None
            )
            
            if news_results['documents']:
                context["news"] = self._format_search_results(news_results)
            
            # Search market data collection
            market_results = self.market_collection.query(
                query_texts=[f"{symbol} market data price volume"],
                n_results=max_results,
                where={"symbol": symbol} if symbol else None
            )
            
            if market_results['documents']:
                context["market"] = self._format_search_results(market_results)
            
            return context
            
        except Exception as e:
            logger.error(f"Failed to retrieve context: {str(e)}")
            return {"news": [], "market": []}
    
    def _format_search_results(self, results) -> List[Dict]:
        """Format ChromaDB search results."""
        formatted = []
        
        for i, doc in enumerate(results['documents'][0]):
            formatted.append({
                "content": doc,
                "metadata": results['metadatas'][0][i] if results['metadatas'] else {},
                "score": results['distances'][0][i] if results['distances'] else 0
            })
        
        return formatted
    
    def _create_explanation_prompt(self, symbol: str, anomaly_score: float, 
                                 processed_data: dict, context: Dict[str, List[Dict]]) -> str:
        """Create a prompt for anomaly explanation."""
        
        indicators = processed_data.get('indicators', {})
        
        prompt = f"""
        Analyze the following market anomaly for {symbol}:
        
        ANOMALY DETAILS:
        - Symbol: {symbol}
        - Anomaly Score: {anomaly_score:.2f}
        - Price Change: {indicators.get('price_change', 0):.2f}%
        - Volume Ratio: {indicators.get('volume_ratio', 1):.2f}x normal
        - RSI: {indicators.get('rsi', 50):.2f}
        - Volatility: {indicators.get('volatility', 0):.4f}
        
        RELEVANT NEWS CONTEXT:
        """
        
        for news in context.get('news', [])[:3]:
            prompt += f"- {news['content'][:200]}...\n"
        
        prompt += f"""
        
        MARKET CONTEXT:
        """
        
        for market in context.get('market', [])[:2]:
            prompt += f"- {market['content'][:150]}...\n"
        
        prompt += f"""
        
        Please provide a clear, concise explanation (2-3 sentences) of:
        1. What type of anomaly this appears to be
        2. Potential causes based on the context
        3. The significance/risk level for investors
        
        Focus on actionable insights for traders and investors.
        """
        
        return prompt
    
    def _store_analysis(self, symbol: str, anomaly_score: float, explanation: str):
        """Store AI analysis for future reference."""
        try:
            doc_id = f"analysis_{symbol}_{int(datetime.now().timestamp())}"
            
            doc_text = f"Anomaly Analysis for {symbol}: {explanation}"
            
            metadata = {
                "symbol": symbol,
                "anomaly_score": float(anomaly_score),
                "timestamp": datetime.now().isoformat(),
                "type": "ai_analysis"
            }
            
            self.analysis_collection.add(
                documents=[doc_text],
                ids=[doc_id],
                metadatas=[metadata]
            )
            
        except Exception as e:
            logger.error(f"Failed to store analysis: {str(e)}")
    
    def query_knowledge_base(self, query: str, symbol: Optional[str] = None) -> str:
        """Query the knowledge base for general financial information."""
        try:
            # Retrieve relevant context
            context = self.retrieve_context(symbol or "", query)
            
            # Create query prompt
            prompt = f"""
            User Query: {query}
            
            RELEVANT CONTEXT:
            
            News Context:
            """
            
            for news in context.get('news', [])[:3]:
                prompt += f"- {news['content'][:200]}...\n"
            
            prompt += "\nMarket Context:\n"
            for market in context.get('market', [])[:2]:
                prompt += f"- {market['content'][:150]}...\n"
            
            prompt += f"""
            
            Based on the above context, please provide a helpful and accurate response to the user's query.
            Focus on recent market data and news when relevant.
            """
            
            response = self.openai_client.chat.completions.create(
                model=self.settings.LLM_MODEL,
                messages=[
                    {"role": "system", "content": "You are a knowledgeable financial AI assistant. Provide helpful, accurate information based on the given context."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.settings.MAX_TOKENS,
                temperature=self.settings.TEMPERATURE
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Knowledge base query failed: {str(e)}")
            return "I'm sorry, I encountered an error while processing your query. Please try again later."
