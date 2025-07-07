import os
import asyncio
import json
import subprocess
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta
import numpy as np
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from mcp_use import MCPAgent, MCPClient
import faiss
from sentence_transformers import SentenceTransformer
import pickle
from typing import List, Dict, Any, Optional

class FAISSDatabase:
    """FAISS vector database for storing and retrieving business data"""
    
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.index = None
        self.encoder = None
        self.documents = []
        self.metadata = []
        self.is_initialized = False
        
    def initialize(self):
        """Initialize FAISS index and sentence transformer"""
        try:
            # Initialize sentence transformer for encoding
            self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Create FAISS index
            self.index = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine similarity
            
            self.is_initialized = True
            return True
        except Exception as e:
            st.error(f"Error initializing FAISS database: {str(e)}")
            return False
    
    def add_documents(self, documents: List[Dict[str, Any]], data_type: str):
        """Add documents to the FAISS database"""
        if not self.is_initialized:
            if not self.initialize():
                return False
        
        if self.index is None:
            return False
        
        try:
            # Ensure encoder is initialized
            if self.encoder is None:
                if not self.initialize():
                    return False
            
            # Convert documents to text representations
            texts = []
            for doc in documents:
                # Create a comprehensive text representation of the document
                text_parts = []
                for key, value in doc.items():
                    if isinstance(value, (str, int, float)):
                        text_parts.append(f"{key}: {value}")
                    elif isinstance(value, dict):
                        text_parts.append(f"{key}: {json.dumps(value)}")
                    else:
                        text_parts.append(f"{key}: {str(value)}")
                
                text = " | ".join(text_parts)
                texts.append(text)
            
            # Encode texts to vectors
            assert self.encoder is not None, "Encoder should be initialized"
            embeddings = self.encoder.encode(texts, show_progress_bar=False)
            
            # Normalize vectors for cosine similarity
            faiss.normalize_L2(embeddings)
            
            # Convert embeddings to numpy array of type float32
            embeddings = np.array(embeddings).astype('float32')
            
            # Only add embeddings if there are embeddings to add
            if embeddings.shape[0] > 0:
                assert self.index is not None, "Index should be initialized"
                self.index.add(embeddings)  # type: ignore
            
            # Store documents and metadata
            for i, doc in enumerate(documents):
                self.documents.append(doc)
                self.metadata.append({
                    'data_type': data_type,
                    'index': len(self.documents) - 1,
                    'timestamp': datetime.now().isoformat()
                })
            
            return True
            
        except Exception as e:
            st.error(f"Error adding documents to FAISS database: {str(e)}")
            return False
    
    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search for relevant documents"""
        if not self.is_initialized:
            if not self.initialize():
                return []
        
        if self.index is None or self.index.ntotal == 0:
            return []
        
        try:
            # Ensure encoder is initialized
            if self.encoder is None:
                if not self.initialize():
                    return []
            
            # Encode query
            assert self.encoder is not None, "Encoder should be initialized"
            query_embedding = self.encoder.encode([query], show_progress_bar=False)
            faiss.normalize_L2(query_embedding)
            
            # Search
            query_embedding = query_embedding.reshape(1, -1).astype('float32')
            
            # Limit k to the number of documents available
            k = min(k, self.index.ntotal)
            
            assert self.index is not None, "Index should be initialized"
            scores, indices = self.index.search(query_embedding, k)  # type: ignore
            
            # Return results
            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx < len(self.documents) and idx >= 0:  # Valid index check
                    results.append({
                        'document': self.documents[idx],
                        'metadata': self.metadata[idx],
                        'score': float(score)
                    })
            
            return results
            
        except Exception as e:
            st.error(f"Error searching FAISS database: {str(e)}")
            return []
    
    def get_document_count(self) -> int:
        """Get total number of documents in the database"""
        return len(self.documents)
    
    def get_data_type_counts(self) -> Dict[str, int]:
        """Get count of documents by data type"""
        counts = {}
        for meta in self.metadata:
            data_type = meta['data_type']
            counts[data_type] = counts.get(data_type, 0) + 1
        return counts
    
    def clear_database(self):
        """Clear all data from the database"""
        if self.index is not None:
            self.index = faiss.IndexFlatIP(self.dimension)
        self.documents = []
        self.metadata = []
    
    def save_database(self, filepath: str):
        """Save database to file"""
        try:
            data = {
                'index': faiss.serialize_index(self.index) if self.index else None,
                'documents': self.documents,
                'metadata': self.metadata,
                'dimension': self.dimension
            }
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)
            return True
        except Exception as e:
            st.error(f"Error saving FAISS database: {str(e)}")
            return False
    
    def load_database(self, filepath: str):
        """Load database from file"""
        try:
            if not os.path.exists(filepath):
                st.error(f"Database file {filepath} not found")
                return False
                
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            
            if data['index'] is not None:
                self.index = faiss.deserialize_index(data['index'])
            else:
                self.index = faiss.IndexFlatIP(self.dimension)
                
            self.documents = data['documents']
            self.metadata = data['metadata']
            self.dimension = data['dimension']
            
            # Reinitialize encoder
            self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
            self.is_initialized = True
            return True
        except Exception as e:
            st.error(f"Error loading FAISS database: {str(e)}")
            return False

# Page configuration
st.set_page_config(
    page_title="Acumen AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #2E86AB;
        font-size: 2.5rem;
        margin-bottom: 1rem;
    }
    .chat-container {
        max-height: 500px;
        overflow-y: auto;
        padding: 1rem;
        border: 1px solid #ddd;
        border-radius: 10px;
        background-color: #f8f9fa;
    }
    .user-message {
        background-color: #01165c;
        padding: 0.5rem 1rem;
        border-radius: 15px;
        margin: 0.5rem 0;
        text-align: left;
        border-left: 4px solid #2196f3;
    }
    .assistant-message {
        background-color: #197a01;
        padding: 0.5rem 1rem;
        border-radius: 15px;
        margin: 0.5rem 0;
        align-items: right;
        text-align: left;
        border-left: 4px solid #4caf50;
    }
    .status-success {
        color: #28a745;
        font-weight: bold;
    }
    .status-error {
        color: #dc3545;
        font-weight: bold;
    }
    .status-warning {
        color: #ffc107;
        font-weight: bold;
    }
    .data-card {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        margin: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'mcp_agent' not in st.session_state:
    st.session_state.mcp_agent = None
if 'mcp_client' not in st.session_state:
    st.session_state.mcp_client = None
if 'is_initialized' not in st.session_state:
    st.session_state.is_initialized = False
if 'initialization_error' not in st.session_state:
    st.session_state.initialization_error = None
if 'faiss_db' not in st.session_state:
    st.session_state.faiss_db = FAISSDatabase()
if 'faiss_initialized' not in st.session_state:
    st.session_state.faiss_initialized = False

# Data storage for different CSV types
if 'reviews_data' not in st.session_state:
    st.session_state.reviews_data = None
if 'sales_data' not in st.session_state:
    st.session_state.sales_data = None
if 'inventory_data' not in st.session_state:
    st.session_state.inventory_data = None

# JSON storage
if 'reviews_json' not in st.session_state:
    st.session_state.reviews_json = None
if 'sales_json' not in st.session_state:
    st.session_state.sales_json = None
if 'inventory_json' not in st.session_state:
    st.session_state.inventory_json = None

# Analysis storage
if 'comprehensive_analysis' not in st.session_state:
    st.session_state.comprehensive_analysis = None

async def initialize_mcp_agent(groq_api_key, selected_model):
    """Initialize MCP Agent with error handling"""
    try:
        # Set environment variable
        os.environ["GROQ_API_KEY"] = groq_api_key
        
        # Fixed config file and max steps
        config_file = "browser_mcp.json"
        max_steps = 15
        
        # Check if config file exists
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Configuration file '{config_file}' not found")
        
        # Validate config file content
        with open(config_file, 'r') as f:
            config_content = f.read()
            json.loads(config_content)  # Validate JSON
        
        # Create MCP Client
        client = MCPClient.from_config_file(config_file)
        
        # Create LLM with selected model
        llm = ChatGroq(model=selected_model)
        
        # Create MCP agent
        agent = MCPAgent(
            llm=llm,
            client=client,
            memory_enabled=True,
            max_steps=max_steps
        )
        
        # Test the agent
        await agent.run("Hello, can you help me with business intelligence analysis?")
        
        return agent, client, None
        
    except Exception as e:
        return None, None, str(e)

def process_csv_file(uploaded_file, data_type):
    """Process CSV file and convert to JSON"""
    try:
        # Read CSV file
        df = pd.read_csv(uploaded_file)
        
        # Clean column names (remove spaces, convert to lowercase)
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
        
        # Data type specific processing
        if data_type == "reviews":
            df = process_reviews_data(df)
        elif data_type == "sales":
            df = process_sales_data(df)
        elif data_type == "inventory":
            df = process_inventory_data(df)
        
        # Convert to JSON
        json_data = df.to_json(orient='records', date_format='iso')
        parsed_json = json.loads(json_data) if json_data else []
        
        # Add to FAISS database
        if parsed_json and st.session_state.faiss_db:
            success = st.session_state.faiss_db.add_documents(parsed_json, data_type)
            if success:
                st.session_state.faiss_initialized = True
        
        return df, parsed_json, None
        
    except Exception as e:
        return None, None, str(e)

def process_reviews_data(df):
    """Process and enhance reviews data"""
    # Expected columns: product_id, review_text, rating, customer_id, date
    required_cols = ['product_id', 'review_text', 'rating']
    
    # Check if required columns exist (with some flexibility)
    for col in required_cols:
        if col not in df.columns:
            # Try to find similar column names
            similar_cols = [c for c in df.columns if any(part in c for part in col.split('_'))]
            if similar_cols:
                df[col] = df[similar_cols[0]]
    
    # Add sentiment analysis placeholder (you can integrate actual sentiment analysis)
    if 'sentiment' not in df.columns and 'rating' in df.columns:
        df['sentiment'] = df['rating'].apply(lambda x: 'positive' if x >= 4 else 'negative' if x <= 2 else 'neutral')
    
    # Add review length
    if 'review_text' in df.columns:
        df['review_length'] = df['review_text'].str.len()
    
    return df

def process_sales_data(df):
    """Process and enhance sales data"""
    # Expected columns: product_id, quantity_sold, sale_date, revenue, customer_id
    
    # Convert date column if exists
    date_cols = [col for col in df.columns if 'date' in col]
    for col in date_cols:
        try:
            df[col] = pd.to_datetime(df[col], errors='coerce')
        except Exception:
            pass
    
    # Add calculated fields
    if 'quantity_sold' in df.columns and 'revenue' in df.columns:
        # Avoid division by zero
        df['price_per_unit'] = df.apply(lambda row: row['revenue'] / row['quantity_sold'] if row['quantity_sold'] != 0 else 0, axis=1)
    
    # Add time-based features
    if 'sale_date' in df.columns:
        df['sale_month'] = df['sale_date'].dt.month
        df['sale_quarter'] = df['sale_date'].dt.quarter
        df['sale_year'] = df['sale_date'].dt.year
    
    return df

def process_inventory_data(df):
    """Process and enhance inventory data"""
    # Expected columns: product_id, current_stock, reorder_level, supplier, last_restock_date
    
    # Convert date columns
    date_cols = [col for col in df.columns if 'date' in col]
    for col in date_cols:
        try:
            df[col] = pd.to_datetime(df[col], errors='coerce')
        except Exception:
            pass
    
    # Add inventory status
    if 'current_stock' in df.columns and 'reorder_level' in df.columns:
        df['stock_status'] = df.apply(lambda row: 
            'critical' if row['current_stock'] <= row['reorder_level'] * 0.5 else
            'low' if row['current_stock'] <= row['reorder_level'] else
            'adequate', axis=1)
    
    # Add days since last restock
    if 'last_restock_date' in df.columns:
        df['days_since_restock'] = (pd.Timestamp.now() - df['last_restock_date']).dt.days
    
    return df

def analyze_business_data():
    """Comprehensive analysis of all business data"""
    analysis = {
        'timestamp': datetime.now().isoformat(),
        'data_availability': {
            'reviews': st.session_state.reviews_data is not None,
            'sales': st.session_state.sales_data is not None,
            'inventory': st.session_state.inventory_data is not None
        },
        'analysis': {}
    }
    
    # Reviews analysis
    if st.session_state.reviews_data is not None:
        reviews_df = st.session_state.reviews_data
        analysis['analysis']['reviews'] = {
            'total_reviews': len(reviews_df),
            'average_rating': float(reviews_df['rating'].mean()) if 'rating' in reviews_df.columns else None,
            'sentiment_distribution': reviews_df['sentiment'].value_counts().to_dict() if 'sentiment' in reviews_df.columns else {},
            'products_reviewed': int(reviews_df['product_id'].nunique()) if 'product_id' in reviews_df.columns else None
        }
    
    # Sales analysis
    if st.session_state.sales_data is not None:
        sales_df = st.session_state.sales_data
        analysis['analysis']['sales'] = {
            'total_revenue': float(sales_df['revenue'].sum()) if 'revenue' in sales_df.columns else None,
            'total_quantity_sold': int(sales_df['quantity_sold'].sum()) if 'quantity_sold' in sales_df.columns else None,
            'unique_products_sold': int(sales_df['product_id'].nunique()) if 'product_id' in sales_df.columns else None,
            'average_order_value': float(sales_df['revenue'].mean()) if 'revenue' in sales_df.columns else None
        }
    
    # Inventory analysis
    if st.session_state.inventory_data is not None:
        inventory_df = st.session_state.inventory_data
        analysis['analysis']['inventory'] = {
            'total_products': len(inventory_df),
            'low_stock_products': len(inventory_df[inventory_df['stock_status'] == 'low']) if 'stock_status' in inventory_df.columns else None,
            'critical_stock_products': len(inventory_df[inventory_df['stock_status'] == 'critical']) if 'stock_status' in inventory_df.columns else None,
            'total_stock_value': float(inventory_df['current_stock'].sum()) if 'current_stock' in inventory_df.columns else None
        }
    
    return analysis

def generate_comprehensive_context(user_query: str = ""):
    """Generate comprehensive context for the AI agent using FAISS search"""
    context = """
    BUSINESS INTELLIGENCE DATA ANALYSIS CONTEXT:
    
    You are a Acumen AI with access to comprehensive business data including:
    1. Product Reviews Data
    2. Sales Data  
    3. Inventory Data
    
    """
    
    # Use FAISS search to retrieve relevant information
    if st.session_state.faiss_initialized and st.session_state.faiss_db and user_query:
        try:
            # Search for relevant documents
            search_results = st.session_state.faiss_db.search(user_query, k=10)
            
            if search_results:
                context += f"""
                RELEVANT DATA RETRIEVED:
                Found {len(search_results)} relevant documents for your query.
                
                """
                
                # Group results by data type
                reviews_data = []
                sales_data = []
                inventory_data = []
                
                for result in search_results:
                    doc = result['document']
                    metadata = result['metadata']
                    score = result['score']
                    
                    if metadata['data_type'] == 'reviews':
                        reviews_data.append((doc, score))
                    elif metadata['data_type'] == 'sales':
                        sales_data.append((doc, score))
                    elif metadata['data_type'] == 'inventory':
                        inventory_data.append((doc, score))
                
                # Add relevant reviews data
                if reviews_data:
                    context += f"""
                    RELEVANT PRODUCT REVIEWS DATA (Top {min(5, len(reviews_data))} results):
                    """
                    for i, (doc, score) in enumerate(reviews_data[:5]):
                        context += f"""
                        Review {i+1} (Relevance Score: {score:.3f}):
                        {json.dumps(doc, indent=2, default=str)}
                        """
                
                # Add relevant sales data
                if sales_data:
                    context += f"""
                    
                    RELEVANT SALES DATA (Top {min(5, len(sales_data))} results):
                    """
                    for i, (doc, score) in enumerate(sales_data[:5]):
                        context += f"""
                        Sales Record {i+1} (Relevance Score: {score:.3f}):
                        {json.dumps(doc, indent=2, default=str)}
                        """
                
                # Add relevant inventory data
                if inventory_data:
                    context += f"""
                    
                    RELEVANT INVENTORY DATA (Top {min(5, len(inventory_data))} results):
                    """
                    for i, (doc, score) in enumerate(inventory_data[:5]):
                        context += f"""
                        Inventory Item {i+1} (Relevance Score: {score:.3f}):
                        {json.dumps(doc, indent=2, default=str)}
                        """
        except Exception as e:
            context += f"""
            Note: Error retrieving data: {str(e)}
            Falling back to general data overview.
            """
    
    # Fallback to general data overview if FAISS search fails or no query provided
    if not user_query or not st.session_state.faiss_initialized:
        # Add reviews context
        if st.session_state.reviews_json:
            context += f"""
            PRODUCT REVIEWS DATA:
            - Total Reviews: {len(st.session_state.reviews_json)}
            - Sample Review Structure: {json.dumps(st.session_state.reviews_json[0] if st.session_state.reviews_json else {}, indent=2, default=str)}
            - Available for sentiment analysis, product feedback analysis, rating trends
            """
        
        # Add sales context
        if st.session_state.sales_json:
            context += f"""
            
            SALES DATA:
            - Total Sales Records: {len(st.session_state.sales_json)}
            - Sample Sales Structure: {json.dumps(st.session_state.sales_json[0] if st.session_state.sales_json else {}, indent=2, default=str)}
            - Available for sales volume analysis, revenue tracking, trend analysis
            """
        
        # Add inventory context
        if st.session_state.inventory_json:
            context += f"""
            
            INVENTORY DATA:
            - Total Inventory Records: {len(st.session_state.inventory_json)}
            - Sample Inventory Structure: {json.dumps(st.session_state.inventory_json[0] if st.session_state.inventory_json else {}, indent=2, default=str)}
            - Available for stock level analysis, reorder recommendations, inventory optimization
            """
    
    # Add comprehensive analysis
    if st.session_state.comprehensive_analysis:
        context += f"""
        
        COMPREHENSIVE BUSINESS ANALYSIS:
        {json.dumps(st.session_state.comprehensive_analysis, indent=2, default=str)}
        """
    
    # Add FAISS database statistics
    if st.session_state.faiss_initialized and st.session_state.faiss_db:
        context += f"""
        
        FAISS DATABASE STATISTICS:
        - Total Documents: {st.session_state.faiss_db.get_document_count()}
        - Data Type Distribution: {st.session_state.faiss_db.get_data_type_counts()}
        """
    
    context += """
    
    CAPABILITIES:
    - Sentiment analysis of customer reviews for specific products
    - Sales volume analysis and trends
    - Inventory status monitoring and restock recommendations
    - Cross-data analysis (e.g., correlation between reviews and sales)
    - Product performance insights
    - Business intelligence recommendations
    - Semantic search through business data using FAISS vector database
    
    When answering questions, use the actual data provided above to give specific, accurate insights.
    If relevant data was retrieved from the FAISS database, prioritize that information in your response.
    """
    
    return context

def display_data_overview():
    """Display overview of uploaded data"""
    st.subheader("📊 Business Data Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="data-card">', unsafe_allow_html=True)
        st.markdown("**📝 Product Reviews**")
        if st.session_state.reviews_data is not None:
            st.markdown(f"✅ {len(st.session_state.reviews_data)} reviews loaded")
            if 'rating' in st.session_state.reviews_data.columns:
                avg_rating = st.session_state.reviews_data['rating'].mean()
                st.markdown(f"⭐ Avg Rating: {avg_rating:.2f}")
        else:
            st.markdown("❌ No reviews data")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="data-card">', unsafe_allow_html=True)
        st.markdown("**💰 Sales Data**")
        if st.session_state.sales_data is not None:
            st.markdown(f"✅ {len(st.session_state.sales_data)} sales records")
            if 'revenue' in st.session_state.sales_data.columns:
                total_revenue = st.session_state.sales_data['revenue'].sum()
                st.markdown(f"💵 Total Revenue: ${total_revenue:,.2f}")
        else:
            st.markdown("❌ No sales data")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="data-card">', unsafe_allow_html=True)
        st.markdown("**📦 Inventory Data**")
        if st.session_state.inventory_data is not None:
            st.markdown(f"✅ {len(st.session_state.inventory_data)} products tracked")
            if 'stock_status' in st.session_state.inventory_data.columns:
                critical_items = len(st.session_state.inventory_data[st.session_state.inventory_data['stock_status'] == 'critical'])
                st.markdown(f"⚠️ Critical Stock: {critical_items} items")
        else:
            st.markdown("❌ No inventory data")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="data-card">', unsafe_allow_html=True)
        st.markdown("**🔍 FAISS Database**")
        if st.session_state.faiss_initialized and st.session_state.faiss_db:
            total_docs = st.session_state.faiss_db.get_document_count()
            type_counts = st.session_state.faiss_db.get_data_type_counts()
            st.markdown(f"✅ {total_docs} documents indexed")
            st.markdown(f"📊 Types: {', '.join([f'{k}: {v}' for k, v in type_counts.items()])}")
        else:
            st.markdown("❌ Not initialized")
        st.markdown('</div>', unsafe_allow_html=True)

def main():
    # Main header
    st.markdown('<h1 class="main-header">📊 Acumen AI</h1>', unsafe_allow_html=True)
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Load environment variables
        load_dotenv()
        
        # API Key input
        groq_api_key = st.text_input(
            "GROQ API Key",
            type="password",
            value=os.getenv("GROQ_API_KEY", ""),
            help="Enter your GROQ API key"
        )
        
        # Model selection
        model_options = [
            "llama-3.1-8b-instant",
            "llama-3.3-70b-versatile"
        ]
        selected_model = st.selectbox("Select Model", model_options, index=0)
        
        # Initialize button
        if st.button("🚀 Initialize AI Agent", type="primary"):
            if not groq_api_key:
                st.error("Please provide a GROQ API key")
            else:
                with st.spinner("Initializing AI Agent..."):
                    try:
                        agent, client, error = asyncio.run(initialize_mcp_agent(groq_api_key, selected_model))
                        if error:
                            st.session_state.initialization_error = error
                            st.session_state.is_initialized = False
                            st.error(f"Initialization failed: {error}")
                        else:
                            st.session_state.mcp_agent = agent
                            st.session_state.mcp_client = client
                            st.session_state.is_initialized = True
                            st.session_state.initialization_error = None
                            st.success("AI Agent initialized successfully!")
                    except Exception as e:
                        st.session_state.initialization_error = str(e)
                        st.session_state.is_initialized = False
                        st.error(f"Initialization failed: {str(e)}")
        
        # Data Upload Section
        st.header("📂 Business Data Upload")
        
        # Reviews data upload
        st.subheader("📝 Product Reviews")
        reviews_file = st.file_uploader(
            "Upload Product Reviews CSV",
            type=['csv'],
            key="reviews_upload",
            help="CSV with columns: product_id, review_text, rating, customer_id, date"
        )
        
        if reviews_file is not None:
            with st.spinner("Processing reviews data..."):
                df, json_data, error = process_csv_file(reviews_file, "reviews")
                if error:
                    st.error(f"Error processing reviews: {error}")
                elif df is not None:
                    st.session_state.reviews_data = df
                    st.session_state.reviews_json = json_data
                    st.success(f"✅ {len(df)} reviews loaded successfully!")
        
        # Sales data upload
        st.subheader("💰 Sales Data")
        sales_file = st.file_uploader(
            "Upload Sales Data CSV",
            type=['csv'],
            key="sales_upload",
            help="CSV with columns: product_id, quantity_sold, sale_date, revenue, customer_id"
        )
        
        if sales_file is not None:
            with st.spinner("Processing sales data..."):
                df, json_data, error = process_csv_file(sales_file, "sales")
                if error:
                    st.error(f"Error processing sales: {error}")
                elif df is not None:
                    st.session_state.sales_data = df
                    st.session_state.sales_json = json_data
                    st.success(f"✅ {len(df)} sales records loaded successfully!")
        
        # Inventory data upload
        st.subheader("📦 Inventory Data")
        inventory_file = st.file_uploader(
            "Upload Inventory Data CSV",
            type=['csv'],
            key="inventory_upload",
            help="CSV with columns: product_id, current_stock, reorder_level, supplier, last_restock_date"
        )
        
        if inventory_file is not None:
            with st.spinner("Processing inventory data..."):
                df, json_data, error = process_csv_file(inventory_file, "inventory")
                if error:
                    st.error(f"Error processing inventory: {error}")
                elif df is not None:
                    st.session_state.inventory_data = df
                    st.session_state.inventory_json = json_data
                    st.success(f"✅ {len(df)} inventory items loaded successfully!")
        
        # Analyze data button
        if any([st.session_state.reviews_data is not None, 
                st.session_state.sales_data is not None, 
                st.session_state.inventory_data is not None]):
            if st.button("🔍 Analyze Business Data", type="secondary"):
                with st.spinner("Analyzing business data..."):
                    st.session_state.comprehensive_analysis = analyze_business_data()
                    st.success("✅ Business data analyzed successfully!")
        
        # FAISS Database Management
        st.header("🔍 FAISS Database")
        
        if st.session_state.faiss_initialized and st.session_state.faiss_db:
            st.markdown(f"**Status:** ✅ Initialized")
            st.markdown(f"**Documents:** {st.session_state.faiss_db.get_document_count()}")
            
            # Save database
            if st.button("💾 Save FAISS Database"):
                if st.session_state.faiss_db.save_database("faiss_business_data.pkl"):
                    st.success("FAISS database saved successfully!")
                else:
                    st.error("Failed to save FAISS database")
            
            # Load database
            if st.button("📂 Load FAISS Database"):
                if st.session_state.faiss_db.load_database("faiss_business_data.pkl"):
                    st.success("FAISS database loaded successfully!")
                else:
                    st.error("Failed to load FAISS database")
            
            # Clear database
            if st.button("🗑️ Clear FAISS Database"):
                st.session_state.faiss_db.clear_database()
                st.session_state.faiss_initialized = False
                st.success("FAISS database cleared!")
        else:
            st.markdown("**Status:** ❌ Not initialized")
            st.markdown("Upload data to initialize FAISS database")
        
        # Clear all data button
        if st.button("🗑️ Clear All Data"):
            st.session_state.reviews_data = None
            st.session_state.sales_data = None
            st.session_state.inventory_data = None
            st.session_state.reviews_json = None
            st.session_state.sales_json = None
            st.session_state.inventory_json = None
            st.session_state.comprehensive_analysis = None
            st.session_state.messages = []
            if st.session_state.faiss_db:
                st.session_state.faiss_db.clear_database()
                st.session_state.faiss_initialized = False
            st.success("All data cleared!")
        
        # Status indicators
        st.header("📊 Status")
        if st.session_state.is_initialized:
            st.markdown('<span class="status-success">🟢 AI Agent Ready</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-error">🔴 AI Agent Not Initialized</span>', unsafe_allow_html=True)
        
        # Data status
        data_loaded = any([st.session_state.reviews_data is not None,
                          st.session_state.sales_data is not None,
                          st.session_state.inventory_data is not None])
        
        if data_loaded:
            st.markdown('<span class="status-success">📊 Business Data Loaded</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-warning">📂 No Business Data</span>', unsafe_allow_html=True)
        
        # FAISS database status
        if st.session_state.faiss_initialized:
            st.markdown('<span class="status-success">🔍 FAISS Database Ready</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-warning">🔍 FAISS Database Not Ready</span>', unsafe_allow_html=True)
    
    # Main content area
    if any([st.session_state.reviews_data is not None,
            st.session_state.sales_data is not None,
            st.session_state.inventory_data is not None]):
        
        # Show data overview and chat in tabs
        tab1, tab2 = st.tabs(["💬 Business Intelligence Chat", "📊 Data Overview"])
        
        with tab2:
            display_data_overview()
            
            # Show detailed data if available
            if st.session_state.reviews_data is not None:
                st.subheader("📝 Reviews Data Sample")
                st.dataframe(st.session_state.reviews_data.head(), use_container_width=True)
            
            if st.session_state.sales_data is not None:
                st.subheader("💰 Sales Data Sample")
                st.dataframe(st.session_state.sales_data.head(), use_container_width=True)
            
            if st.session_state.inventory_data is not None:
                st.subheader("📦 Inventory Data Sample")
                st.dataframe(st.session_state.inventory_data.head(), use_container_width=True)
        
        with tab1:
            business_chat_interface()
    else:
        # Show upload instructions
        st.markdown("""
        ## 🚀 Welcome to Acumen AI!
        
        To get started, please:
        1. **Initialize the AI Agent** in the sidebar with your GROQ API key
        2. **Upload your business data** (Reviews, Sales, or Inventory CSV files)
        3. **Start asking questions** about your business data
        
        ### 📊 Supported Data Types:
        - **📝 Product Reviews**: Customer feedback, ratings, sentiment analysis
        - **💰 Sales Data**: Revenue, quantity sold, sales trends
        - **📦 Inventory Data**: Stock levels, reorder points, inventory optimization
        
        ### 🔍 FAISS Vector Database Features:
        - **Semantic Search**: Find relevant data using natural language queries
        - **Intelligent Retrieval**: Get the most relevant documents for your questions
        - **Persistent Storage**: Save and load your indexed data
        - **Real-time Indexing**: Automatically index uploaded CSV data
        
        ### 🎯 Example Questions You Can Ask:
        - "What's the sentiment of reviews for product XYZ?"
        - "What are the top-selling products this month?"
        - "Which products need restocking urgently?"
        - "Show me the correlation between customer ratings and sales volume"
        - "What's the average order value by product category?"
        - "Find products with low ratings but high sales"
        - "Which inventory items have the most customer complaints?"
        """)

def business_chat_interface():
    """Business intelligence chat interface"""
    st.header("💬 Business Intelligence Chat")
    
    # Display initialization status
    if not st.session_state.is_initialized:
        st.warning("⚠️ Please initialize the AI Agent in the sidebar before starting to chat.")
        return
    
    # Chat container
    chat_container = st.container()
    
    with chat_container:
        # Display chat history
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.markdown(f'<div class="user-message"><strong>You:</strong> {message["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="assistant-message"><strong>AI Assistant:</strong> {message["content"]}</div>', unsafe_allow_html=True)
    
    # Chat input
    user_input = st.chat_input("Ask me anything about your business data...", key="business_chat")
    
    if user_input:
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Get response from AI agent
        with st.spinner("Analyzing your business data..."):
            try:
                # Synchronous wrapper for async call
                def get_business_response_sync():
                    async def get_business_response():
                        # Generate comprehensive context with user query for FAISS search
                        business_context = generate_comprehensive_context(user_input)
                        
                        enhanced_prompt = f"""
                            You are an expert business intelligence analyst with deep expertise in data 
                            interpretation, market analysis, and strategic recommendations. Your role is to
                            transform raw business data into actionable insights that drive decision-making.

                            ## CONTEXT & DATA SOURCES
                            {business_context}

                            ## ANALYSIS REQUEST
                            **User Question:** {user_input}

                            ## RESPONSE FRAMEWORK
                            Please structure your response using the following format:

                        ### 1. EXECUTIVE SUMMARY
                        - Provide a 2-3 sentence key finding that directly answers the user's question
                        - Highlight the most critical insight or recommendation

                        ### 2. DATA ANALYSIS & INSIGHTS
                        Based on the analysis type requested:

                        **For Sentiment Analysis:**
                        - Quantify sentiment distribution (positive/negative/neutral percentages)
                        - Identify sentiment trends over time periods
                        - Highlight specific themes in customer feedback
                        - Correlate sentiment with business metrics (sales, ratings, etc.)

                        **For Sales Analysis:**
                        - Present key performance indicators (revenue, volume, growth rates)
                        - Identify seasonal patterns and trends
                        - Highlight top/bottom performing products, regions, or time periods
                        - Calculate year-over-year or period-over-period changes

                        **For Inventory Analysis:**
                        - Assess current stock levels vs. demand patterns
                        - Identify overstocked and understocked items
                        - Calculate inventory turnover rates
                        - Provide reorder point calculations and safety stock recommendations

                        **For Cross-Data Analysis:**
                        - Establish correlations between different data sets
                        - Identify cause-and-effect relationships
                        - Highlight unexpected patterns or anomalies

                        ### 3. SPECIFIC METRICS & EVIDENCE
                        - Cite exact numbers, percentages, and data points from the provided data
                        - Include relevant calculations (growth rates, ratios, averages)
                        - Reference specific time periods, product categories, or customer segments
                        - Quote relevant text from reviews or feedback when applicable

                        ### 4. BUSINESS IMPLICATIONS
                        - Explain what the data means for business performance
                        - Identify opportunities and risks
                        - Assess competitive positioning where relevant
                        - Highlight areas requiring immediate attention

                        ### 5. ACTIONABLE RECOMMENDATIONS
                        Provide 3-5 specific, prioritized recommendations:
                        - **HIGH PRIORITY:** Actions needed within 1-4 weeks
                        - **MEDIUM PRIORITY:** Actions needed within 1-3 months  
                        - **STRATEGIC:** Long-term initiatives (3-12 months)

                        For each recommendation, include:
                        - Specific action steps
                        - Expected business impact
                        - Resource requirements
                        - Success metrics

                        **Data Quality Notes:**
                        - Highlight any data limitations or gaps
                        - Note sample sizes and time periods
                        - Mention confidence levels in your analysis

                        ## ANALYSIS GUIDELINES
                        1. **Be Quantitative:** Use numbers, percentages, and metrics wherever possible
                        2. **Show Trends:** Compare current performance to historical data
                        3. **Context Matters:** Consider seasonal factors, market conditions, and business cycles
                        4. **Actionable Focus:** Every insight should lead to a clear business action
                        5. **Risk Assessment:** Identify potential negative impacts of recommendations
                        6. **Stakeholder Perspective:** Consider impact on customers, operations, and finances

                        ## OUTPUT REQUIREMENTS
                        - Use clear, professional business language
                        - Include specific data points in every major claim
                        - Provide confidence levels for predictions or forecasts
                        - Suggest additional data that would enhance the analysis
                        - Format numbers clearly (e.g., $1.2M, 15.3%, 2.5x growth)

                        Remember: Your analysis should enable immediate decision-making by business stakeholders. Focus on insights that can drive revenue, reduce costs, or improve customer satisfaction.
                        """
                        
                        return await st.session_state.mcp_agent.run(enhanced_prompt)
                    return asyncio.run(get_business_response())
                
                response = get_business_response_sync()

                if "Final Answer:" in response:
                    response_only = response.split("Final Answer:")[-1].strip()
                else:
                    response_only = response.strip()

                # Add assistant response to history
                st.session_state.messages.append({"role": "assistant", "content": response_only})

                # Force rerun to update chat display
                st.rerun()
                
            except Exception as e:
                st.error(f"Error getting response: {str(e)}")
    
    # Quick action buttons
    st.subheader("🔧 Quick Business Insights")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Overall Business Summary"):
            st.session_state.messages.append({"role": "user", "content": "Provide a comprehensive business summary based on all available data"})
            st.rerun()
    
    with col2:
        if st.button("⚠️ Critical Stock Alert"):
            st.session_state.messages.append({"role": "user", "content": "Which products need immediate restocking? Provide urgent inventory recommendations."})
            st.rerun()
    
    with col3:
        if st.button("⭐ Top Performing Products"):
            st.session_state.messages.append({"role": "user", "content": "Which products are performing best based on sales and customer reviews?"})
            st.rerun()
    
    # Example questions
    with st.expander("💡 Example Questions"):
        st.markdown("""
        **Sentiment Analysis:**
        - "What's the sentiment of reviews for product P004?"
        - "Which products have the most negative reviews?"
        - "Show me customer satisfaction trends"
        
        **Sales Analysis:**
        - "What's the sales volume for product P004?"
        - "Which products generate the most revenue?"
        - "Show me monthly sales trends"
        
        **Inventory Management:**
        - "Which products need restocking urgently?"
        - "What's the inventory status for product DEF456?"
        - "When should I reorder product GHI789?"
        
        **Cross-Data Analysis:**
        - "Is there a correlation between customer ratings and sales volume?"
        - "Which high-selling products have low customer satisfaction?"
        - "Recommend products to promote based on inventory and reviews"
        """)


if __name__ == "__main__":
    main()