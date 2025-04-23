# Importing all required packages
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
from phi.agent.agent import Agent
from phi.model.openai import OpenAIChat
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
from phi.tools.googlesearch import GoogleSearch
import yfinance as yf
from dotenv import load_dotenv
from fuzzywuzzy import fuzz, process
from phi.tools.yfinance import YFinanceTools
import os
from phi.knowledge.pdf import PDFUrlKnowledgeBase
from phi.vectordb.pgvector import PgVector2
from phi.utils.log import logger
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from openai import OpenAI
from langchain_openai import OpenAI
# from langchain.chat_models import ChatOpenAI
from langchain_community.chat_models import ChatOpenAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# Security configuration
LOGIN_CREDENTIALS = {"admin": "admin"}

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GroqAPi")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize session state
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None  # Initialize vector_store to None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# Valid exchanges
VALID_EXCHANGES = {
    'NYQ': 'NYSE', 'NGM': 'NASDAQ', 'NMS': 'NASDAQ', 'NAS': 'NASDAQ',
    'TSX': 'Toronto Stock Exchange', 'V': 'TSX Venture Exchange'
}


# Custom UI 
st.markdown("""
    <style>
    /* Global App Styling */
    .stApp {
        max-width: auto;
        margin: 0 auto;
        padding: 2rem;
        background: #ffffff; /* White background for light theme */
        color: #333333; /* Dark gray text */
        font-family: 'Arial', sans-serif;
    }

    /* Headers and Subheaders */
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff; /* White text on green background */
        background: rgb(0, 138, 0); /* Green background */
        padding: 0.5rem 1rem; /* Add padding for better appearance */
        border-radius: 5px; /* Rounded corners */
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1); /* Subtle shadow for depth */
        text-align: center;
    }

    /* Login Container */
    .login-container {
        max-width: 400px;
        margin: 100px auto;
        padding: 2rem;
        background: #f5f5f5; /* Light gray background for container */
        border-radius: 15px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1); /* Subtle shadow for depth */
        color: #333333; /* Dark gray text */
    }

    /* Input Fields */
    .stTextInput input[type="text"], .stTextInput input[type="password"] {
        background: #ffffff; /* White background for inputs */
        color: #333333; /* Dark gray text for readability */
        border: 1px solid #cccccc; /* Light gray border */
        border-radius: 8px;
        padding: 1rem;
        width: 100%;
        transition: border-color 0.3s, box-shadow 0.3s;
    }

    /* Placeholder Text */
    .stTextInput input[type="text"]::placeholder, .stTextInput input[type="password"]::placeholder {
        color: #999999; /* Medium gray placeholder text */
    }

    /* Hover Effect for Input Fields */
    .stTextInput input[type="text"]:hover, .stTextInput input[type="password"]:hover {
        border-color: #666666; /* Darker gray border on hover */
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05); /* Subtle shadow on hover */
    }

    /* Focus Effect for Input Fields */
    .stTextInput input[type="text"]:focus, .stTextInput input[type="password"]:focus {
        border-color: rgb(0, 138, 0); /* Green border on focus to match theme */
        outline: none;
        box-shadow: 0 4px 20px rgba(0, 138, 0, 0.2); /* Green shadow on focus */
    }

    /* Buttons */
    .stButton>button {
        background: rgb(0, 138, 0); /* Green background to match headers */
        color: #ffffff; /* White text for contrast */
        border-radius: 8px;
        padding: 0.8rem 2rem;
        font-weight: bold;
        transition: background 0.3s;
    }
    .stButton>button:hover {
        background: rgb(0, 115, 0); /* Slightly darker green on hover */
    }

    /* Dropdowns (Selectbox) */
    .stSelectbox > div > div {
        background: #f5f5f5; /* Light gray background for dropdown */
        color: #333333; /* Dark gray text for readability */
        border: 1px solid #cccccc; /* Light gray border */
        border-radius: 8px;
    }
    .stSelectbox > div > div:hover {
        border-color: #666666; /* Darker gray border on hover */
    }

    /* Sidebar */
    .sidebar {
        background: #f5f5f5; /* Light gray background for sidebar */
        border-radius: 15px;
        padding: 2rem;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05); /* Subtle shadow for depth */
        color: #333333; /* Dark gray text for readability */
    }

    /* Metric Cards */
    .metric-card { 
        background-color: #fafafa; /* Very light gray background */
        border-radius: 10px; 
        padding: 1rem; 
        margin: 0.5rem 0; 
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05); /* Subtle shadow */
        transition: transform 0.2s; 
    }
    .metric-card:hover { 
        transform: translateY(-2px); 
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); /* Slightly stronger shadow on hover */
    }

    /* News Cards */
    .news-card {
        background: #fafafa; /* Very light gray background */
        border-left: 5px solid rgb(0, 138, 0); /* Green left border to match theme */
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 8px;
        color: #333333; /* Dark gray text for readability */
    }

    /* Links */
    a {
        color: rgb(0, 138, 0); /* Green links to match theme */
        text-decoration: none;
    }
    a:hover {
        color: rgb(0, 115, 0); /* Slightly darker green on hover */
    }
    </style>
""", unsafe_allow_html=True)


# Authentication check
def check_login():
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False

    if not st.session_state.authenticated:
        with st.form("login_form"):
            st.markdown("<h2 style='text-align: center; color: #2c3e50;'>Login</h2>", unsafe_allow_html=True)
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login")

            if submitted:
                if username in LOGIN_CREDENTIALS and password == LOGIN_CREDENTIALS[username]:
                    st.session_state.authenticated = True
                    st.session_state.agents_initialized = False  # Reset this flag
                    st.rerun()
                else:
                    st.error("Invalid credentials")
        return False
    return True

# Database URL for storing vector embeddings
db_url = "postgresql+psycopg://ai:ai@localhost:5532/ai"

def process_pdf_document(uploaded_file):
    """Process the uploaded PDF and store its knowledge base."""
    try:
        # Save the uploaded file temporarily
        temp_file_path = f"temp/{uploaded_file.name}"
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getvalue())

        # Create a knowledge base for the PDF
        knowledge_base = PDFUrlKnowledgeBase(
            urls=[temp_file_path],
            vector_db=PgVector2(collection="financial_documents", db_url=db_url)
        )
        knowledge_base.load()

        st.success("Document processed and knowledge base created.")
        return knowledge_base
    except Exception as e:
        st.error(f"Error processing document: {str(e)}")
        return None
    

from phi.assistant import Assistant
from phi.storage.assistant.postgres import PgAssistantStorage

storage = ""#PgAssistantStorage(table_name="document_assistant", db_url=db_url)

def chat_with_document_agent(query):
    """Interact with the document using the AI agent."""
    try:
        assistant = Assistant(
            # model=Ollama(id="llama3.1"),
            model = OpenAIChat(id="gpt-4o-mini"),
            storage=storage,
            knowledge_base=st.session_state.knowledge_base,
            search_knowledge=True,
            read_chat_history=True,
            show_tool_calls=True
        )
        responses = assistant.run(query, stream=False)
        response = clean_response(responses)
        return response.content if hasattr(response, "content") else "No response available."
    except Exception as e:
        st.error(f"Error during chat: {str(e)}")
        return "An error occurred while processing your request."


# Initialize agents and session state
if 'agents_initialized' not in st.session_state:
    st.session_state.agents_initialized = False
    st.session_state.watchlist = set()
    st.session_state.analysis_history = []
    st.session_state.last_refresh = None


# Agents creation
def initialize_agents():
    """Initialize all agent instances with improved error handling"""
    # Check for API key first
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key and "openai_api_key" in st.session_state:
        api_key = st.session_state.openai_api_key
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
    
    # If no API key and we're trying to initialize, just return False without error
    if not api_key:
        return False
        
    # Only proceed with initialization if we have an API key
    if not st.session_state.agents_initialized and api_key:
        try:
            st.session_state.web_agent = Agent(
                name="Web Search Agent",
                role="Financial data researcher specialized in North American markets",
                model=OpenAIChat(id="gpt-4o-mini"),
                tools=[
                    GoogleSearch(fixed_language='english', fixed_max_results=8),
                    DuckDuckGo(fixed_max_results=8)
                ],
                instructions=[
                    "You are a financial data researcher specialized in North American financial markets.",
                    "Your primary task is to search and retrieve accurate, relevant, and up-to-date information about US/Canadian stocks, financial institutions, and market trends.",
                    "Focus exclusively on financial topics related to North American markets.",
                    "For company lookups, include the stock symbol, exchange, and basic financial data when available.",
                    "Always cite your sources with URLs where possible.",
                    "Organize information in a clear, concise manner using markdown formatting.",
                    "When searching for companies, prioritize results from official financial sources, earnings reports, and reputable financial news outlets.",
                    "If a query is ambiguous, seek clarification or provide information on the most likely interpretation.",
                    "Never share investment advice or make price predictions.",
                    "Avoid all non-financial topics completely."
                ],
                show_tool_calls=False,
                markdown=True
            )

            st.session_state.finance_agent = Agent(
                name="Financial AI Agent",
                role="Financial data analyst for North American markets",
                model=OpenAIChat(id="gpt-4o-mini"),
                tools=[
                    YFinanceTools(
                        stock_price=True,
                        company_news=True,
                        analyst_recommendations=True,
                        historical_prices=True
                    )
                ],
                instructions=[
                    "You are a financial data analyst specializing in North American financial markets.",
                    "Provide detailed financial analysis and insights based on verifiable data.",
                    "Always verify stock symbols before analysis and confirm the exchange is US or Canadian.",
                    "Include relevant ratios and metrics in your analysis, explaining their significance.",
                    "Format output using clear markdown with sections and bullet points where appropriate.",
                    "For company analysis, always include: current price, key financial metrics, recent performance, and relevant news.",
                    "When analyzing trends, explain the underlying factors and potential market implications.",
                    "Compare metrics to industry averages when relevant data is available.",
                    "For unclear queries about companies, try both the company name and potential stock symbols.",
                    "If data appears outdated, note this limitation in your response.",
                    "Avoid making specific investment recommendations or price predictions.",
                    "Always cite your data sources."
                ],
                show_tool_calls=False,
                markdown=True
            )

            st.session_state.multi_ai_agent = Agent(
                name='Stock Market Analyst',
                role='North American financial markets expert',
                model=OpenAIChat(id="gpt-4o-mini"),
                team=[st.session_state.web_agent, st.session_state.finance_agent],
                instructions=[
                    "You are a comprehensive financial markets expert specializing in North American stocks and financial institutions.",
                    "Coordinate data collection and analysis across your team of specialized agents.",
                    "For stock analysis, always verify the symbol and exchange first.",
                    "Provide context-rich, structured analysis using markdown formatting.",
                    "Include relevant financial metrics, recent news, and market context in all responses.",
                    "For comparisons between stocks, highlight key differences and similarities in a structured format.",
                    "When asked about trends, provide evidence-based explanations with supporting data.",
                    "For specific company questions, always include the stock symbol, current price, and key financial metrics.",
                    "Clearly attribute information to your sources.",
                    "Reject non-financial queries or topics outside North American markets.",
                    "Avoid making specific investment recommendations or price predictions.",
                    "Provide objective analysis without expressing personal opinions on investments.",
                    "When handling ambiguous company names, search for the most relevant matches.",
                    "Use visualizations and structured data presentation when applicable."
                ],
                show_tool_calls=False,
                markdown=True
            )

            st.session_state.agents_initialized = True
            return True
        except Exception as e:
            st.error(f"Error initializing agents: {str(e)}")
            return False
    
    # If we already have initialized agents, just return True
    elif st.session_state.agents_initialized:
        return True
    
    return False
        

def get_symbol_from_name(stock_input):
    """Enhanced function to fetch and validate stock symbol for US/Canadian markets
    Works with both stock symbols and company names"""
    try:
        stock_input = stock_input.strip()
        
        # Direct symbol lookup first (for efficiency)
        ticker = yf.Ticker(stock_input.upper())
        info = ticker.info
        
        # If we got valid info for direct lookup, return the symbol
        if info and 'exchange' in info and info['exchange'] in VALID_EXCHANGES:
            return stock_input.upper()
        
        
        # Create a list of possible matches to search
        # This list could be much larger in a production app
        common_companies = [
            {"symbol": "AAPL", "name": "Apple Inc."},
            {"symbol": "MSFT", "name": "Microsoft Corporation"},
            {"symbol": "AMZN", "name": "Amazon.com Inc."},
            {"symbol": "GOOGL", "name": "Alphabet Inc."},
            {"symbol": "META", "name": "Meta Platforms Inc."},
            {"symbol": "TSLA", "name": "Tesla Inc."},
            {"symbol": "NVDA", "name": "NVIDIA Corporation"},
            {"symbol": "JPM", "name": "JPMorgan Chase & Co."},
            {"symbol": "V", "name": "Visa Inc."},
            {"symbol": "WMT", "name": "Walmart Inc."},
            {"symbol": "BAC", "name": "Bank of America Corporation"},
            {"symbol": "PG", "name": "Procter & Gamble Company"},
            {"symbol": "DIS", "name": "The Walt Disney Company"},
            {"symbol": "NFLX", "name": "Netflix Inc."},
            {"symbol": "KO", "name": "The Coca-Cola Company"},
            {"symbol": "TD.TO", "name": "Toronto-Dominion Bank"},
            {"symbol": "RY.TO", "name": "Royal Bank of Canada"},
            {"symbol": "BMO.TO", "name": "Bank of Montreal"},
            {"symbol": "ENB.TO", "name": "Enbridge Inc."},
            {"symbol": "CM.TO", "name": "Canadian Imperial Bank of Commerce"}
        ]
        
        
        # Search for matches by company name
        company_names = [company["name"].lower() for company in common_companies]
        name_matches = process.extract(stock_input.lower(), company_names, limit=5)
        
        # If we have good matches (score > 80), suggest them
        good_matches = [match for match in name_matches if match[1] > 80]
        
        if good_matches:
            # Get the matching companies
            matching_companies = []
            for match in good_matches:
                matched_name = match[0]
                for company in common_companies:
                    if company["name"].lower() == matched_name:
                        matching_companies.append({
                            "symbol": company["symbol"],
                            "name": company["name"],
                            "score": match[1]
                        })
                        break
            
            # If we have exactly one good match with score > 90, use it directly
            if len(matching_companies) == 1 and matching_companies[0]["score"] > 90:
                return matching_companies[0]["symbol"]
            
            # Otherwise, show options to the user
            st.write("Did you mean one of these companies?")
            
            # Create columns for each match
            cols = st.columns(len(matching_companies))
            selected_symbol = None
            
            for i, col in enumerate(cols):
                if i < len(matching_companies):
                    with col:
                        company = matching_companies[i]
                        st.write(f"**{company['symbol']}**")
                        st.write(f"{company['name']}")
                        if st.button(f"Select", key=f"select_{i}"):
                            selected_symbol = company['symbol']
            
            # If user selected a symbol, return it
            if selected_symbol:
                return selected_symbol
            
            # If no selection made, try the first match
            if matching_companies:
                # Verify that the first match is valid
                ticker = yf.Ticker(matching_companies[0]["symbol"])
                info = ticker.info
                if info and 'exchange' in info:
                    return matching_companies[0]["symbol"]
        
        # If no good matches found, inform the user
        st.error(f"Could not find stock symbol for '{stock_input}'. Please try with an exact stock symbol.")
        return None
            
    except Exception as e:
        st.error(f"Error processing {stock_input}: {str(e)}")
        return None
    
def get_stock_data(symbol, period="1y"):
    """Fetch stock data for validated symbols"""
    try:
        stock = yf.Ticker(symbol)
        info = stock.info
        if not info or 'exchange' not in info or info['exchange'] not in VALID_EXCHANGES:
            raise ValueError(f"{symbol} is not a valid US/Canadian stock.")
        
        hist = stock.history(period=period, interval="1d", auto_adjust=True)
        if hist.empty:
            raise ValueError("No historical data available")
        
        return info, hist
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {str(e)}")
        return None, None


def create_price_chart(hist_data, symbol):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=hist_data.index,
        open=hist_data['Open'],
        high=hist_data['High'],
        low=hist_data['Low'],
        close=hist_data['Close'],
        name='Price'
    ))
    ma20 = hist_data['Close'].rolling(window=20).mean()
    ma50 = hist_data['Close'].rolling(window=50).mean()
    fig.add_trace(go.Scatter(x=hist_data.index, y=ma20, name='20 Day MA', line=dict(color='orange')))
    fig.add_trace(go.Scatter(x=hist_data.index, y=ma50, name='50 Day MA', line=dict(color='blue')))
    fig.update_layout(
        title=f'{symbol} Stock Price',
        yaxis_title='Price',
        template='plotly_white',
        xaxis_rangeslider_visible=False,
        height=600
    )
    return fig

def create_volume_chart(hist_data):
    volume_ma = hist_data['Volume'].rolling(window=20).mean()
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=hist_data.index,
        y=hist_data['Volume'],
        name='Volume',
        marker_color='rgb(0, 138, 0)'
    ))
    fig.add_trace(go.Scatter(
        x=hist_data.index,
        y=volume_ma,
        name='20 Day Volume MA',
        line=dict(color='red')
    ))
    fig.update_layout(
        title='Trading Volume Analysis',
        yaxis_title='Volume',
        template='plotly_white',
        height=400
    )
    return fig

def format_large_number(number):
    if number >= 1e12:
        return f"${number/1e12:.2f}T"
    elif number >= 1e9:
        return f"${number/1e9:.2f}B"
    elif number >= 1e6:
        return f"${number/1e6:.2f}M"
    else:
        return f"${number:,.2f}"

def display_metrics(info):
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        market_cap = info.get('marketCap', 'N/A')
        if market_cap != 'N/A':
            market_cap = format_large_number(market_cap)
        st.metric("Market Cap", market_cap)
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        pe_ratio = info.get('trailingPE', 'N/A')
        if pe_ratio != 'N/A':
            pe_ratio = f"{pe_ratio:.2f}"
        st.metric("P/E Ratio", pe_ratio)
        st.markdown('</div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        high = info.get('fiftyTwoWeekHigh', 'N/A')
        if high != 'N/A':
            high = f"${high:.2f}"
        st.metric("52 Week High", high)
        st.markdown('</div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        low = info.get('fiftyTwoWeekLow', 'N/A')
        if low != 'N/A':
            low = f"${low:.2f}"
        st.metric("52 Week Low", low)
        st.markdown('</div>', unsafe_allow_html=True)

def fetch_news(symbol):
    """Fetch news using YFinanceTools"""
    try:
        news_data = st.session_state.finance_agent.tools[0].company_news(symbol)
        return news_data if news_data else []
    except Exception as e:
        st.error(f"Error fetching news: {str(e)}")
        return []
    
def is_comparison_query(query):
    """Detect if the query is a comparison request and extract symbols and period."""
    query_lower = query.lower()
    if "compare" in query_lower and "and" in query_lower:
        parts = query_lower.split()
        try:
            compare_index = parts.index("compare")
            and_index = parts.index("and", compare_index + 1)
            symbol1 = parts[compare_index + 1].upper()
            symbol2 = parts[and_index + 1].upper()
            # Default to 5 years if no period specified
            period = "5y"
            if "last" in query_lower or "past" in query_lower:
                for i, part in enumerate(parts):
                    if part in ("last", "past") and i + 1 < len(parts) and parts[i + 1].isdigit():
                        num = int(parts[i + 1])
                        unit = parts[i + 2] if i + 2 < len(parts) else "years"
                        period = f"{num}y" if "year" in unit else f"{num}m" if "month" in unit else "5y"
            return True, symbol1, symbol2, period
        except (ValueError, IndexError):
            return False, None, None, "5y"
    return False, None, None, "5y"

def create_comparison_chart(hist1, hist2, symbol1, symbol2):
    """Create a normalized comparison chart for two stocks."""
    if hist1.empty or hist2.empty:
        return None
    initial_price1 = hist1['Close'].iloc[0]
    initial_price2 = hist2['Close'].iloc[0]
    normalized1 = (hist1['Close'] / initial_price1 - 1) * 100
    normalized2 = (hist2['Close'] / initial_price2 - 1) * 100
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist1.index, y=normalized1, name=symbol1, line=dict(color="#00cc96")))
    fig.add_trace(go.Scatter(x=hist2.index, y=normalized2, name=symbol2, line=dict(color="#ef553b")))
    fig.update_layout(
        title=f"Stock Price Comparison: {symbol1} vs {symbol2} (Normalized)",
        yaxis_title="Percentage Change (%)",
        template="plotly_white",
        height=400
    )
    return fig


import re

def clean_response(response):
    """
    Remove internal task transfer details from the AI agent's response.
    Handle RunResponse objects properly.
    """
    # Extract content from RunResponse object
    if hasattr(response, 'content'):
        content = response.content
    else:
        # Try to convert response to string if it's not a string already
        try:
            content = str(response)
        except:
            return "Unable to process response"
    
    # Define patterns to match internal task transfer details
    internal_patterns = [
        r"transfer_task_to_financial_ai_agent\(.*?\)",  # Matches task transfer instructions
        r"<\|python_tag\|>",  # Matches unwanted tags
        r"```",  # Matches code block markers
        r"response = .*?\n"  # Matches response assignment
    ]
    
    # Clean the content by removing matches
    for pattern in internal_patterns:
        content = re.sub(pattern, "", content)
    
    # Strip extra whitespace and return
    return content.strip()


# Custom CSS for enhanced header
st.markdown("""
    <style>
    /* Header Styling */
    .header-container {
        background: linear-gradient(135deg, #008000 0%, #00b300 100%);
        color: white;
        padding: 1.5rem;
        font-size: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    .header-title {
        font-size: 1.0rem;
        font-weight: semibold;
        margin-bottom: 1rem;
    }
    .header-subtitle {
        font-size: 1.2rem;
        color: rgba(255, 255, 255, 0.8);
    }
    </style>
""", unsafe_allow_html=True)

# Header Section
# with st.container():
#     # st.markdown('<div class="header-container">', unsafe_allow_html=True)
#     st.markdown('<h2 class="header-container">🤖 Advanced Stock Market Analysis</h2>', unsafe_allow_html=True)
#     st.markdown('<p class="header-subtitle"> Developed by <a href="https://www.linkedin.com/in/harbhajan21" target="_blank" style="color: #008000; text-decoration: none;">Data Solutions & Analytics Team</a> </p>', unsafe_allow_html=True)
#     st.markdown('</div>', unsafe_allow_html=True)


# Main function to run the Streamlit app
def main():
    if not check_login():
        return
    
    # Sidebar for API key and mode selection
    with st.sidebar:
        st.image("Logo.png", width=100, use_container_width=True)
        # st.logo("Logo.png", size="large", link=None, icon_image=None)
        
        # Add API Key input with improved handling
        st.markdown('<h4 class="header-container"> 🔑 API Key </h4>', unsafe_allow_html=True)
        api_key = st.text_input("Enter OpenAI API Key", 
                               type="password", 
                               key="api_key_input",
                               help="Required for AI analysis features")
        
        if api_key:
            # Store the API key in session state and environment
            st.session_state.openai_api_key = api_key
            os.environ["OPENAI_API_KEY"] = api_key
            
            # Initialize agents if we have an API key but haven't initialized yet
            if not st.session_state.agents_initialized:
                with st.spinner("Initializing AI agents..."):
                    initialize_agents()
        
        # Show a message when API key is missing
        elif "openai_api_key" not in st.session_state or not st.session_state.openai_api_key:
            st.info("⚠️ Please enter your OpenAI API key to use AI features")
        
        st.markdown('<h4 class="header-container"> ⚙️ Mode </h4>', unsafe_allow_html=True)
        analysis_type = st.selectbox(
            "Select Mode",
            ["Comprehensive Analysis", "Chat with AI Chatbot", "Chat with Documents"],
            label_visibility="visible",
            index=0,
        )
        st.markdown('<h4 class="header-container"> ℹ️ About </h4>', unsafe_allow_html=True)
        st.markdown("""
            This tool focuses on:
            - North American stock market analysis
            - AI-powered financial analysis
            - Comprehensive stock metrics
            - Document upload and chat interface
        """)
    
    # Check if we have an API key before proceeding with sections that need it
    has_api_key = "openai_api_key" in st.session_state and st.session_state.openai_api_key
    
    # Header Section
    with st.container():
        st.markdown('<h2 class="header-container">🤖 Advanced Stock Market Analysis</h2>', unsafe_allow_html=True)
        st.markdown('<p class="header-subtitle" style="color: #000000;"> Developed by <a href="https://www.linkedin.com/in/harbhajan21" target="_blank" style="color: #008A00; text-decoration: none;">Data Solutions & Analytics Team</a> </p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    

    # Main content
    if analysis_type == "Comprehensive Analysis":
        # Main content
        col1, col2 = st.columns([3, 1])
        with col1:
            stock_input = st.text_input(
                "Enter Stock Symbol or Company Name",
                placeholder="e.g., NVDA, Apple, TD.TO",
                help="Enter US/Canadian stock symbol or company name"
            )
        with col2:
            date_range = st.selectbox(
                "Time Range",
                ["1M", "3M", "6M", "1Y", "5Y"],
                index=3
            )
            period_map = {
                "1M": "1mo", "3M": "3mo", "6M": "6mo",
                "1Y": "1y", "5Y": "5y"
            }
            period = period_map[date_range]

        if st.button("🚀 Analyze", use_container_width=True):
            if not stock_input:
                st.error("Please enter a stock symbol or company name")
                return

            # Use enhanced function to get stock symbol
            with st.spinner("Searching for stock..."):
                stock_symbol = get_symbol_from_name(stock_input)
            
            if stock_symbol:
                try:
                    if initialize_agents():
                        with st.spinner(f"Analyzing {stock_symbol}..."):
                            info, hist = get_stock_data(stock_symbol, period=period)
                            if info and hist is not None:
                                # Record this analysis in history
                                if "analysis_history" not in st.session_state:
                                    st.session_state.analysis_history = []
                                
                                st.session_state.analysis_history.append({
                                    "symbol": stock_symbol,
                                    "name": info.get('shortName', stock_symbol),
                                    "timestamp": datetime.now(),
                                    "price": info.get('currentPrice', 'N/A')
                                })
                                
                                # Limit history to last 10 entries
                                if len(st.session_state.analysis_history) > 10:
                                    st.session_state.analysis_history = st.session_state.analysis_history[-10:]
                                
                                # Display market status
                                market_status = "🟢 Market Open" if info.get('regularMarketOpen') else "🔴 Market Closed"
                                st.markdown(f"<div class='market-indicator'>{market_status}</div>", unsafe_allow_html=True)
                                
                                # Rest of your display code remains the same
                                overview_tab, charts_tab, analysis_tab = st.tabs([
                                    "📊 Overview", 
                                    "📈 Charts", 
                                    "🤖 AI Analysis"
                                ])
                                
                                with overview_tab:
                                    st.markdown("### Company Overview")
                                    st.write(info.get('longBusinessSummary', 'No description available.'))
                                    st.markdown("### Key Metrics")
                                    display_metrics(info)
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.markdown("### Company Details")
                                        st.write(f"Sector: {info.get('sector', 'N/A')}")
                                        st.write(f"Industry: {info.get('industry', 'N/A')}")
                                        st.write(f"Country: {info.get('country', 'N/A')}")
                                        st.write(f"Employees: {info.get('fullTimeEmployees', 'N/A'):,}")
                                        st.write(f"Headquarter: {info.get('address1', 'N/A')}, {info.get('city', 'N/A')}, {info.get('state', 'N/A')}, {info.get('country', 'N/A')}, {info.get('zip', 'N/A')}")
                                    with col2:
                                        st.markdown("### Trading Information")
                                        st.write(f"Exchange: {VALID_EXCHANGES.get(info.get('exchange', 'N/A'), 'N/A')}")
                                        st.write(f"Currency: {info.get('currency', 'N/A')}")
                                        st.write(f"Volume: {info.get('volume', 'N/A'):,}")
                                        st.write(f"Market Cap: {format_large_number(info.get('marketCap', 0))}")
                                        st.write(f"Dividend Yield: {info.get('dividendYield', 'N/A')}")
                                
                                with charts_tab:
                                    st.markdown("### Price Analysis")
                                    price_chart = create_price_chart(hist, stock_symbol)
                                    st.plotly_chart(price_chart, use_container_width=True)
                                    volume_chart = create_volume_chart(hist)
                                    st.plotly_chart(volume_chart, use_container_width=True)
                                    st.markdown("### Technical Indicators")
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        rsi = hist['Close'].diff()
                                        rsi_pos = rsi.copy()
                                        rsi_neg = rsi.copy()
                                        rsi_pos[rsi_pos < 0] = 0
                                        rsi_neg[rsi_neg > 0] = 0
                                        rsi_14_pos = rsi_pos.rolling(window=14).mean()
                                        rsi_14_neg = abs(rsi_neg.rolling(window=14).mean())
                                        rsi_14 = 100 - (100 / (1 + rsi_14_pos / rsi_14_neg))
                                        st.metric("RSI (14)", f"{rsi_14.iloc[-1]:.2f}")
                                    with col2:
                                        ma20 = hist['Close'].rolling(window=20).mean()
                                        ma50 = hist['Close'].rolling(window=50).mean()
                                        cross_signal = "Bullish" if ma20.iloc[-1] > ma50.iloc[-1] else "Bearish"
                                        st.metric("MA Cross Signal", cross_signal)
                                    with col3:
                                        volatility = hist['Close'].pct_change().std() * (252 ** 0.5) * 100
                                        st.metric("Annualized Volatility", f"{volatility:.2f}%")
                                
                            
                                with analysis_tab:
                                    st.markdown("### AI-Powered Analysis")
                                    query = f"Provide a {analysis_type.lower()} for {stock_symbol}."
                                    
                                    try:
                                        responses = st.session_state.multi_ai_agent.run(query, stream=False)
                                        analysis_content = clean_response(responses)
                                        
                                        # If empty, provide a default message
                                        if not analysis_content:
                                            analysis_content = "No analysis available"
                                            
                                        st.markdown(analysis_content)
                                    except Exception as e:
                                        st.error(f"Analysis error: {str(e)}")
                                        st.markdown("Unable to generate analysis at this time. Please try again later.")

                                
                                # if st.button("🔄 Refresh Data"):
                                #     st.session_state.last_refresh = datetime.now()
                                #     st.experimental_rerun()

                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")


    elif analysis_type == "Chat with AI Chatbot":
        if not has_api_key:
            st.warning("Please enter your OpenAI API key in the sidebar to use the chatbot")
        else:
            st.markdown("### Stock Market Chatbot")
            st.success("Ask about North American stocks, e.g., 'What is the price of TD Bank?' or 'Compare TD and RY for the last 2 years'.")
            
            # Initialize chat history
            if "stock_chat_history" not in st.session_state:
                st.session_state.stock_chat_history = []
            
            # Chat input
            user_input = st.chat_input("Type your question here...")
            
            if user_input:
                # Add user message
                st.session_state.stock_chat_history.append({"role": "user", "content": user_input})
                
                # Check for comparison query
                is_compare, symbol1, symbol2, period = is_comparison_query(user_input)
                
                if is_compare:
                    symbol1_valid = get_symbol_from_name(symbol1)
                    symbol2_valid = get_symbol_from_name(symbol2)
                    
                    if symbol1_valid and symbol2_valid:
                        with st.spinner("Fetching data..."):
                            # Fetch historical data
                            _, hist1 = get_stock_data(symbol1_valid, period=period)
                            _, hist2 = get_stock_data(symbol2_valid, period=period)
                            
                            if hist1 is not None and hist2 is not None:
                                fig = create_comparison_chart(hist1, hist2, symbol1_valid, symbol2_valid)
                                if fig:
                                    response = f"Here is the stock price comparison for {symbol1_valid} and {symbol2_valid} over the last {period}:"
                                    st.session_state.stock_chat_history.append({
                                        "role": "assistant",
                                        "content": response,
                                        "chart": fig
                                    })
                                else:
                                    st.session_state.stock_chat_history.append({
                                        "role": "assistant",
                                        "content": "Unable to generate comparison chart due to missing data."
                                    })
                            else:
                                st.session_state.stock_chat_history.append({
                                    "role": "assistant",
                                    "content": "Failed to fetch data for one or both stocks."
                                })
                    else:
                        st.session_state.stock_chat_history.append({
                            "role": "assistant",
                            "content": "One or both stock symbols are invalid."
                        })
                else:
                    # General query
                    with st.spinner("Thinking..."):
                        if hasattr(st.session_state, 'multi_ai_agent'):
                            response = st.session_state.multi_ai_agent.run(user_input, stream=False)

                            # Extract and clean the response content
                            if hasattr(response, 'content') and isinstance(response.content, str):
                                cleaned_content = clean_response(response.content)
                            else:
                                cleaned_content = "Error: Unable to process response."
                            
                            # Add the cleaned response to chat history
                            st.session_state.stock_chat_history.append({
                                "role": "assistant",
                                "content": cleaned_content
                            })
                        else:
                            st.session_state.stock_chat_history.append({
                                "role": "assistant",
                                "content": "Error: AI agent not initialized properly. Please refresh the page."
                            })
            
            # Display chat history
            for message in st.session_state.stock_chat_history:
                with st.chat_message(message["role"]):
                    if "content" in message:
                        st.markdown(message["content"])
                    if "chart" in message:
                        st.plotly_chart(message["chart"], use_container_width=True)

            pass

    elif analysis_type == "Chat with Documents":
        st.markdown("### Upload PDF Document")
        uploaded_files = st.file_uploader("Upload a PDF file (max 5MB)", type=["pdf"], accept_multiple_files=False)

        if uploaded_files:
            # Check file size
            if uploaded_files.size > 5 * 1024 * 1024:  # 5MB limit
                st.error("File size exceeds 5MB. Please upload a smaller file.")
            else:
                # Save the uploaded file temporarily
                temp_file_path = f"temp_{uploaded_files.name}"
                with open(temp_file_path, "wb") as f:
                    f.write(uploaded_files.getbuffer())

                # Load the PDF document
                try:
                    loader = PyPDFLoader(temp_file_path)
                    documents = loader.load()
                    if not documents:
                        st.error("No content could be extracted from the PDF. Please check the file.")
                        return
                except Exception as e:
                    st.error(f"Error loading PDF: {str(e)}")
                    return

                # Split documents into chunks
                try:
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                    chunks = text_splitter.split_documents(documents)
                    if not chunks:
                        st.error("No text chunks were generated from the document. The PDF may be empty or unreadable.")
                        return
                except Exception as e:
                    st.error(f"Error splitting document: {str(e)}")
                    return

                # Generate embeddings and store in FAISS
                try:
                    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
                    # Test embedding generation with a small sample
                    sample_text = chunks[0].page_content if chunks else "Test"
                    sample_embedding = embeddings.embed_query(sample_text)
                    if not sample_embedding:
                        st.error("Failed to generate embeddings. Check your OpenAI API key or network connection.")
                        return
                    
                    st.session_state.vector_store = FAISS.from_documents(chunks, embeddings)
                    st.success(f"Uploaded and processed {uploaded_files.name}")
                except Exception as e:
                    st.error(f"Error creating vector store: {str(e)}")
                    return

        # Chat Interface
        if st.session_state.vector_store is not None:
            st.markdown("### Chat with Uploaded Document")
            user_input = st.chat_input("Ask a question about the document...")

            if user_input:
                # Define prompt template
                prompt_template = ChatPromptTemplate.from_template("""
                Answer the following question based only on the provided context. 
                Think step by step before providing a detailed answer. 
                Restrict responses to financial advising, economics, or banking topics.
                <context>
                {context}
                </context>

                Question: {input}""")

                # Create retrieval chain
                if st.session_state.vector_store is not None:
                    if not has_api_key:
                        st.warning("Please enter your OpenAI API key to chat with the document")
                    else:
                        retriever = st.session_state.vector_store.as_retriever()

                        # Initialize OpenAI model
                        llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY)

                        # Create document chain
                        document_chain = create_stuff_documents_chain(llm, prompt_template)

                        # Create retrieval chain
                        retrieval_chain = create_retrieval_chain(retriever, document_chain)

                        # Get response from the retrieval chain
                        try:
                            response = retrieval_chain.invoke({"input": user_input})
                            answer = response["answer"]

                            # Update chat history
                            st.session_state.chat_history.append({"role": "user", "content": user_input})
                            st.session_state.chat_history.append({"role": "assistant", "content": answer})

                            # Display chat messages
                            for message in st.session_state.chat_history:
                                with st.chat_message(message["role"]):
                                    st.markdown(message["content"])

                            # Display relevant document chunks
                            with st.expander("Document Similarity Search"):
                                for i, doc in enumerate(response["context"]):
                                    st.write(doc.page_content)
                                    st.write("--------------------------------")
                        except Exception as e:
                            st.error(f"Error processing query: {str(e)}")

                        pass

    # Analysis History
    if st.session_state.analysis_history:
        st.markdown("---")
        st.markdown("### Recent Analysis History")
        history_df = pd.DataFrame(st.session_state.analysis_history)
        history_df['timestamp'] = history_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
        st.dataframe(history_df, use_container_width=True)

    st.markdown("---")
    
    if st.session_state.last_refresh:
        st.markdown(f"<div class='market-indicator'>Last refreshed: {st.session_state.last_refresh.strftime('%Y-%m-%d %H:%M:%S')}</div>", 
                unsafe_allow_html=True)

if __name__ == "__main__":
    main()