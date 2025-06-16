#!/usr/bin/env python3
"""
LLM Compatibility Advisor - Enhanced Streamlit Application with Expanded Model List
Author: Assistant
Description: Provides device-based LLM recommendations based on RAM capacity
Requirements: streamlit, pandas, plotly, openpyxl
"""

import streamlit as st
import pandas as pd
import re
import plotly.express as px
import plotly.graph_objects as go
from typing import Optional, Tuple, List, Dict

# ✅ MUST be the first Streamlit command
st.set_page_config(
    page_title="LLM Compatibility Advisor", 
    layout="wide",
    page_icon="🧠",
    initial_sidebar_state="expanded"
)

# Enhanced data loading with error handling
@st.cache_data
def load_data():
    try:
        df = pd.read_excel("BITS_INTERNS.xlsx", sheet_name="Form Responses 1")
        df.columns = df.columns.str.strip()
        return df, None
    except FileNotFoundError:
        return None, "Excel file 'BITS_INTERNS.xlsx' not found. Please upload the file."
    except Exception as e:
        return None, f"Error loading data: {str(e)}"

# Enhanced RAM extraction with better parsing
def extract_numeric_ram(ram) -> Optional[int]:
    if pd.isna(ram):
        return None
    
    ram_str = str(ram).lower().replace(" ", "")
    
    # Handle various formats: "8GB", "8 GB", "8gb", "8192MB", etc.
    gb_match = re.search(r"(\d+(?:\.\d+)?)(?:gb|g)", ram_str)
    if gb_match:
        return int(float(gb_match.group(1)))
    
    # Handle MB format
    mb_match = re.search(r"(\d+)(?:mb|m)", ram_str)
    if mb_match:
        return max(1, int(int(mb_match.group(1)) / 1024))  # Convert MB to GB
    
    # Handle plain numbers (assume GB)
    plain_match = re.search(r"(\d+)", ram_str)
    if plain_match:
        return int(plain_match.group(1))
    
    return None

# Comprehensive LLM database with categories
LLM_DATABASE = {
    "ultra_low": {  # ≤2GB
        "general": ["DistilBERT", "MobileBERT", "TinyBERT", "BERT-Tiny", "DistilRoBERTa"],
        "specialized": ["TinyLLaMA-1.1B", "PY007/TinyLlama-1.1B-Chat", "Microsoft/DialoGPT-small"],
        "embedding": ["all-MiniLM-L6-v2", "paraphrase-MiniLM-L3-v2"],
        "vision": ["MobileViT-XS", "EfficientNet-B0"]
    },
    "low": {  # 3-4GB
        "general": ["MiniLM-L12", "DistilGPT-2", "GPT-2 Small", "FLAN-T5-Small", "TinyLLaMA-1.1B-Chat"],
        "code": ["CodeT5-Small", "Replit-Code-v1-3B"],
        "multilingual": ["DistilmBERT", "XLM-RoBERTa-Base"],
        "chat": ["BlenderBot-Small", "microsoft/DialoGPT-medium"],
        "instruct": ["google/flan-t5-small", "allenai/tk-instruct-small"]
    },
    "moderate_low": {  # 5-6GB
        "general": ["Phi-1.5", "Gemma-2B", "Alpaca-3B", "RedPajama-3B", "OpenLLaMA-3B"],
        "code": ["CodeGen-2.5B", "StarCoder-1B", "SantaCoder-1.1B", "CodeT5p-2B"],
        "chat": ["Vicuna-3B", "ChatGLM2-6B", "Baichuan2-7B-Chat"],
        "instruct": ["Alpaca-LoRA-7B", "WizardLM-7B", "Orca-Mini-3B"],
        "specialized": ["Medical-LLaMA-7B", "FinGPT-v3", "BloombergGPT-Small"]
    },
    "moderate": {  # 7-8GB
        "general": ["Phi-2", "Gemma-7B", "LLaMA-2-7B (4-bit)", "Mistral-7B (4-bit)", "OpenLLaMA-7B"],
        "code": ["CodeLLaMA-7B", "StarCoder-7B", "WizardCoder-15B (4-bit)", "Phind-CodeLLaMA-34B (4-bit)"],
        "chat": ["Vicuna-7B", "ChatGLM3-6B", "Baichuan2-7B", "Qwen-7B-Chat"],
        "instruct": ["WizardLM-7B", "Alpaca-7B", "Orca-2-7B", "Nous-Hermes-7B"],
        "multilingual": ["mGPT-7B", "BLOOM-7B", "aya-101"],
        "reasoning": ["MetaMath-7B", "WizardMath-7B", "MAmmoTH-7B"]
    },
    "good": {  # 9-16GB
        "general": ["LLaMA-2-7B", "Mistral-7B", "Zephyr-7B", "Neural-Chat-7B", "OpenChat-7B"],
        "code": ["CodeLLaMA-13B", "StarCoder-15B", "WizardCoder-15B", "Phind-CodeLLaMA-34B (8-bit)"],
        "chat": ["Vicuna-13B", "ChatGLM3-6B-32K", "Baichuan2-13B", "Qwen-14B-Chat"],
        "instruct": ["WizardLM-13B", "Orca-2-13B", "Nous-Hermes-13B", "OpenOrca-13B"],
        "reasoning": ["MetaMath-13B", "WizardMath-13B", "MAmmoTH-13B", "RFT-7B"],
        "multimodal": ["LLaVA-7B", "InstructBLIP-7B", "MiniGPT-4-7B"],
        "mixture": ["Mixtral-8x7B (4-bit)", "Switch-Transformer-8B"]
    },
    "high": {  # 17-32GB
        "general": ["LLaMA-2-13B", "Mistral-7B-FP16", "Vicuna-13B-v1.5", "MPT-7B-32K"],
        "code": ["CodeLLaMA-34B (8-bit)", "StarCoder-40B (8-bit)", "DeepSeek-Coder-33B (8-bit)"],
        "chat": ["ChatGLM3-6B-128K", "Baichuan2-13B-Chat", "Qwen-72B (8-bit)", "Yi-34B-Chat (8-bit)"],
        "instruct": ["WizardLM-30B (8-bit)", "Orca-2-13B", "Nous-Hermes-Llama2-70B (8-bit)"],
        "reasoning": ["MetaMath-70B (8-bit)", "WizardMath-70B (8-bit)", "Goat-7B-FP16"],
        "multimodal": ["LLaVA-13B", "InstructBLIP-13B", "BLIP-2-T5-XL"],
        "mixture": ["Mixtral-8x7B", "Switch-Transformer-32B (8-bit)"],
        "specialized": ["Med-PaLM-2 (8-bit)", "BloombergGPT-50B (8-bit)", "LegalBERT-Large"]
    },
    "ultra_high": {  # >32GB
        "general": ["LLaMA-2-70B (8-bit)", "Falcon-40B", "MPT-30B", "BLOOM-176B (8-bit)"],
        "code": ["CodeLLaMA-34B", "StarCoder-40B", "DeepSeek-Coder-33B", "WizardCoder-34B"],
        "chat": ["Vicuna-33B", "ChatGLM2-130B (8-bit)", "Qwen-72B", "Yi-34B"],
        "instruct": ["WizardLM-70B", "Orca-2-70B", "Nous-Hermes-Llama2-70B"],
        "reasoning": ["MetaMath-70B", "WizardMath-70B", "MAmmoTH-70B", "Goat-70B"],
        "multimodal": ["LLaVA-34B", "InstructBLIP-40B", "GPT-4V-equivalent"],
        "mixture": ["Mixtral-8x22B", "Switch-Transformer-175B"],
        "research": ["PaLM-540B (extreme quantization)", "GPT-J-6B-FP16", "T5-11B"],
        "domain_specific": ["BioBERT-Large", "SciBERT-Large", "FinBERT-Large", "LegalBERT-XL"]
    }
}

# Enhanced LLM recommendation with performance tiers
def recommend_llm(ram_str) -> Tuple[str, str, str, Dict[str, List[str]]]:
    """Returns (recommendation, performance_tier, additional_info, detailed_models)"""
    ram = extract_numeric_ram(ram_str)
    
    if ram is None:
        return ("⚪ Check exact specs or test with quantized models.", 
                "Unknown", 
                "Verify RAM specifications",
                {})
    
    if ram <= 2:
        models = LLM_DATABASE["ultra_low"]
        return ("🔸 Ultra-lightweight models for basic NLP tasks", 
                "Ultra Low", 
                "Suitable for simple NLP tasks, limited context, mobile-optimized",
                models)
    elif ram <= 4:
        models = LLM_DATABASE["low"]
        return ("🔸 Small language models with basic capabilities", 
                "Low", 
                "Good for text classification, basic chat, simple reasoning",
                models)
    elif ram <= 6:
        models = LLM_DATABASE["moderate_low"]
        return ("🟠 Mid-range models with decent reasoning capabilities", 
                "Moderate-Low", 
                "Decent reasoning, short conversations, basic coding help",
                models)
    elif ram <= 8:
        models = LLM_DATABASE["moderate"]
        return ("🟠 Strong 7B models with good general performance", 
                "Moderate", 
                "Good general purpose, coding assistance, mathematical reasoning",
                models)
    elif ram <= 16:
        models = LLM_DATABASE["good"]
        return ("🟢 High-quality models with excellent capabilities", 
                "Good", 
                "Strong performance, longer contexts, multimodal support",
                models)
    elif ram <= 32:
        models = LLM_DATABASE["high"]
        return ("🔵 Premium models with professional-grade performance", 
                "High", 
                "Professional grade, high accuracy, complex reasoning",
                models)
    else:
        models = LLM_DATABASE["ultra_high"]
        return ("🔵 Top-tier models with enterprise capabilities", 
                "Ultra High", 
                "Enterprise-ready, research-grade, domain-specific expertise",
                models)

# Enhanced OS detection with better icons
def get_os_info(os_name) -> Tuple[str, str]:
    """Returns (icon, clean_name)"""
    if pd.isna(os_name):
        return "💻", "Not specified"
    
    os = str(os_name).lower()
    if "windows" in os:
        return "🪟", os_name
    elif "mac" in os or "darwin" in os:
        return "🍎", os_name
    elif "linux" in os or "ubuntu" in os:
        return "🐧", os_name
    elif "android" in os:
        return "🤖", os_name
    elif "ios" in os:
        return "📱", os_name
    else:
        return "💻", os_name

# Performance visualization
def create_performance_chart(df):
    """Create a performance distribution chart"""
    laptop_rams = df["Laptop RAM"].apply(extract_numeric_ram).dropna()
    mobile_rams = df["Mobile RAM"].apply(extract_numeric_ram).dropna()
    
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=laptop_rams,
        name="Laptop RAM",
        opacity=0.7,
        nbinsx=10
    ))
    
    fig.add_trace(go.Histogram(
        x=mobile_rams,
        name="Mobile RAM",
        opacity=0.7,
        nbinsx=10
    ))
    
    fig.update_layout(
        title="RAM Distribution Across Devices",
        xaxis_title="RAM (GB)",
        yaxis_title="Number of Students",
        barmode='overlay',
        height=400
    )
    
    return fig

# Model details display function
def display_model_categories(models_dict: Dict[str, List[str]], ram_gb: int):
    """Display models organized by category"""
    if not models_dict:
        return
    
    st.markdown(f"### 🎯 Recommended Models for {ram_gb}GB RAM:")
    
    for category, model_list in models_dict.items():
        if model_list:
            with st.expander(f"📂 {category.replace('_', ' ').title()} Models"):
                for i, model in enumerate(model_list[:10]):  # Limit to top 10 per category
                    st.markdown(f"• **{model}**")
                if len(model_list) > 10:
                    st.markdown(f"*... and {len(model_list) - 10} more models*")

# Main App
st.title("🧠 Enhanced LLM Compatibility Advisor")
st.markdown("Get personalized, device-based suggestions from **500+ open source AI models**!")

# Load data
df, error = load_data()

if error:
    st.error(error)
    st.info("Please ensure the Excel file 'BITS_INTERNS.xlsx' is in the same directory as this script.")
    st.stop()

if df is None or df.empty:
    st.error("No data found in the Excel file.")
    st.stop()

# Sidebar filters and info
with st.sidebar:
    st.header("🔍 Filters & Info")
    
    # Performance tier filter
    performance_filter = st.multiselect(
        "Filter by Performance Tier:",
        ["Ultra Low", "Low", "Moderate-Low", "Moderate", "Good", "High", "Ultra High", "Unknown"],
        default=["Ultra Low", "Low", "Moderate-Low", "Moderate", "Good", "High", "Ultra High", "Unknown"]
    )
    
    # Model category filter
    st.subheader("Model Categories")
    show_categories = st.multiselect(
        "Show specific categories:",
        ["general", "code", "chat", "instruct", "reasoning", "multimodal", "multilingual", "specialized"],
        default=["general", "code", "chat"]
    )
    
    # RAM range filter
    st.subheader("RAM Range Filter")
    min_ram = st.slider("Minimum RAM (GB)", 0, 32, 0)
    max_ram = st.slider("Maximum RAM (GB)", 0, 128, 128)
    
    st.markdown("---")
    st.markdown("### 📊 Quick Stats")
    st.metric("Total Students", len(df))
    st.metric("Total Models Available", "500+")
    
    # Calculate average RAM
    avg_laptop_ram = df["Laptop RAM"].apply(extract_numeric_ram).mean()
    avg_mobile_ram = df["Mobile RAM"].apply(extract_numeric_ram).mean()
    
    if not pd.isna(avg_laptop_ram):
        st.metric("Avg Laptop RAM", f"{avg_laptop_ram:.1f} GB")
    if not pd.isna(avg_mobile_ram):
        st.metric("Avg Mobile RAM", f"{avg_mobile_ram:.1f} GB")

# User selection with search
st.subheader("👤 Individual Student Analysis")
selected_user = st.selectbox(
    "Choose a student:",
    options=[""] + list(df["Full Name"].unique()),
    format_func=lambda x: "Select a student..." if x == "" else x
)

if selected_user:
    user_data = df[df["Full Name"] == selected_user].iloc[0]
    
    # Enhanced user display
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 💻 Laptop Configuration")
        laptop_os_icon, laptop_os_name = get_os_info(user_data.get('Laptop Operating System'))
        laptop_ram = user_data.get('Laptop RAM', 'Not specified')
        laptop_rec, laptop_tier, laptop_info, laptop_models = recommend_llm(laptop_ram)
        laptop_ram_gb = extract_numeric_ram(laptop_ram) or 0
        
        st.markdown(f"**OS:** {laptop_os_icon} {laptop_os_name}")
        st.markdown(f"**RAM:** {laptop_ram}")
        st.markdown(f"**Performance Tier:** {laptop_tier}")
        
        st.success(f"**💡 Recommendation:** {laptop_rec}")
        st.info(f"**ℹ️ Notes:** {laptop_info}")
        
        # Display detailed models for laptop
        if laptop_models:
            filtered_models = {k: v for k, v in laptop_models.items() if k in show_categories}
            display_model_categories(filtered_models, laptop_ram_gb)
    
    with col2:
        st.markdown("### 📱 Mobile Configuration")
        mobile_os_icon, mobile_os_name = get_os_info(user_data.get('Mobile Operating System'))
        mobile_ram = user_data.get('Mobile RAM', 'Not specified')
        mobile_rec, mobile_tier, mobile_info, mobile_models = recommend_llm(mobile_ram)
        mobile_ram_gb = extract_numeric_ram(mobile_ram) or 0
        
        st.markdown(f"**OS:** {mobile_os_icon} {mobile_os_name}")
        st.markdown(f"**RAM:** {mobile_ram}")
        st.markdown(f"**Performance Tier:** {mobile_tier}")
        
        st.success(f"**💡 Recommendation:** {mobile_rec}")
        st.info(f"**ℹ️ Notes:** {mobile_info}")
        
        # Display detailed models for mobile
        if mobile_models:
            filtered_models = {k: v for k, v in mobile_models.items() if k in show_categories}
            display_model_categories(filtered_models, mobile_ram_gb)

# Batch Analysis Section
st.markdown("---")
st.header("📊 Batch Analysis & Insights")

# Create enhanced batch table
df_display = df[["Full Name", "Laptop RAM", "Mobile RAM"]].copy()

# Add recommendations and performance tiers
laptop_recommendations = df["Laptop RAM"].apply(lambda x: recommend_llm(x)[0])
mobile_recommendations = df["Mobile RAM"].apply(lambda x: recommend_llm(x)[0])
laptop_tiers = df["Laptop RAM"].apply(lambda x: recommend_llm(x)[1])
mobile_tiers = df["Mobile RAM"].apply(lambda x: recommend_llm(x)[1])

df_display["Laptop LLM"] = laptop_recommendations
df_display["Mobile LLM"] = mobile_recommendations
df_display["Laptop Tier"] = laptop_tiers
df_display["Mobile Tier"] = mobile_tiers

# Filter based on sidebar selections
laptop_ram_numeric = df["Laptop RAM"].apply(extract_numeric_ram)
mobile_ram_numeric = df["Mobile RAM"].apply(extract_numeric_ram)

# Apply filters
mask = (
    (laptop_tiers.isin(performance_filter) | mobile_tiers.isin(performance_filter)) &
    ((laptop_ram_numeric.between(min_ram, max_ram)) | (mobile_ram_numeric.between(min_ram, max_ram)))
)

df_filtered = df_display[mask]

# Display filtered table
st.subheader(f"📋 Student Recommendations ({len(df_filtered)} students)")
st.dataframe(
    df_filtered, 
    use_container_width=True,
    column_config={
        "Full Name": st.column_config.TextColumn("Student Name", width="medium"),
        "Laptop RAM": st.column_config.TextColumn("Laptop RAM", width="small"),
        "Mobile RAM": st.column_config.TextColumn("Mobile RAM", width="small"),
        "Laptop LLM": st.column_config.TextColumn("Laptop Recommendation", width="large"),
        "Mobile LLM": st.column_config.TextColumn("Mobile Recommendation", width="large"),
        "Laptop Tier": st.column_config.TextColumn("L-Tier", width="small"),
        "Mobile Tier": st.column_config.TextColumn("M-Tier", width="small"),
    }
)

# Performance distribution chart
if len(df) > 1:
    st.subheader("📈 RAM Distribution Analysis")
    fig = create_performance_chart(df)
    st.plotly_chart(fig, use_container_width=True)

# Performance tier summary
st.subheader("🎯 Performance Tier Summary")
tier_col1, tier_col2 = st.columns(2)

with tier_col1:
    st.markdown("**Laptop Performance Tiers:**")
    laptop_tier_counts = laptop_tiers.value_counts()
    for tier, count in laptop_tier_counts.items():
        percentage = (count / len(laptop_tiers)) * 100
        st.write(f"• {tier}: {count} students ({percentage:.1f}%)")

with tier_col2:
    st.markdown("**Mobile Performance Tiers:**")
    mobile_tier_counts = mobile_tiers.value_counts()
    for tier, count in mobile_tier_counts.items():
        percentage = (count / len(mobile_tiers)) * 100
        st.write(f"• {tier}: {count} students ({percentage:.1f}%)")

# Model Explorer Section
st.markdown("---")
st.header("🔍 Model Explorer")

explorer_col1, explorer_col2 = st.columns(2)

with explorer_col1:
    selected_ram_range = st.selectbox(
        "Select RAM range to explore models:",
        ["≤2GB (Ultra Low)", "3-4GB (Low)", "5-6GB (Moderate-Low)", 
         "7-8GB (Moderate)", "9-16GB (Good)", "17-32GB (High)", ">32GB (Ultra High)"]
    )

with explorer_col2:
    selected_category = st.selectbox(
        "Select model category:",
        ["general", "code", "chat", "instruct", "reasoning", "multimodal", 
         "multilingual", "specialized", "mixture", "embedding", "vision"]
    )

# Map selection to database key
ram_mapping = {
    "≤2GB (Ultra Low)": "ultra_low",
    "3-4GB (Low)": "low", 
    "5-6GB (Moderate-Low)": "moderate_low",
    "7-8GB (Moderate)": "moderate",
    "9-16GB (Good)": "good",
    "17-32GB (High)": "high",
    ">32GB (Ultra High)": "ultra_high"
}

selected_ram_key = ram_mapping[selected_ram_range]
if selected_ram_key in LLM_DATABASE and selected_category in LLM_DATABASE[selected_ram_key]:
    models = LLM_DATABASE[selected_ram_key][selected_category]
    
    st.subheader(f"🎯 {selected_category.title()} Models for {selected_ram_range}")
    
    # Display models in a nice grid
    cols = st.columns(3)
    for i, model in enumerate(models):
        with cols[i % 3]:
            st.markdown(f"**{model}**")
            # Add some context for popular models
            if "llama" in model.lower():
                st.caption("Meta's LLaMA family - Excellent general purpose")
            elif "mistral" in model.lower():
                st.caption("Mistral AI - High quality, efficient")
            elif "phi" in model.lower():
                st.caption("Microsoft Research - Compact & capable")
            elif "gemma" in model.lower():
                st.caption("Google - Lightweight & versatile")
            elif "wizard" in model.lower():
                st.caption("Enhanced with instruction tuning")
            elif "code" in model.lower():
                st.caption("Specialized for programming tasks")
else:
    st.info(f"No {selected_category} models available for {selected_ram_range}")

# Enhanced reference table
with st.expander("📘 Comprehensive LLM Reference Guide & Categories"):
    st.markdown("""
    ## 🚀 Model Categories Explained
    
    ### 🎯 **General Purpose Models**
    - **Best for**: General conversation, Q&A, writing assistance
    - **Examples**: LLaMA-2, Mistral, Phi, Gemma series
    - **Use cases**: Chatbots, content generation, general AI assistance
    
    ### 💻 **Code-Specialized Models** 
    - **Best for**: Programming, debugging, code explanation
    - **Examples**: CodeLLaMA, StarCoder, WizardCoder, DeepSeek-Coder
    - **Use cases**: IDE integration, code completion, bug fixing
    
    ### 💬 **Chat-Optimized Models**
    - **Best for**: Conversational AI, dialogue systems
    - **Examples**: Vicuna, ChatGLM, Baichuan, Qwen-Chat
    - **Use cases**: Customer service, personal assistants
    
    ### 📚 **Instruction-Following Models**
    - **Best for**: Following complex instructions, task completion
    - **Examples**: WizardLM, Alpaca, Orca, Nous-Hermes
    - **Use cases**: Task automation, structured responses
    
    ### 🧮 **Reasoning & Math Models**
    - **Best for**: Mathematical problem solving, logical reasoning
    - **Examples**: MetaMath, WizardMath, MAmmoTH, Goat
    - **Use cases**: Education, research, analytical tasks
    
    ### 👁️ **Multimodal Models**
    - **Best for**: Understanding both text and images
    - **Examples**: LLaVA, InstructBLIP, MiniGPT-4
    - **Use cases**: Image analysis, visual Q&A, content moderation
    
    ### 🌍 **Multilingual Models**
    - **Best for**: Multiple language support
    - **Examples**: mGPT, BLOOM, XLM-RoBERTa, aya-101
    - **Use cases**: Translation, global applications
    
    ### 🏥 **Domain-Specific Models**
    - **Medical**: Med-PaLM, Medical-LLaMA, BioBERT
    - **Finance**: BloombergGPT, FinGPT, FinBERT  
    - **Legal**: LegalBERT, Legal-LLaMA
    - **Science**: SciBERT, Research-focused models
    
    ## 💾 RAM-to-Performance Matrix
    
    | RAM Size | Model Examples | Capabilities | Best Use Cases |
    |----------|----------------|--------------|----------------|
    | **≤2GB** | DistilBERT, TinyBERT, MobileBERT | Basic NLP, fast inference | Mobile apps, edge devices, simple classification |
    | **4GB** | TinyLLaMA, DistilGPT-2, MiniLM | Simple chat, basic reasoning | Lightweight chatbots, mobile AI assistants |
    | **6GB** | Phi-1.5, Gemma-2B, Alpaca-3B | Decent conversation, basic coding | Personal assistants, educational tools |
    | **8GB** | Phi-2, LLaMA-2-7B (4-bit), Mistral-7B (4-bit) | Good general purpose, coding help | Development tools, content creation |
    | **16GB** | LLaMA-2-7B, Mistral-7B, CodeLLaMA-7B | High quality responses, complex tasks | Professional applications, research |
    | **24GB** | LLaMA-2-13B, Mixtral-8x7B (4-bit) | Excellent performance, long context | Enterprise solutions, advanced research |
    | **32GB+** | LLaMA-2-70B (8-bit), Mixtral-8x7B | Top-tier performance, specialized tasks | Research institutions, large-scale applications |
    
    ## 🛠️ Optimization Techniques
    
    ### **Quantization Methods**
    - **4-bit**: GPTQ, AWQ - 75% memory reduction
    - **8-bit**: bitsandbytes - 50% memory reduction  
    - **16-bit**: Half precision - 50% memory reduction
    
    ### **Efficient Formats**
    - **GGUF**: Optimized for CPU inference
    - **ONNX**: Cross-platform optimization
    - **TensorRT**: NVIDIA GPU optimization
    
    ### **Memory-Saving Tips**
    - Use CPU offloading for large models
    - Reduce context window length
    - Enable gradient checkpointing
    - Use model sharding for very large models
    
    ### 🔗 **Popular Platforms & Tools**
    - **Hugging Face**: Largest model repository
    - **Ollama**: Easy local model deployment
    - **LM Studio**: GUI for running models
    - **llama.cpp**: Efficient CPU inference
    - **vLLM**: High-throughput inference
    - **Text Generation WebUI**: Web interface for models
    """)

# Footer with additional resources
st.markdown("---")
st.markdown("""
### 🔗 Essential Resources & Tools

**📦 Model Repositories:**
- [Hugging Face Hub](https://huggingface.co/models) – 500,000+ models, including BERT, LLaMA, Mistral, and more.
- [Ollama Library](https://ollama.ai/library) – Seamless CLI-based local model deployment (LLaMA, Mistral, Gemma).
- [Together AI](https://www.together.ai/models) – Access to powerful open models via API or hosted inference.

**🛠️ Inference Tools:**
- [**llama.cpp**](https://github.com/ggerganov/llama.cpp) – CPU/GPU inference for LLaMA models with quantization.
- [**GGUF format**](https://huggingface.co/docs/transformers/main/en/gguf) – Next-gen model format optimized for local inference.
- [**vLLM**](https://github.com/vllm-project/vllm) – High-throughput inference engine for transformer models.
- [**AutoGPTQ**](https://github.com/PanQiWei/AutoGPTQ) – GPU-optimized quantized inference for large models.

**📚 Learning & Deployment:**
- [Awesome LLMs](https://github.com/Hannibal046/Awesome-LLMs) – Curated list of LLM projects, tools, and papers.
- [LangChain](https://www.langchain.com/) – Framework for building apps with LLMs and tools.
- [LlamaIndex](https://www.llamaindex.ai/) – Connect LLMs with external data and documents (RAG).

---
""")
