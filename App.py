#!/usr/bin/env python3
"""
Enhanced LLM Compatibility Advisor - Complete with Quantization & Advanced Features
Author: Assistant
Description: Comprehensive device-based LLM recommendations with quantization, comparison, and download assistance
Requirements: streamlit, pandas, plotly, openpyxl
"""

import streamlit as st
import pandas as pd
import numpy as np
import re
import plotly.express as px
import plotly.graph_objects as go
from typing import Optional, Tuple, List, Dict
import json

# ✅ MUST be the first Streamlit command
st.set_page_config(
    page_title="Enhanced LLM Compatibility Advisor", 
    layout="wide",
    page_icon="🧠",
    initial_sidebar_state="expanded"
)

# Enhanced data loading with error handling
@st.cache_data
def load_data():
    paths = [
        "src/BITS_INTERNS.xlsx",
        "src/Summer of AI - ICFAI  (Responses) (3).xlsx"
    ]

    combined_df = pd.DataFrame()
    for path in paths:
        try:
            df = pd.read_excel(path, sheet_name="Form Responses 1")
            df.columns = df.columns.str.strip()
            combined_df = pd.concat([combined_df, df], ignore_index=True)
        except FileNotFoundError:
            return None, f"Excel file '{path}' not found. Please upload the file."
        except Exception as e:
            return None, f"Error loading '{path}': {str(e)}"
    
    if combined_df.empty:
        return None, "No data found in Excel files."
    else:
        return combined_df, None

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

# Quantization options and size calculations
QUANTIZATION_FORMATS = {
    "FP16": {
        "multiplier": 1.0, 
        "description": "Full precision, best quality", 
        "icon": "🔥",
        "quality": "Excellent",
        "speed": "Moderate",
        "memory_efficiency": "Low"
    },
    "8-bit": {
        "multiplier": 0.5, 
        "description": "50% smaller, good quality", 
        "icon": "⚡",
        "quality": "Very Good",
        "speed": "Good",
        "memory_efficiency": "Good"
    },
    "4-bit": {
        "multiplier": 0.25, 
        "description": "75% smaller, acceptable quality", 
        "icon": "💎",
        "quality": "Good",
        "speed": "Very Good",
        "memory_efficiency": "Excellent"
    },
    "2-bit": {
        "multiplier": 0.125, 
        "description": "87.5% smaller, experimental", 
        "icon": "🧪",
        "quality": "Fair",
        "speed": "Excellent",
        "memory_efficiency": "Outstanding"
    }
}

def calculate_quantized_size(base_size_str, quant_format):
    """Calculate quantized model size with better formatting"""
    size_match = re.search(r'(\d+\.?\d*)', base_size_str)
    if not size_match:
        return base_size_str
    
    base_size = float(size_match.group(1))
    unit = base_size_str.replace(size_match.group(1), "").strip()
    
    multiplier = QUANTIZATION_FORMATS[quant_format]["multiplier"]
    new_size = base_size * multiplier
    
    # Smart unit conversion
    if unit.upper() == "GB" and new_size < 1:
        return f"{new_size * 1024:.0f}MB"
    elif unit.upper() == "MB" and new_size > 1024:
        return f"{new_size / 1024:.1f}GB"
    else:
        return f"{new_size:.1f}{unit}"

# Enhanced LLM database with more models and metadata
LLM_DATABASE = {
    "ultra_low": {  # ≤2GB
        "general": [
            {"name": "TinyLlama-1.1B-Chat", "size": "637MB", "description": "Compact chat model", "parameters": "1.1B", "context": "2K"},
            {"name": "DistilBERT-base", "size": "268MB", "description": "Efficient BERT variant", "parameters": "66M", "context": "512"},
            {"name": "all-MiniLM-L6-v2", "size": "91MB", "description": "Sentence embeddings", "parameters": "22M", "context": "256"},
            {"name": "OpenELM-270M", "size": "540MB", "description": "Apple's efficient model", "parameters": "270M", "context": "2K"}
        ],
        "code": [
            {"name": "CodeT5-small", "size": "242MB", "description": "Code generation", "parameters": "60M", "context": "512"},
            {"name": "Replit-code-v1-3B", "size": "1.2GB", "description": "Code completion", "parameters": "3B", "context": "4K"}
        ]
    },
    "low": {  # 3-4GB
        "general": [
            {"name": "Phi-1.5", "size": "2.8GB", "description": "Microsoft's efficient model", "parameters": "1.3B", "context": "2K"},
            {"name": "Gemma-2B", "size": "1.4GB", "description": "Google's compact model", "parameters": "2B", "context": "8K"},
            {"name": "OpenLLaMA-3B", "size": "2.1GB", "description": "Open source LLaMA", "parameters": "3B", "context": "2K"},
            {"name": "StableLM-3B", "size": "2.2GB", "description": "Stability AI model", "parameters": "3B", "context": "4K"}
        ],
        "code": [
            {"name": "CodeGen-2B", "size": "1.8GB", "description": "Salesforce code model", "parameters": "2B", "context": "2K"},
            {"name": "StarCoder-1B", "size": "1.1GB", "description": "BigCode project", "parameters": "1B", "context": "8K"}
        ],
        "chat": [
            {"name": "Alpaca-3B", "size": "2.0GB", "description": "Stanford's instruction model", "parameters": "3B", "context": "2K"},
            {"name": "Vicuna-3B", "size": "2.1GB", "description": "ChatGPT-style training", "parameters": "3B", "context": "2K"}
        ]
    },
    "moderate_low": {  # 5-6GB
        "general": [
            {"name": "Phi-2", "size": "5.2GB", "description": "Microsoft's 2.7B model", "parameters": "2.7B", "context": "2K"},
            {"name": "Gemma-7B-it", "size": "4.2GB", "description": "Google instruction tuned", "parameters": "7B", "context": "8K"},
            {"name": "Mistral-7B-v0.1", "size": "4.1GB", "description": "Mistral AI base model", "parameters": "7B", "context": "8K"},
            {"name": "Llama-2-7B", "size": "4.0GB", "description": "Meta's foundation model", "parameters": "7B", "context": "4K"}
        ],
        "code": [
            {"name": "CodeLlama-7B", "size": "3.8GB", "description": "Meta's code specialist", "parameters": "7B", "context": "16K"},
            {"name": "StarCoder-7B", "size": "4.0GB", "description": "Code generation expert", "parameters": "7B", "context": "8K"}
        ],
        "chat": [
            {"name": "Zephyr-7B-beta", "size": "4.2GB", "description": "HuggingFace chat model", "parameters": "7B", "context": "32K"},
            {"name": "Neural-Chat-7B", "size": "4.1GB", "description": "Intel optimized", "parameters": "7B", "context": "32K"}
        ]
    },
    "moderate": {  # 7-8GB
        "general": [
            {"name": "Llama-2-7B-Chat", "size": "3.5GB", "description": "Meta's popular chat model", "parameters": "7B", "context": "4K"},
            {"name": "Mistral-7B-Instruct-v0.2", "size": "4.1GB", "description": "Latest Mistral instruct", "parameters": "7B", "context": "32K"},
            {"name": "Qwen-7B-Chat", "size": "4.0GB", "description": "Alibaba's multilingual", "parameters": "7B", "context": "32K"},
            {"name": "Solar-10.7B-Instruct", "size": "5.8GB", "description": "Upstage's efficient model", "parameters": "10.7B", "context": "4K"}
        ],
        "code": [
            {"name": "CodeLlama-7B-Instruct", "size": "3.8GB", "description": "Instruction-tuned CodeLlama", "parameters": "7B", "context": "16K"},
            {"name": "WizardCoder-7B", "size": "4.0GB", "description": "Enhanced coding abilities", "parameters": "7B", "context": "16K"},
            {"name": "Phind-CodeLlama-34B-v2", "size": "4.2GB", "description": "4-bit quantized version", "parameters": "34B", "context": "16K"}
        ],
        "reasoning": [
            {"name": "WizardMath-7B", "size": "4.0GB", "description": "Mathematical reasoning", "parameters": "7B", "context": "2K"},
            {"name": "MetaMath-7B", "size": "3.9GB", "description": "Math problem solving", "parameters": "7B", "context": "2K"}
        ]
    },
    "good": {  # 9-16GB
        "general": [
            {"name": "Llama-2-13B-Chat", "size": "7.3GB", "description": "Larger Llama variant", "parameters": "13B", "context": "4K"},
            {"name": "Vicuna-13B-v1.5", "size": "7.2GB", "description": "Enhanced Vicuna", "parameters": "13B", "context": "16K"},
            {"name": "OpenChat-3.5", "size": "7.1GB", "description": "High-quality chat model", "parameters": "7B", "context": "8K"},
            {"name": "Nous-Hermes-2-Mixtral-8x7B-DPO", "size": "12.9GB", "description": "4-bit quantized MoE", "parameters": "47B", "context": "32K"}
        ],
        "code": [
            {"name": "CodeLlama-13B-Instruct", "size": "7.3GB", "description": "Larger code model", "parameters": "13B", "context": "16K"},
            {"name": "WizardCoder-15B", "size": "8.2GB", "description": "Advanced coding", "parameters": "15B", "context": "16K"},
            {"name": "StarCoder-15B", "size": "8.5GB", "description": "Large code model", "parameters": "15B", "context": "8K"}
        ],
        "multimodal": [
            {"name": "LLaVA-7B", "size": "7.0GB", "description": "Vision + language", "parameters": "7B", "context": "2K"},
            {"name": "MiniGPT-4-7B", "size": "6.8GB", "description": "Multimodal chat", "parameters": "7B", "context": "2K"},
            {"name": "Instructblip-7B", "size": "7.2GB", "description": "Instruction-tuned VLM", "parameters": "7B", "context": "2K"}
        ],
        "reasoning": [
            {"name": "WizardMath-13B", "size": "7.3GB", "description": "Advanced math", "parameters": "13B", "context": "2K"},
            {"name": "Orca-2-13B", "size": "7.4GB", "description": "Microsoft reasoning", "parameters": "13B", "context": "4K"}
        ]
    },
    "high": {  # 17-32GB
        "general": [
            {"name": "Mixtral-8x7B-Instruct-v0.1", "size": "26.9GB", "description": "Mixture of experts", "parameters": "47B", "context": "32K"},
            {"name": "Llama-2-70B-Chat", "size": "38.0GB", "description": "8-bit quantized", "parameters": "70B", "context": "4K"},
            {"name": "Yi-34B-Chat", "size": "19.5GB", "description": "01.AI's large model", "parameters": "34B", "context": "200K"},
            {"name": "Nous-Hermes-2-Yi-34B", "size": "19.2GB", "description": "Enhanced Yi variant", "parameters": "34B", "context": "200K"}
        ],
        "code": [
            {"name": "CodeLlama-34B-Instruct", "size": "19.0GB", "description": "Large code specialist", "parameters": "34B", "context": "16K"},
            {"name": "DeepSeek-Coder-33B", "size": "18.5GB", "description": "DeepSeek's coder", "parameters": "33B", "context": "16K"},
            {"name": "WizardCoder-34B", "size": "19.2GB", "description": "Enterprise coding", "parameters": "34B", "context": "16K"}
        ],
        "reasoning": [
            {"name": "WizardMath-70B", "size": "38.5GB", "description": "8-bit quantized math", "parameters": "70B", "context": "2K"},
            {"name": "MetaMath-70B", "size": "38.0GB", "description": "8-bit math reasoning", "parameters": "70B", "context": "2K"}
        ]
    },
    "ultra_high": {  # >32GB
        "general": [
            {"name": "Llama-2-70B", "size": "130GB", "description": "Full precision", "parameters": "70B", "context": "4K"},
            {"name": "Mixtral-8x22B", "size": "176GB", "description": "Latest mixture model", "parameters": "141B", "context": "64K"},
            {"name": "Qwen-72B", "size": "145GB", "description": "Alibaba's flagship", "parameters": "72B", "context": "32K"},
            {"name": "Llama-3-70B", "size": "140GB", "description": "Meta's latest", "parameters": "70B", "context": "8K"}
        ],
        "code": [
            {"name": "CodeLlama-34B", "size": "68GB", "description": "Full precision code", "parameters": "34B", "context": "16K"},
            {"name": "DeepSeek-Coder-33B", "size": "66GB", "description": "Full precision coding", "parameters": "33B", "context": "16K"}
        ],
        "reasoning": [
            {"name": "WizardMath-70B", "size": "130GB", "description": "Full precision math", "parameters": "70B", "context": "2K"},
            {"name": "Goat-70B", "size": "132GB", "description": "Arithmetic reasoning", "parameters": "70B", "context": "2K"}
        ]
    }
}

# GPU compatibility database
# Enhanced GPU compatibility database with more details
GPU_DATABASE = {
    "RTX 3060": {"vram": 8, "performance": "mid", "architecture": "Ampere", "tensor_cores": "2nd gen", "memory_bandwidth": "360 GB/s"},
    "RTX 3070": {"vram": 8, "performance": "high", "architecture": "Ampere", "tensor_cores": "2nd gen", "memory_bandwidth": "448 GB/s"},
    "RTX 3080": {"vram": 10, "performance": "high", "architecture": "Ampere", "tensor_cores": "2nd gen", "memory_bandwidth": "760 GB/s"},
    "RTX 3090": {"vram": 24, "performance": "ultra", "architecture": "Ampere", "tensor_cores": "2nd gen", "memory_bandwidth": "936 GB/s"},
    "RTX 4060": {"vram": 8, "performance": "mid", "architecture": "Ada Lovelace", "tensor_cores": "4th gen", "memory_bandwidth": "272 GB/s"},
    "RTX 4070": {"vram": 12, "performance": "high", "architecture": "Ada Lovelace", "tensor_cores": "4th gen", "memory_bandwidth": "504 GB/s"},
    "RTX 4080": {"vram": 16, "performance": "ultra", "architecture": "Ada Lovelace", "tensor_cores": "4th gen", "memory_bandwidth": "716 GB/s"},
    "RTX 4090": {"vram": 24, "performance": "ultra", "architecture": "Ada Lovelace", "tensor_cores": "4th gen", "memory_bandwidth": "1008 GB/s"},
    "Apple M1": {"vram": 8, "performance": "mid", "architecture": "Apple Silicon", "tensor_cores": "None", "memory_bandwidth": "68.25 GB/s"},
    "Apple M2": {"vram": 16, "performance": "high", "architecture": "Apple Silicon", "tensor_cores": "None", "memory_bandwidth": "100 GB/s"},
    "Apple M3": {"vram": 24, "performance": "ultra", "architecture": "Apple Silicon", "tensor_cores": "None", "memory_bandwidth": "150 GB/s"},
    "RX 6700 XT": {"vram": 12, "performance": "mid", "architecture": "RDNA 2", "tensor_cores": "None", "memory_bandwidth": "384 GB/s"},
    "RX 7900 XTX": {"vram": 24, "performance": "ultra", "architecture": "RDNA 3", "tensor_cores": "None", "memory_bandwidth": "960 GB/s"},
}

def get_gpu_recommendations(gpu_name, ram_gb):
    """Get GPU-specific model recommendations"""
    if gpu_name == "No GPU":
        return "CPU-only models recommended", "Use 4-bit quantization for better performance"
    
    gpu_info = GPU_DATABASE.get(gpu_name.split(" (")[0], {"vram": 0, "performance": "low"})
    vram = gpu_info["vram"]
    
    if vram <= 8:
        return f"7B models with 4-bit quantization", f"Estimated VRAM usage: ~{vram-1}GB"
    elif vram <= 12:
        return f"13B models with 8-bit quantization", f"Estimated VRAM usage: ~{vram-1}GB"
    elif vram <= 16:
        return f"13B models at FP16 or 30B with 4-bit", f"Estimated VRAM usage: ~{vram-1}GB"
    else:
        return f"70B models with 4-bit quantization", f"Estimated VRAM usage: ~{vram-2}GB"

def predict_inference_speed(model_size_gb, ram_gb, has_gpu=False, gpu_name=""):
    """Predict approximate inference speed"""
    if model_size_gb > ram_gb:
        return "❌ Insufficient RAM", "Consider smaller model or quantization"
    
    if has_gpu and gpu_name != "No GPU":
        gpu_info = GPU_DATABASE.get(gpu_name.split(" (")[0], {"performance": "low"})
        perf = gpu_info["performance"]
        
        if perf == "ultra":
            if model_size_gb <= 4:
                return "⚡ Blazing Fast", "~50-100 tokens/sec"
            elif model_size_gb <= 8:
                return "🚀 Very Fast", "~30-60 tokens/sec"
            elif model_size_gb <= 16:
                return "🏃 Fast", "~15-30 tokens/sec"
            else:
                return "🐌 Moderate", "~5-15 tokens/sec"
        elif perf == "high":
            if model_size_gb <= 4:
                return "⚡ Very Fast", "~30-50 tokens/sec"
            elif model_size_gb <= 8:
                return "🚀 Fast", "~15-30 tokens/sec"
            else:
                return "🐌 Moderate", "~5-15 tokens/sec"
        else:  # mid performance
            if model_size_gb <= 4:
                return "⚡ Fast", "~15-30 tokens/sec"
            else:
                return "🐌 Slow", "~3-10 tokens/sec"
    else:
        # CPU inference
        if model_size_gb <= 2:
            return "⚡ Acceptable", "~5-15 tokens/sec"
        elif model_size_gb <= 4:
            return "🐌 Slow", "~1-5 tokens/sec"
        else:
            return "🐌 Very Slow", "~0.5-2 tokens/sec"

# Enhanced LLM recommendation with performance tiers
def recommend_llm(ram_str) -> Tuple[str, str, str, Dict[str, List[Dict]]]:
    """Returns (recommendation, performance_tier, additional_info, detailed_models)"""
    ram = extract_numeric_ram(ram_str)
    
    if ram is None:
        return ("⚪ Check exact specs or test with quantized models.", 
                "Unknown", 
                "Verify RAM specifications",
                {})
    
    if ram <= 2:
        models = LLM_DATABASE["ultra_low"]
        return ("🔸 Ultra-lightweight models - basic NLP tasks", 
                "Ultra Low", 
                "Mobile-optimized, simple tasks, limited context",
                models)
    elif ram <= 4:
        models = LLM_DATABASE["low"]
        return ("🔸 Small language models - decent capabilities", 
                "Low", 
                "Basic chat, simple reasoning, text classification",
                models)
    elif ram <= 6:
        models = LLM_DATABASE["moderate_low"]
        return ("🟠 Mid-range models - good general performance", 
                "Moderate-Low", 
                "Solid reasoning, coding help, longer conversations",
                models)
    elif ram <= 8:
        models = LLM_DATABASE["moderate"]
        return ("🟠 Strong 7B models - excellent capabilities", 
                "Moderate", 
                "Professional use, coding assistance, complex reasoning",
                models)
    elif ram <= 16:
        models = LLM_DATABASE["good"]
        return ("🟢 High-quality models - premium performance", 
                "Good", 
                "Advanced tasks, multimodal support, research use",
                models)
    elif ram <= 32:
        models = LLM_DATABASE["high"]
        return ("🔵 Premium models - professional grade", 
                "High", 
                "Enterprise ready, complex reasoning, specialized tasks",
                models)
    else:
        models = LLM_DATABASE["ultra_high"]
        return ("🔵 Top-tier models - enterprise capabilities", 
                "Ultra High", 
                "Research grade, maximum performance, domain expertise",
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

# Model comparison function
def create_model_comparison_table(selected_models, quantization_type="FP16"):
    """Create a comparison table for selected models"""
    comparison_data = []
    
    for model_info in selected_models:
        quant_size = calculate_quantized_size(model_info['size'], quantization_type)
        
        # Extract numeric size for VRAM calculation
        size_match = re.search(r'(\d+\.?\d*)', quant_size)
        if size_match:
            size_num = float(size_match.group(1))
            estimated_vram = f"{size_num * 1.2:.1f}GB"
        else:
            estimated_vram = "Unknown"
        
        comparison_data.append({
            'Model': model_info['name'],
            'Parameters': model_info.get('parameters', 'Unknown'),
            'Context': model_info.get('context', 'Unknown'),
            'Original Size': model_info['size'],
            f'{quantization_type} Size': quant_size,
            'Est. VRAM': estimated_vram,
            'Description': model_info['description']
        })
    
    return pd.DataFrame(comparison_data)

# Enhanced model details display function
def display_model_categories(models_dict: Dict[str, List[Dict]], ram_gb: int, show_quantization=True):
    """Display models with quantization options"""
    if not models_dict:
        return
    
    st.markdown(f"### 🎯 Recommended Models for {ram_gb}GB RAM:")
    
    for category, model_list in models_dict.items():
        if model_list:
            with st.expander(f"📂 {category.replace('_', ' ').title()} Models"):
                for model in model_list[:6]:  # Show top 6 models per category
                    st.markdown(f"**{model['name']}**")
                    
                    # Model details
                    detail_col1, detail_col2, detail_col3 = st.columns(3)
                    with detail_col1:
                        st.caption(f"📊 {model.get('parameters', 'Unknown')} params")
                    with detail_col2:
                        st.caption(f"🔍 {model.get('context', 'Unknown')} context")
                    with detail_col3:
                        st.caption(f"💾 {model['size']} original")
                    
                    st.markdown(f"*{model['description']}*")
                    
                    if show_quantization:
                        # Create quantization size table
                        quant_cols = st.columns(4)
                        for i, (quant_type, quant_info) in enumerate(QUANTIZATION_FORMATS.items()):
                            with quant_cols[i]:
                                quant_size = calculate_quantized_size(model['size'], quant_type)
                                st.metric(
                                    label=f"{quant_info['icon']} {quant_type}",
                                    value=quant_size,
                                    help=quant_info['description']
                                )
                    
                    st.markdown("---")

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
        nbinsx=10,
        marker_color='#1f77b4'
    ))
    
    fig.add_trace(go.Histogram(
        x=mobile_rams,
        name="Mobile RAM",
        opacity=0.7,
        nbinsx=10,
        marker_color='#ff7f0e'
    ))
    
    fig.update_layout(
        title="RAM Distribution Across Devices",
        xaxis_title="RAM (GB)",
        yaxis_title="Number of Students",
        barmode='overlay',
        height=400,
        showlegend=True
    )
    
    return fig

# Demo data generator for when Excel files are not available
def generate_demo_data():
    """Generate demo data for testing when Excel files are missing"""
    demo_data = {
        "Full Name": [
            "Demo Student 1", "Demo Student 2", "Demo Student 3", "Demo Student 4",
            "Demo Student 5", "Demo Student 6", "Demo Student 7", "Demo Student 8",
            "Demo Student 9", "Demo Student 10", "Demo Student 11", "Demo Student 12"
        ],
        "Laptop RAM": ["8GB", "16GB", "4GB", "32GB", "6GB", "12GB", "2GB", "24GB", "64GB", "3GB", "20GB", "10GB"],
        "Mobile RAM": ["4GB", "8GB", "3GB", "12GB", "6GB", "4GB", "2GB", "8GB", "16GB", "3GB", "6GB", "8GB"],
        "Laptop Operating System": [
            "Windows 11", "macOS Monterey", "Ubuntu 22.04", "Windows 10",
            "macOS Big Sur", "Fedora 36", "Windows 11", "macOS Ventura",
            "Ubuntu 20.04", "Windows 10", "macOS Sonoma", "Pop!_OS 22.04"
        ],
        "Mobile Operating System": [
            "Android 13", "iOS 16", "Android 12", "iOS 15",
            "Android 14", "iOS 17", "Android 11", "iOS 16",
            "Android 13", "iOS 15", "Android 14", "iOS 17"
        ]
    }
    return pd.DataFrame(demo_data)

# Function to safely prepare user options
def prepare_user_options(df):
    """Safely prepare user options for selectbox, handling NaN values and mixed types"""
    try:
        unique_names = df["Full Name"].dropna().unique()
        
        valid_names = []
        for name in unique_names:
            try:
                str_name = str(name).strip()
                if str_name and str_name.lower() != 'nan':
                    valid_names.append(str_name)
            except:
                continue
        
        options = ["Select a student..."] + sorted(valid_names)
        return options
    except Exception as e:
        st.error(f"Error preparing user options: {e}")
        return ["Select a student..."]

# Main App
st.title("🧠 LLM Compatibility Advisor")
st.markdown("Get personalized recommendations from **150+ popular open source AI models** with download sizes!")

# Load data with better error handling
df, error = load_data()

if error or df is None or df.empty:
    st.warning("⚠️ Excel files not found. Running with demo data for testing.")
    st.info("📁 To use real data, place 'BITS_INTERNS.xlsx' and 'Summer of AI - ICFAI  (Responses) (3).xlsx' in the 'src/' directory.")
    df = generate_demo_data()
    
    with st.expander("📋 Expected Data Format"):
        st.markdown("""
        The app expects Excel files with the following columns:
        - **Full Name**: Student name
        - **Laptop RAM**: RAM specification (e.g., "8GB", "16 GB", "8192MB")
        - **Mobile RAM**: Mobile device RAM
        - **Laptop Operating System**: OS name
        - **Mobile Operating System**: Mobile OS name
        """)

# Verify required columns exist
required_columns = ["Full Name", "Laptop RAM", "Mobile RAM"]
missing_columns = [col for col in required_columns if col not in df.columns]

if missing_columns:
    st.error(f"Missing required columns: {missing_columns}")
    st.info("Please ensure your Excel file contains the required columns.")
    st.stop()

# Clean the dataframe
df = df.copy()
df["Full Name"] = df["Full Name"].astype(str).str.strip()

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
        ["general", "code", "chat", "reasoning", "multimodal"],
        default=["general", "code", "chat"]
    )
    
    st.markdown("---")
    st.markdown("### 📊 Quick Stats")
    st.metric("Total Students", len(df))
    st.metric("Popular Models", "150+")
    
    # Calculate average RAM
    avg_laptop_ram = df["Laptop RAM"].apply(extract_numeric_ram).mean()
    avg_mobile_ram = df["Mobile RAM"].apply(extract_numeric_ram).mean()
    
    if not pd.isna(avg_laptop_ram):
        st.metric("Avg Laptop RAM", f"{avg_laptop_ram:.1f} GB")
    if not pd.isna(avg_mobile_ram):
        st.metric("Avg Mobile RAM", f"{avg_mobile_ram:.1f} GB")

# User selection with search - FIXED VERSION
st.subheader("👤 Individual Student Analysis")

# Prepare options safely
user_options = prepare_user_options(df)

selected_user = st.selectbox(
    "Choose a student:",
    options=user_options,
    index=0  # Default to first option ("Select a student...")
)

if selected_user and selected_user != "Select a student...":
    # Find user data with safe lookup
    user_data_mask = df["Full Name"].astype(str).str.strip() == selected_user
    if user_data_mask.any():
        user_data = df[user_data_mask].iloc[0]
        
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
mask = (laptop_tiers.isin(performance_filter) | mobile_tiers.isin(performance_filter))
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
        percentage = (count / len(mobile_tier_counts)) * 100
        st.write(f"• {tier}: {count} students ({percentage:.1f}%)")

# Model Explorer Section
st.markdown("---")
st.header("🔍 Popular Model Explorer")

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
        ["general", "code", "chat", "reasoning", "multimodal"]
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
    
    # Display models in a detailed table
    for model in models:
        with st.container():
            col1, col2, col3 = st.columns([3, 1, 3])
            with col1:
                st.markdown(f"### {model['name']}")
            with col2:
                st.markdown(f"**{model['size']}**")
                st.caption("Download Size")
            with col3:
                st.markdown(f"*{model['description']}*")
                # Add download suggestion
                if "Llama" in model['name']:
                    st.caption("🔗 Available on Hugging Face & Ollama")
                elif "Mistral" in model['name']:
                    st.caption("🔗 Available on Hugging Face & Mistral AI")
                elif "Gemma" in model['name']:
                    st.caption("🔗 Available on Hugging Face & Google")
                else:
                    st.caption("🔗 Available on Hugging Face")
            st.markdown("---")
else:
    st.info(f"No {selected_category} models available for {selected_ram_range}")

# Enhanced reference guide
with st.expander("📘 Model Guide & Download Information"):
    st.markdown("""
    ## 🚀 Popular Models by Category
    
    ### 🎯 **General Purpose Champions**
    - **Llama-2 Series**: Meta's flagship models (7B, 13B, 70B)
    - **Mistral Series**: Excellent efficiency and performance
    - **Gemma**: Google's efficient models (2B, 7B)
    - **Phi**: Microsoft's compact powerhouses
    
    ### 💻 **Code Specialists** 
    - **CodeLlama**: Meta's dedicated coding models
    - **StarCoder**: BigCode's programming experts
    - **WizardCoder**: Enhanced coding capabilities
    - **DeepSeek-Coder**: Chinese tech giant's coder
    
    ### 💬 **Chat Optimized**
    - **Vicuna**: UC Berkeley's ChatGPT alternative
    - **Zephyr**: HuggingFace's chat specialist
    - **OpenChat**: High-quality conversation models
    - **Neural-Chat**: Intel-optimized chat models
 
    ### 🧮 **Reasoning Masters**
    - **WizardMath**: Mathematical problem solving
    - **MetaMath**: Advanced arithmetic reasoning
    - **Orca-2**: Microsoft's reasoning specialist
    - **Goat**: Specialized arithmetic model
    
    ### 👁️ **Multimodal Models**
    - **LLaVA**: Large Language and Vision Assistant
    - **MiniGPT-4**: Multimodal conversational AI
    
    ## 💾 Download Size Reference
    
    | Model Size | FP16 | 8-bit | 4-bit | Use Case |
    |------------|------|-------|-------|----------|
    | **1-3B** | 2-6GB | 1-3GB | 0.5-1.5GB | Mobile, Edge |
    | **7B** | 13GB | 7GB | 3.5GB | Desktop, Laptop |
    | **13B** | 26GB | 13GB | 7GB | Workstation |
    | **30-34B** | 60GB | 30GB | 15GB | Server, Cloud |
    | **70B** | 140GB | 70GB | 35GB | High-end Server |
    
    ## 🛠️ Where to Download
    
    ### **Primary Sources**
    - **🤗 Hugging Face**: Largest repository with 400,000+ models
    - **🦙 Ollama**: Simple CLI tool for local deployment
    - **📦 LM Studio**: User-friendly GUI for model management
    
    ### **Quantized Formats**
    - **GGUF**: Best for CPU inference (llama.cpp)
    - **GPTQ**: GPU-optimized quantization
    - **AWQ**: Advanced weight quantization
    
    ### **Download Tips**
    - Use git lfs for large models from Hugging Face
    - Consider bandwidth and storage before downloading
    - Start with 4-bit quantized versions for testing
    - Use ollama pull model_name for easiest setup
    
    ## 🔧 Optimization Strategies
    
    ### **Memory Reduction**
    - **4-bit quantization**: 75% memory reduction
    - **8-bit quantization**: 50% memory reduction
    - **CPU offloading**: Use system RAM for overflow
    
    ### **Speed Optimization**
    - **GPU acceleration**: CUDA, ROCm, Metal
    - **Batch processing**: Process multiple requests
    - **Context caching**: Reuse computations
    """)

# Footer with updated resources
st.markdown("---")
st.markdown("""
### 🔗 Essential Download & Deployment Tools
**📦 Easy Model Deployment:**
- [**Ollama**](https://ollama.ai/) – curl -fsSL https://ollama.ai/install.sh | sh
- [**LM Studio**](https://lmstudio.ai/) – Drag-and-drop GUI for running models locally
- [**GPT4All**](https://gpt4all.io/) – Cross-platform desktop app for local LLMs
**🤗 Model Repositories:**
- [**Hugging Face Hub**](https://huggingface.co/models) – Filter by model size, task, and license
- [**TheBloke's Quantizations**](https://huggingface.co/TheBloke) – Pre-quantized models in GGUF/GPTQ format
- [**Awesome LLM**](https://github.com/Hannibal046/Awesome-LLMs) – Curated list of models and resources
---
""")
