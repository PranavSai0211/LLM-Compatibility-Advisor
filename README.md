## Direct Link for the app##
# https://huggingface.co/spaces/qwerty45-uiop/LLM-Compatibilty-Advisor #

# 🧠 Enhanced LLM Compatibility Advisor

A powerful Streamlit application for recommending the best open-source large language models (LLMs) based on device specifications. Designed for students, researchers, and developers looking to run LLMs on laptops or mobile devices with limited or varying RAM/GPU capabilities.

---

## 🚀 Features

- 🔍 **Personalized LLM Recommendations** based on device RAM
- 🧠 **150+ Popular Open Source Models** across categories (general, code, chat, reasoning, multimodal)
- 📉 **Quantization Support** (FP16, 8-bit, 4-bit, 2-bit) with download size estimations
- ⚙️ **Inference Speed & VRAM Estimator**
- 📱 **Mobile & Laptop Analysis Support**
- 📊 **Batch Insights** for analyzing multiple student entries
- 📈 **RAM Distribution Visualizations**
- 🔁 **Model Comparison Tool**
- ✅ **Offline/Demo Mode** when Excel files are missing
- 💾 **GPU Recommendations** using a built-in compatibility database

---

## 📁 Project Structure

```
.
├── streamlit_app.py
├── src/
│   ├── BITS_INTERNS.xlsx
│   └── Summer of AI - ICFAI  (Responses) (3).xlsx
```

- `streamlit_app.py` – Main Streamlit app file
- `src/` – Directory containing Excel response data

---
``


## ▶️ Running the App

```bash
streamlit run streamlit_app.py
```

Make sure Excel files are placed in the `src/` folder.

---


## 📊 RAM-based Model Categories

| Tier         | RAM Range | Description                       |
|--------------|-----------|------------------------------------|
| Ultra Low    | ≤ 2GB     | Tiny models for basic tasks        |
| Low          | 3–4GB     | Entry-level LLMs                   |
| Moderate Low | 5–6GB     | Standard 7B models (quantized)     |
| Moderate     | 7–8GB     | Chat/code-focused, 7B-class        |
| Good         | 9–16GB    | Advanced 13B-class models          |
| High         | 17–32GB   | Large 30B–70B quantized models     |
| Ultra High   | >32GB     | Full-size 70B+ models (FP16)       |

## 📚 Credits

- Developed with ❤️ using [Streamlit](https://streamlit.io), [Hugging Face](https://huggingface.co), and [Plotly](https://plotly.com)
- Data format based on internship responses from BITS & ICFAI students.

---

## 📝 License

This project is open source and available under the [MIT License](LICENSE).
