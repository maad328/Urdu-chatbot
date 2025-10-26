# 💬 Urdu Chatbot - Streamlit Web Application

A professional conversational AI chatbot for Urdu language, powered by a fine-tuned encoder-decoder transformer model. This application provides an intuitive web interface for natural language conversations in Urdu.

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?logo=python&logoColor=white)](https://www.python.org/)

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Architecture](#-architecture)
- [Installation](#-installation)
- [Usage](#-usage)
- [Deployment](#-deployment)
- [Configuration](#-configuration)
- [Examples](#-examples)
- [Troubleshooting](#-troubleshooting)
- [Project Structure](#-project-structure)

---

## 🎯 Overview

This Urdu chatbot is built using state-of-the-art transformer architecture (encoder-decoder) and fine-tuned on conversational data in Urdu. The application provides:

- **Natural Language Understanding** - Processes Urdu text with proper tokenization
- **Context-Aware Responses** - Generates relevant responses based on user input
- **Web Interface** - Clean, modern UI built with Streamlit
- **Session Management** - Maintains conversation history
- **Easy Deployment** - One-click deployment to Streamlit Cloud

---

## ✨ Features

### Core Features

✅ **Auto-Loading Models** - Models load automatically on application startup  
✅ **Clean Chat Interface** - Modern, user-friendly chat UI with message bubbles  
✅ **Session History** - Maintains conversation context throughout the session  
✅ **Example Questions** - Quick access to sample questions in the sidebar  
✅ **Typing Indicator** - Visual feedback during response generation  
✅ **Clear Chat Option** - Reset conversation anytime  
✅ **Responsive Design** - Works seamlessly on desktop and mobile devices

### Technical Features

✅ **Encoder-Decoder Architecture** - Advanced transformer-based model  
✅ **SentencePiece Tokenization** - Efficient text processing for Urdu  
✅ **Automatic GPU/CPU Detection** - Optimizes performance based on available hardware  
✅ **Caching** - Models cached in memory for fast response times  
✅ **Error Handling** - Comprehensive error handling and user feedback

---

## 🏗️ Architecture

### Model Architecture

The chatbot uses an **Encoder-Decoder Transformer** model with the following specifications:

```
Model Configuration:
├── Embedding Dimension: 512
├── Attention Heads: 4
├── Feed-Forward Dimension: 2048
├── Encoder Layers: 3
├── Decoder Layers: 3
├── Dropout: 0.4
└── Max Sequence Length: 128
```

### Application Stack

```
Frontend Layer (Streamlit)
    ↓
Inference Layer (inference.py)
    ↓
Model Layer (PyTorch)
    ↓
Tokenizer Layer (SentencePiece)
```

### Workflow

1. **User Input** → Streamlit captures user message
2. **Tokenization** → SentencePiece tokenizes Urdu text
3. **Encoding** → Input encoded as token IDs
4. **Model Inference** → Encoder-decoder processes sequence
5. **Decoding** → Output decoded back to Urdu text
6. **Display** → Response shown in chat interface

---

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Git (for version control)

### Step 1: Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/urdu-chatbot.git
cd urdu-chatbot
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

**Required packages:**

- `streamlit>=1.28.0` - Web framework
- `torch>=2.0.0` - PyTorch for model
- `sentencepiece>=0.1.99` - Urdu tokenization
- `numpy>=1.24.0` - Numerical operations

### Step 3: Verify Installation

```bash
streamlit --version
python -c "import torch; print(torch.__version__)"
```

---

## 🚀 Usage

### Running Locally

1. **Start the Application**

```bash
streamlit run app.py
```

2. **Access the App**

Open your browser and navigate to:

```
http://localhost:8501
```

3. **Start Chatting**

Type your question in Urdu in the chat input box and press Enter.

### Example Interaction

```
You: آپ کیسے ہیں؟
Bot: میں ٹھیک ہوں، شکریہ! آپ کا دن کیسا گزر رہا ہے؟

You: آپ کا نام کیا ہے؟
Bot: میرا نام Urdu Chatbot ہے۔ میں آپ کی مدد کرنے کے لیے یہاں ہوں!
```

---

## ☁️ Deployment

### Streamlit Cloud Deployment

#### Option 1: Deploy Existing GitHub Repository

1. **Push to GitHub**

   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
   git push -u origin main
   ```

2. **Deploy on Streamlit Cloud**
   - Visit https://share.streamlit.io
   - Sign in with GitHub
   - Click **"New app"**
   - Select your repository
   - Set **Main file path**: `app.py`
   - Click **Deploy**

#### Option 2: Deploy from Streamlit Desktop

1. Open Streamlit Desktop app
2. Click **"Deploy"**
3. Follow the prompts to deploy

### Deployment Considerations

**File Size Limits:**

- Streamlit Cloud: 1GB per app
- GitHub: 100MB per file (use Git LFS for larger files)

**Model Files:**

- Ensure `backend/model/` and `backend/tokenizer/` directories are included
- Large model files may require Git LFS

### First Load Performance

- First load: 30-60 seconds (model initialization)
- Subsequent requests: <1 second (cached in memory)

---

## ⚙️ Configuration

### Model Configuration

Edit `backend/config.py` to customize:

```python
# Model Architecture
EMBED_DIM = 512          # Embedding dimension
NUM_HEADS = 4             # Attention heads
FF_DIM = 2048            # Feed-forward dimension
ENC_LAYERS = 3           # Encoder layers
DEC_LAYERS = 3           # Decoder layers
DROP_MAX_SEQ_LEN = 128   # Maximum sequence length

# File Paths
TOKENIZER_PATH = "backend/tokenizer/urdu_tokenizer.model"
MODEL_PATH = "backend/model/best_finetuned_epoch30_loss0.3141 (1).pt"

# Generation Settings
DEFAULT_MAX_LENGTH = 100
MAX_GENERATION_LENGTH = 200
```

### Streamlit Configuration

Edit `.streamlit/config.toml`:

```toml
[theme]
primaryColor = "#007bff"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f0f0"
textColor = "#000000"
font = "sans serif"

[server]
headless = true
port = 8501
```

---

## 💡 Examples

### Example Questions (Urdu)

Here are some example questions you can try:

| Urdu Question                | Translation               |
| ---------------------------- | ------------------------- |
| آپ کیسے ہیں؟                 | How are you?              |
| آپ کا نام کیا ہے؟            | What is your name?        |
| آج موسم کیسا ہے؟             | How is the weather today? |
| کیا آپ مجھے مدد کر سکتے ہیں؟ | Can you help me?          |
| آپ کہاں سے آئے ہیں؟          | Where are you from?       |
| آپ کیا کر رہے ہیں؟           | What are you doing?       |
| کیا آپ اردو بول سکتے ہیں؟    | Can you speak Urdu?       |
| آپ کا دن کیسا گزرا؟          | How was your day?         |

### Sample Conversation

```
You: آپ کو اردو میں بات کرنا کیسا لگتا ہے؟
Bot: مجھے اردو زبان پسند ہے! یہ ایک خوبصورت اور شائستہ زبان ہے۔ میں اس میں بات کرنا پسند کرتا ہوں۔

You: کیا آپ مجھے کوئی مدد کر سکتے ہیں؟
Bot: بالکل! میں یہاں آپ کی مدد کرنے کے لیے موجود ہوں۔ آپ کیا چاہتے ہیں؟
```

---

## 🐛 Troubleshooting

### Common Issues

#### Issue: "Module not found"

**Solution:**

```bash
pip install -r requirements.txt
```

#### Issue: "Tokenizer not found"

**Solution:**

- Verify `urdu_tokenizer.model` exists in `backend/tokenizer/`
- Check path in `backend/config.py`

#### Issue: "Model not found"

**Solution:**

- Verify model file exists in `backend/model/`
- Check filename matches exactly in `backend/config.py`

#### Issue: "Port already in use"

**Solution:**

```bash
streamlit run app.py --server.port 8502
```

#### Issue: "Out of memory"

**Solution:**

- Reduce `MAX_SEQ_LEN` in config
- Close other applications
- Use cloud GPU instance

#### Issue: Model loads slowly

**Solution:**

- Normal for first load (30-60 seconds)
- Models are cached after first load
- Consider model optimization/quantization

### Clear Streamlit Cache

```bash
streamlit cache clear
```

---

## 📁 Project Structure

```
urdu-chatbot/
│
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── .gitignore                      # Git ignore rules
│
├── .streamlit/
│   └── config.toml                 # Streamlit configuration
│
└── backend/
    ├── config.py                  # Model & path configuration
    ├── inference.py               # Model inference logic
    │
    ├── tokenizer/
    │   └── urdu_tokenizer.model   # SentencePiece tokenizer
    │
    └── model/
        └── best_finetuned_epoch30_loss0.3141 (1).pt  # Trained model
```

### Key Files Explained

| File                     | Purpose                                                 |
| ------------------------ | ------------------------------------------------------- |
| `app.py`                 | Streamlit web application entry point                   |
| `backend/inference.py`   | Model loading, tokenization, and text generation        |
| `backend/config.py`      | Configuration for paths, model parameters, and settings |
| `.streamlit/config.toml` | Streamlit UI theme and server settings                  |
| `requirements.txt`       | Python package dependencies                             |

---

## 🎓 Technical Details

### Model Training

The model was fine-tuned using:

- **Architecture**: Encoder-Decoder Transformer
- **Task**: Conversational response generation
- **Language**: Urdu
- **Training**: Fine-tuned for 30 epochs
- **Loss**: 0.3141 (final model)

### Tokenization

- **Method**: SentencePiece
- **Vocabulary Size**: Model-specific (from tokenizer)
- **Special Tokens**: `<pad>`, `<s>`, `</s>` (BOS/EOS)

### Generation Process

1. Input question → Encode with BOS token
2. Pass through encoder layers
3. Generate tokens sequentially with decoder
4. Stop at EOS token
5. Decode tokens → Urdu response

---

## 📄 License

This project is provided as-is for educational and research purposes.

---

## 👥 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## 📞 Support

For issues, questions, or suggestions:

- Open an issue on GitHub
- Check the troubleshooting section above
- Review project documentation

---

## 🙏 Acknowledgments

- Built with [Streamlit](https://streamlit.io/)
- Powered by [PyTorch](https://pytorch.org/)
- Tokenization by [SentencePiece](https://github.com/google/sentencepiece)

---

**Made with ❤️ for the Urdu language community**
