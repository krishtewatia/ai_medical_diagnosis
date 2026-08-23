# 🩺 AI Medical Diagnosis Assistant

[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg?style=flat&logo=FastAPI&logoColor=white)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-19.0+-61DAFB.svg?style=flat&logo=react&logoColor=black)](https://reactjs.org/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.0+-3178C6.svg?style=flat&logo=typescript&logoColor=white)](https://www.typescriptlang.org/)
[![MongoDB](https://img.shields.io/badge/MongoDB-Atlas%20M0-47A248.svg?style=flat&logo=mongodb&logoColor=white)](https://www.mongodb.com/atlas)
[![Keras / PyTorch](https://img.shields.io/badge/Deep%20Learning-Keras%20%7C%20PyTorch-EE4C2C.svg?style=flat&logo=pytorch&logoColor=white)](https://keras.io)
[![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn%20%7C%20XGBoost-F7931E.svg?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

An enterprise-grade, full-stack **AI-Assisted Clinical Decision Support System** designed to assist healthcare professionals in diagnosing conditions from both **clinical tabular data** and **high-resolution medical imaging (Chest X-Rays and Brain MRIs)**. 

Built with **FastAPI**, **React 19 + TypeScript**, **MongoDB Atlas**, and state-of-the-art **Deep Learning & Machine Learning models**, with **zero-cost free-tier deployment architecture**.

---

## 🌟 Highlights & Key Capabilities

- 🔬 **Multi-Modal AI Inference**: Unified support for tabular biomarker data and medical imaging uploads (DICOM/PNG/JPG).
- 🧠 **Pre-Trained Deep Learning**:
  - **Pneumonia Detection**: DenseNet121 CNN for chest radiograph (CXR) lung opacity screening.
  - **Brain Tumor Classification**: Deep ResNet Convolutional Neural Network for brain MRI scans.
  - **Diabetes Risk Assessment**: XGBoost / Ensemble classifiers trained on metabolic biomarkers.
  - **Heart Disease Risk Scoring**: Random Forest / Gradient Boosted tabular assessment.
- 📋 **Dynamic Schema-Driven UI**: Frontend dynamically adapts forms, validation rules, and input constraints based on backend disease registry metadata.
- 📑 **Instant Clinical PDF Reports**: Dynamic, downloadable medical report generation powered by ReportLab (complete with risk stratifications, patient demographics, biomarker tables, and disclaimers).
- 🔒 **HIPAA/Security-First Architecture**:
  - Encrypted password hashing with `bcrypt`.
  - Stateless JWT (HS256) authentication with strict token expiry.
  - Granular patient data isolation with MongoDB Atlas.
  - Secure local buffer & stream inference (no sensitive medical images persisted unnecessarily).
- 🚀 **Free-Tier Ready**: Designed from the ground up to run 100% free with no credit card required (Vercel/Netlify frontend + Free FastAPI backend + MongoDB Atlas M0).

---

## 🏗️ Architecture

```
                    ┌─────────────────────────┐
                    │      React 19 + TS      │
                    │   (Vercel / Netlify)    │
                    └────────────┬────────────┘
                                 │ REST API / JWT
                                 ▼
                    ┌─────────────────────────┐
                    │     FastAPI Backend     │
                    │ (Render / Free Hosting) │
                    └────────────┬────────────┘
                                 │
         ┌───────────────────────┼───────────────────────┐
         ▼                       ▼                       ▼
┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│  MongoDB Atlas   │    │  Inference Core  │    │  PDF & Storage   │
│  - User Auth     │    │  - DenseNet121   │    │  - ReportLab PDF │
│  - Patient Data  │    │  - ResNet MRI    │    │  - Temp In-Mem / │
│  - Audit History │    │  - XGBoost / SK  │    │    Local Buffer  │
└──────────────────┘    └──────────────────┘    └──────────────────┘
```

---

## 🗂️ Supported Disease Modules

| Disease | Modality | Model Architecture | Metrics / Output |
| :--- | :--- | :--- | :--- |
| **Pneumonia** | Chest X-Ray (CXR) | DenseNet121 Deep CNN | Binary classification (Opacity vs. Normal) + Confidence Score |
| **Brain Tumor** | Brain MRI (T1/T2) | ResNet50 Deep CNN | Multi-class Classification + Probability Breakdown |
| **Diabetes** | Tabular Biomarkers | XGBoost / Scikit-Learn | Diabetes Risk Probability + Feature Importance |
| **Heart Disease** | Tabular Clinical Data | Random Forest Classifier | Cardiovascular Disease Risk Stratification |

---

## 📁 Repository Structure

```
AI-Medical-Diagnosis-Assistant/
├── backend/
│   ├── app/
│   │   ├── api/              # FastAPI endpoints (Auth, Diseases, Predictions, History, Reports, Profiles)
│   │   ├── core/             # Configuration, Security, JWT utilities
│   │   ├── database/         # MongoDB Atlas connection manager & lifecycles
│   │   ├── ml/               # Disease registry, loaders, tabular & image predictors
│   │   │   ├── image_models/ # DenseNet121 & ResNet deep learning artifacts
│   │   │   └── tabular_models/# Scikit-Learn / XGBoost models
│   │   ├── schemas/          # Pydantic v2 input/output validation schemas
│   │   └── services/         # Business logic (Prediction, Storage, Report, History, Profile)
│   ├── storage/uploads/      # Temporary medical upload buffer
│   ├── tests/                # 35+ comprehensive automated pytest suites
│   ├── requirements.txt      # Python dependencies
│   └── Dockerfile            # Container deployment definition
├── frontend/
│   ├── public/               # Static assets & SPA redirects
│   ├── src/
│   │   ├── assets/           # UI graphics and brand icons
│   │   ├── components/       # Reusable components (Forms, Cards, Navbar, ProtectRoute)
│   │   ├── context/          # React AuthContext & Global state
│   │   ├── layouts/          # Responsive App Layouts
│   │   ├── pages/            # View Pages (Home, Diseases, Predict, History, Reports, Profile)
│   │   ├── services/         # Axios API clients
│   │   └── types/            # TypeScript interfaces & API contracts
│   ├── package.json          # Node dependencies & scripts
│   └── vite.config.ts        # Vite build configuration
├── docs/                     # Project technical specifications & guides
└── README.md
```

---

## 🚀 Getting Started Locally

### Prerequisites
- **Python 3.10+** (Python 3.11 / 3.12 recommended)
- **Node.js 18+** & **npm**
- **Git LFS** (Run `git lfs install` to clone large ML model weights)
- **MongoDB Atlas** free M0 cluster connection URI

---

### 1. Backend Setup

```bash
# Navigate to backend directory
cd backend

# Create and activate Python virtual environment
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create .env file from template
cp .env.example .env
```

Configure your `backend/.env` with your settings:
```env
APP_NAME="AI Medical Diagnosis Assistant"
ENVIRONMENT="development"
DEBUG=True
PORT=8000

# MongoDB Atlas
MONGODB_URI="mongodb+srv://<username>:<password>@<cluster>.mongodb.net/?retryWrites=true&w=majority"
DATABASE_NAME="ai_medical_diagnosis"

# JWT Security
JWT_SECRET_KEY="your_secure_random_64_character_hex_key"
JWT_ALGORITHM="HS256"
JWT_ACCESS_TOKEN_EXPIRE_MINUTES=60

# Free Local Storage Driver
STORAGE_DRIVER=local
CORS_ORIGINS="http://localhost:5173,http://127.0.0.1:5173"
```

Start the backend server:
```bash
uvicorn app.main:app --reload --port 8000
```
API Documentation will be live at:
- **Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc**: [http://localhost:8000/redoc](http://localhost:8000/redoc)

---

### 2. Frontend Setup

```bash
# Navigate to frontend directory
cd frontend

# Install packages
npm install

# Start Vite development server
npm run dev
```

Open [http://localhost:5173](http://localhost:5173) in your browser.

---

## 🧪 Running Automated Tests

The repository contains an exhaustive suite of unit and end-to-end integration tests:

```bash
cd backend
python -m pytest tests/ -v
```

Tests cover:
- ✅ JWT Authentication & Authorization isolation
- ✅ Dynamic Disease Discovery & Configuration integrity
- ✅ Image Preprocessing & DenseNet / ResNet inference
- ✅ Tabular Validation & XGBoost / Scikit-learn predictions
- ✅ MongoDB Atlas History persistence & isolation
- ✅ ReportLab PDF generation and download streaming
- ✅ Local storage security & HMAC signed URLs

---

## 🌐 Production Deployment (Free Tier)

### 1. Frontend (Vercel or Netlify)
- **Framework Preset**: Vite
- **Root Directory**: `frontend`
- **Build Command**: `npm run build`
- **Output Directory**: `dist`
- **Environment Variable**: `VITE_API_URL=https://your-backend.onrender.com`

### 2. Backend (Render / Railway / Koyeb)
- **Root Directory**: `backend`
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
- **Environment Variables**: Set `MONGODB_URI`, `JWT_SECRET_KEY`, `STORAGE_DRIVER=local`, `CORS_ORIGINS`.

---

## ⚠️ Clinical Disclaimer

> **IMPORTANT**: This software is designed solely as an **AI-Assisted Educational and Research Prototype** for clinical decision support. It does **NOT** constitute medical advice, a definitive clinical diagnosis, or a replacement for certified medical professionals. Always seek the advice of a qualified healthcare provider with any medical questions.

---

## 📄 License

Distributed under the **MIT License**. See `LICENSE` for more information.
