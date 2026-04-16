# MindEase: AI-Based Multimodal Mental Health Assessment System

## 📌 Overview
MindEase is an AI-based multimodal system designed to assess a user’s mental well-being by analyzing emotions from text, voice, and facial expressions. The system integrates multiple AI models and combines their outputs using a confidence-weighted fusion approach for improved accuracy and reliability.

## 🚀 Key Features
- Multimodal emotion detection (Text, Voice, Facial expressions)
- Text emotion analysis using fine-tuned BERT
- Speech emotion recognition using pretrained Wav2Vec2 model
- Facial emotion detection using DeepFace (pretrained)
- Confidence-weighted fusion of multiple modalities
- Mental health screening using PHQ-9 and GAD-7
- Real-time analysis and response generation

## 🛠️ Tech Stack
- Python
- BERT (fine-tuned for text emotion classification)
- Wav2Vec2 (pretrained speech model)
- DeepFace (pretrained facial emotion recognition)
- NumPy / Pandas
- Machine Learning & Deep Learning techniques

## 📊 Performance
- Accuracy: 95%
- Precision: 94%
- Reduced malformed outputs to 2%
- Average response time: 1.2 seconds

## 🔬 Implementation Details
- Fine-tuned BERT model for text-based emotion detection
- Integrated pretrained models (Wav2Vec2, DeepFace) for multimodal analysis
- Designed a confidence-based fusion mechanism to combine outputs
- Built an end-to-end pipeline for emotion detection and screening

## ▶️ How It Works
1. Input is taken in the form of text, voice, or facial image
2. Each modality is processed using:
   - BERT → Text
   - Wav2Vec2 (pretrained) → Voice
   - DeepFace (pretrained) → Facial expressions
3. Outputs are combined using a fusion algorithm
4. Final emotional state is determined
5. Screening and suggestions are generated

## ⚠️ Disclaimer
This project is intended for educational and research purposes only and is not a substitute for professional mental health diagnosis or treatment.

## 👩‍💻 Author
Maryam Fatima
