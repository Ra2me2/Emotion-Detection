# Audio Emotion Detection using Wav2Vec2BERT

This project demonstrates Parameter Efficient Fine-Tuning (PEFT) of the Wav2Vec2BERT model from Hugging Face's Transformers library. The model is fine-tuned to recognize emotions from audio clips

## Project Overview

This repository contains the implementation for training an emotion detection model using audio data. The key steps include:
1. Loading and preprocessing audio data.
2. Extracting features using a pre-trained Wav2Vec2BERT model.
3. Training the model with the PEFT approach using LoRA (Low-Rank Adaptation).
4. Evaluating the model performance using standard metrics like accuracy, precision, recall, and F1 score.
5. Visualizing the training and evaluation results.

## Frameworks and Libraries Used

- **Python**: Programming language
- **PyTorch**: Deep learning framework
- **Hugging Face Transformers**: For pre-trained models and feature extraction
- **PEFT (LoRA)**: For parameter-efficient fine-tuning
- **Librosa**: For audio processing
- **Scikit-learn**: For evaluation metrics
- **Matplotlib and Seaborn**: For data visualization
- **Pandas**: For handling and saving metrics

## Dataset

The project uses the **RAVDESS Emotional speech audio**  available on [kaggle](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio).
