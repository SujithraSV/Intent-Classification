# Intent Classification System

A simple intent classification system using DistilBERT to automatically categorize user queries for chatbots.

## What It Does

- Takes user text like "Hello" or "What time is it?"
- Predicts the intent (Greeting, TimeQuery, etc.)
- Gives a confidence score for each prediction

## Quick Start

1. Install dependencies:
```bash
pip install transformers torch pandas
Train the model:

bash
python train_model.py
Test with examples:

bash
python test_model.py


## Project Files
-Intent.json - Training data with examples

-train_model.py - Trains the model

-test_model.py - Tests the model

-saved_intent_model/ - Trained model files
