from transformers import pipeline, AutoConfig

# 1. Load the model configuration to get the intent mapping
config = AutoConfig.from_pretrained("./saved_intent_model")
id2label = config.id2label

# 2. Load the trained model pipeline
classifier = pipeline(
    "text-classification",
    model="./saved_intent_model",
    tokenizer="./saved_intent_model"
)

# 3. Test phrases that match your dataset
test_phrases = [
    "Hello",
    "Hi there", 
    "What's your name?",
    "Open the pod bay doors",
    "Goodbye",
    "Tell me a joke",
    "Who am I?",
    "What time is it?"
]

print("🤖 Intent Classification Model - Live Demo")
print("=" * 50)
print(f"Available Intents: {list(id2label.values())}")
print("=" * 50)

# 4. Test and display results
for phrase in test_phrases:
    result = classifier(phrase)
    intent = result[0]['label']
    confidence = result[0]['score']
    print(f"Input: '{phrase}'")
    print(f"→ Predicted Intent: {intent} ({(confidence*100):.1f}% confidence)")
    print("-" * 40)