from flask import Flask, request, jsonify, render_template
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
import random

# Static class definitions
CLASSES = [
    "Anxiety",
    "Bipolar",
    "Depression",
    "Normal",
    "Personality disorder",
    "Stress",
    "Suicidal",
]

# Binary mapping for secondary classification
BINARY_CLASSES = ["Depression", "Suicidal"]

# Indices for refinement targets in the multiclass space
DEPR_IDX = CLASSES.index("Depression")  # 2
SUIC_IDX = CLASSES.index("Suicidal")    # 6
TARGET_CODES = {DEPR_IDX, SUIC_IDX}

# Result messages
ENCOURAGING_MESSAGES = {
    "Anxiety": [
        "You may be experiencing <strong>Anxiety</strong> — it's okay to take things one step at a time.",
        "Feeling <strong>Anxiety</strong> can be overwhelming, but remember you’re stronger than you think.",
        "Deep breaths can help ease <strong>Anxiety</strong> — calming moments matter.",
        "You’re not alone in facing <strong>Anxiety</strong>; support is always available.",
        "Even in anxious times, small steps forward can ease <strong>Anxiety</strong>."
    ],
    "Bipolar": [
        "Life can feel like ups and downs with <strong>Bipolar</strong> — balance is possible.",
        "Your journey with <strong>Bipolar</strong> may have highs and lows, but growth is always within reach.",
        "Remember, stability with <strong>Bipolar</strong> comes with time and support.",
        "You are resilient even when <strong>Bipolar</strong> moods shift.",
        "Every chapter of your <strong>Bipolar</strong> story matters, even the difficult ones."
    ],
    "Depression": [
        "Life can be tough and you can sometimes experience <strong>Depression</strong> — you're not alone.",
        "Even in dark times, small sparks of hope can grow beyond <strong>Depression</strong>.",
        "<strong>Depression</strong> doesn’t define you; healing is possible.",
        "You deserve kindness, especially when facing <strong>Depression</strong>.",
        "Take it slow — recovery from <strong>Depression</strong> is a journey, not a race."
    ],
    "Normal": [
        "You’re feeling <strong>Normal</strong> and balanced — keep nurturing your well-being.",
        "Enjoy this moment of <strong>Normal</strong> calm and stability.",
        "Balance is a gift — cherish your <strong>Normal</strong> state.",
        "You’re in a good place — keep building on your <strong>Normal</strong> feelings.",
        "Feeling <strong>Normal</strong> and steady is worth celebrating."
    ],
    "Personality disorder": [
        "Everyone has unique challenges — facing a <strong>Personality disorder</strong> doesn’t define you.",
        "Your individuality is part of your strength, even with a <strong>Personality disorder</strong>.",
        "Support can help you navigate complex feelings tied to a <strong>Personality disorder</strong>.",
        "You are more than any <strong>Personality disorder</strong> label.",
        "Healing is a journey, and you’re on the path with your <strong>Personality disorder</strong>."
    ],
    "Stress": [
        "You may be experiencing <strong>Stress</strong> — rest and recharge are important.",
        "Take a deep breath — you’ve got this, even under <strong>Stress</strong>.",
        "<strong>Stress</strong> fades when you give yourself space to recover.",
        "You’re capable of handling challenges one step at a time, even with <strong>Stress</strong>.",
        "Remember to pause and care for yourself when <strong>Stress</strong> builds."
    ],
    "Suicidal": [
        "If you’re feeling <strong>Suicidal</strong>, please know your life matters.",
        "Reaching out for help is a brave step when feeling <strong>Suicidal</strong>.",
        "You are valued, even when <strong>Suicidal</strong> thoughts appear.",
        "Your story isn’t over — support is out there for <strong>Suicidal</strong> feelings.",
        "Hold on — brighter days can come with time and care, even if you feel <strong>Suicidal</strong>."
    ]
}

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Multi-class model
multiclass_model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=len(CLASSES)
)
multiclass_model.load_state_dict(torch.load("model_state.pt", map_location=device))
multiclass_model.to(device)
multiclass_model.eval()
trainer = Trainer(model=multiclass_model, tokenizer=tokenizer)

# Binary model (Depression vs Suicidal)
binary_model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=2
)
binary_model.load_state_dict(torch.load("binary_model_state.pt", map_location=device))
binary_model.to(device)
binary_model.eval()
binary_trainer = Trainer(model=binary_model, tokenizer=tokenizer)

def predict_text(texts):
    if isinstance(texts, str):
        texts = [texts]

    # Multiclass predictions
    encodings = tokenizer(texts, truncation=True, padding=True)
    ds = Dataset.from_dict(encodings)
    preds = trainer.predict(ds)
    multiclass_labels = preds.predictions.argmax(-1)

    # Identify indices needing binary refinement (Depression vs Suicidal)
    refine_indices = [i for i, label in enumerate(multiclass_labels) if label in TARGET_CODES]

    if refine_indices:
        refine_texts = [texts[i] for i in refine_indices]
        refine_encodings = tokenizer(refine_texts, truncation=True, padding=True)
        refine_ds = Dataset.from_dict(refine_encodings)

        binary_preds = binary_trainer.predict(refine_ds)
        binary_labels = binary_preds.predictions.argmax(-1)  # 0 -> Depression, 1 -> Suicidal

        for j, idx in enumerate(refine_indices):
            multiclass_labels[idx] = DEPR_IDX if binary_labels[j] == 0 else SUIC_IDX

    return [CLASSES[i] for i in multiclass_labels]

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    message = None
    prediction = None   # <-- initialize here
    if request.method == "POST":
        message = request.form.get("message", "")
        if message:
            prediction = predict_text(message)[0]
            result = random.choice(ENCOURAGING_MESSAGES.get(prediction, [prediction]))
    return render_template("index.html", result=result, message=message, label=prediction)

@app.route("/predict", methods=["POST"])
def predict_api():
    data = request.get_json(force=True)
    texts = data.get("texts", [])
    if isinstance(texts, str):
        texts = [texts]
    predictions = predict_text(texts)
    return jsonify({"predictions": predictions})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)