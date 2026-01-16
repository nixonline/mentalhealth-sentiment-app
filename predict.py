from flask import Flask, request, jsonify, render_template
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import random

# Static class definitions
CLASSES = [
    "Anxiety", "Bipolar", "Depression", "Normal",
    "Personality disorder", "Stress", "Suicidal"
]

BINARY_CLASSES = ["Depression", "Suicidal"]
DEPR_IDX = CLASSES.index("Depression")
SUIC_IDX = CLASSES.index("Suicidal")
TARGET_CODES = {DEPR_IDX, SUIC_IDX}

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

# Load tokenizer/config from local folder (pre-bundled)
tokenizer = AutoTokenizer.from_pretrained("./model", local_files_only=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.set_num_threads(1)

# Multi-class model
multiclass_model = AutoModelForSequenceClassification.from_pretrained(
    "./model", num_labels=len(CLASSES), local_files_only=True
)
multiclass_model.load_state_dict(torch.load("model_state.pt", map_location=device))
multiclass_model.to(device)
multiclass_model.eval()

# Binary model (Depression vs Suicidal)
binary_model = AutoModelForSequenceClassification.from_pretrained(
    "./model", num_labels=2, local_files_only=True
)
binary_model.load_state_dict(torch.load("binary_model_state.pt", map_location=device))
binary_model.to(device)
binary_model.eval()

def predict_text(texts):
    if isinstance(texts, str):
        texts = [texts]

    encodings = tokenizer(texts, truncation=True, padding=True, return_tensors="pt")
    encodings = {k: v.to(device) for k, v in encodings.items()}

    with torch.no_grad():
        outputs = multiclass_model(**encodings)
        multiclass_labels = outputs.logits.argmax(-1).cpu().numpy()

    refine_indices = [i for i, label in enumerate(multiclass_labels) if label in TARGET_CODES]

    if refine_indices:
        refine_texts = [texts[i] for i in refine_indices]
        refine_encodings = tokenizer(refine_texts, truncation=True, padding=True, return_tensors="pt")
        refine_encodings = {k: v.to(device) for k, v in refine_encodings.items()}

        with torch.no_grad():
            binary_outputs = binary_model(**refine_encodings)
            binary_labels = binary_outputs.logits.argmax(-1).cpu().numpy()

        for j, idx in enumerate(refine_indices):
            multiclass_labels[idx] = DEPR_IDX if binary_labels[j] == 0 else SUIC_IDX

    return [CLASSES[i] for i in multiclass_labels]

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    message = None
    prediction = None
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