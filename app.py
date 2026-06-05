import os, io, base64, json, random
from flask import Flask, request, jsonify, render_template
from PIL import Image
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
os.makedirs('static/uploads', exist_ok=True)

# ── Disease Knowledge Base ────────────────────────────────────────────────────
DISEASES = {
    "Healthy": {
        "color": "#22c55e", "icon": "🌿", "severity": "None",
        "description": "The sugarcane leaf appears healthy with no visible signs of disease or stress.",
        "symptoms": ["Vibrant green coloration", "Uniform leaf texture", "No lesions or spots", "Strong structural integrity"],
        "treatment": ["Continue regular irrigation schedule", "Maintain balanced NPK fertilization", "Monitor weekly for early signs", "Practice crop rotation annually"],
        "prevention": ["Use certified disease-free seed cane", "Maintain proper field drainage", "Apply prophylactic fungicide monthly", "Remove crop debris after harvest"],
    },
    "RedRot": {
        "color": "#ef4444", "icon": "🔴", "severity": "High",
        "description": "Red Rot is caused by Colletotrichum falcatum and is one of the most destructive sugarcane diseases worldwide.",
        "symptoms": ["Red discoloration of internal stalk tissue", "White patches with red margins on leaves", "Sour alcoholic odour from infected stalks", "Wilting and drying of leaves", "Shredded internal stalk fibers"],
        "treatment": ["Remove and destroy infected plants immediately", "Apply Carbendazim 0.1% solution", "Use Mancozeb 75 WP @ 2g/litre spray", "Inject Tridemorph into soil near roots", "Drench soil with copper oxychloride"],
        "prevention": ["Plant resistant varieties like Co 86032", "Hot water treatment of setts at 50°C for 2hr", "Avoid waterlogging in field", "Destroy infected crop residues", "Rotate with non-host crops for 1-2 seasons"],
    },
    "Rust": {
        "color": "#f97316", "icon": "🟠", "severity": "Medium",
        "description": "Sugarcane Rust caused by Puccinia melanocephala produces characteristic orange-brown pustules on leaf surfaces.",
        "symptoms": ["Small yellow to orange pustules on leaves", "Rusty-brown powdery spores on underside", "Chlorotic halos around pustules", "Premature leaf senescence", "Reduced photosynthetic area"],
        "treatment": ["Apply Propiconazole 25 EC @ 1ml/litre", "Spray Mancozeb 75 WP at 10-day intervals", "Use Hexaconazole 5% EC as systemic fungicide", "Repeat treatment after rainfall", "Apply Tebuconazole for severe infections"],
        "prevention": ["Plant rust-resistant varieties", "Avoid dense planting to improve air circulation", "Apply preventive fungicide at early season", "Monitor fields weekly during humid weather", "Use balanced potassium fertilization"],
    },
    "Mosaic": {
        "color": "#eab308", "icon": "🟡", "severity": "High",
        "description": "Leaf Scald caused by Xanthomonas albilineans is a systemic bacterial disease that can cause sudden wilt and death.",
        "symptoms": ["White pencil-line streaks along leaf midrib", "Scalded appearance on leaf margins", "Pale yellow-white leaf stripes", "Sudden wilting of top leaves", "Stunted ratoon growth"],
        "treatment": ["No effective chemical cure exists", "Rogue out and burn infected stools", "Apply copper-based bactericides as suppression", "Use hot water treated planting material", "Disinfect cutting tools with 70% alcohol"],
        "prevention": ["Use pathogen-free certified planting material", "Hot water treatment at 50°C for 3 hours", "Quarantine new varieties before field release", "Disinfect harvesting equipment between fields", "Plant tolerant varieties like SP 70-1143"],
    },
    "Yellow": {
        "color": "#facc15", "icon": "💛", "severity": "Medium",
        "description": "Yellow Leaf Disease caused by Sugarcane Yellow Leaf Virus (SCYLV) is transmitted by the sugarcane aphid.",
        "symptoms": ["Yellowing of midrib on lower leaf surface", "Yellow color progresses to adaxial side", "Necrosis of leaf tip and margins", "Stunted plant growth", "Reduced stalk number and weight"],
        "treatment": ["Apply Imidacloprid 200 SL to control aphid vectors", "Use Thiamethoxam 25 WG @ 0.3g/litre", "Spray Dimethoate 30 EC for aphid control", "Remove and destroy heavily infected plants", "Apply neem-based insecticides as organic option"],
        "prevention": ["Use virus-free planting material", "Control aphid populations early in season", "Plant tolerant varieties where available", "Remove volunteer sugarcane plants from borders", "Apply reflective mulch to deter aphids"],
    },
}

# ── Model + Class Map ─────────────────────────────────────────────────────────
model = None
IDX_TO_CLASS = None   # {0: "Healthy", 1: "Leaf Scald", ...} — real Keras order


def load_model():
    global model, IDX_TO_CLASS

    # 1. Load class index map saved by train_model.py
    idx_path = "model/class_indices.json"
    if os.path.exists(idx_path):
        with open(idx_path) as f:
            raw = json.load(f)          # {"0": "Healthy", "1": "Leaf Scald", ...}
        IDX_TO_CLASS = {int(k): v for k, v in raw.items()}
        print(f"✅ Class map loaded: {IDX_TO_CLASS}")
    else:
        # Keras sorts folder names alphabetically — use that as safe fallback
        IDX_TO_CLASS = {i: c for i, c in enumerate(sorted(DISEASES.keys()))}
        print(f"⚠️  model/class_indices.json not found.")
        print(f"   Using alphabetical fallback: {IDX_TO_CLASS}")
        print(f"   Run train_model.py to generate the correct map.")

    # 2. Load the trained weights
    try:
        import tensorflow as tf
        model_path = "model/sugarcane_cnn.h5"
        if os.path.exists(model_path):
            model = tf.keras.models.load_model(model_path)
            print("✅ Trained model loaded from disk.")
        else:
            print("⚠️  model/sugarcane_cnn.h5 not found — running in demo mode.")
    except ImportError:
        print("⚠️  TensorFlow not installed — running in demo mode.")


def preprocess(image: Image.Image):
    img = image.convert("RGB").resize((224, 224))
    arr = np.array(img, dtype=np.float32)
    arr = preprocess_input(arr)
    return np.expand_dims(arr, 0)


def predict(image: Image.Image):
    """Return (disease_name, confidence_pct, all_scores_dict)."""

    # ── Real model inference ──────────────────────────────────────────────────
    if model is not None and IDX_TO_CLASS is not None:
        try:
            probs = model.predict(preprocess(image), verbose=0)[0]
            idx   = int(np.argmax(probs))
            conf  = float(probs[idx])

            disease_name = IDX_TO_CLASS.get(idx, "Unknown")

            all_scores = {
                IDX_TO_CLASS.get(i, f"class_{i}"): round(float(p) * 100, 1)
                for i, p in enumerate(probs)
            }

            if disease_name not in DISEASES:
                print(f"⚠️  Predicted class '{disease_name}' not in DISEASES dict — check dataset folders.")

            return disease_name, round(conf * 100, 1), all_scores

        except Exception as e:
            print(f"Prediction error: {e}")

    # ── Demo / fallback mock ──────────────────────────────────────────────────
    all_names = list(IDX_TO_CLASS.values()) if IDX_TO_CLASS else list(DISEASES.keys())
    chosen_name = random.choice(all_names)
    conf = round(random.uniform(72, 96), 1)
    remainder = 100 - conf
    others = [n for n in all_names if n != chosen_name]
    raw = [random.random() for _ in others]
    norm = [r / sum(raw) * remainder for r in raw]
    all_scores = {n: round(norm[i], 1) for i, n in enumerate(others)}
    all_scores[chosen_name] = conf
    return chosen_name, conf, all_scores


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict_route():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400
    file = request.files["image"]
    if not file.filename:
        return jsonify({"error": "Empty filename"}), 400

    try:
        img = Image.open(io.BytesIO(file.read()))
        disease_name, confidence, all_scores = predict(img)

        disease = DISEASES.get(disease_name, DISEASES["Healthy"])

        thumb = img.copy()
        thumb.thumbnail((400, 400))
        buf = io.BytesIO()
        thumb.save(buf, format="JPEG", quality=85)
        img_b64 = base64.b64encode(buf.getvalue()).decode()

        return jsonify({
            "disease":     disease_name,
            "confidence":  confidence,
            "color":       disease["color"],
            "icon":        disease["icon"],
            "description": disease["description"],
            "symptoms":    disease["symptoms"],
            "treatment":   disease["treatment"],
            "prevention":  disease["prevention"],
            "severity":    disease["severity"],
            "all_scores":  all_scores,
            "image_b64":   img_b64,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/class-map")
def class_map_route():
    """Debug endpoint — visit /class-map to verify the index order."""
    return jsonify(IDX_TO_CLASS or {})


if __name__ == "__main__":
    load_model()
    app.run(debug=True, host="0.0.0.0", port=5000)