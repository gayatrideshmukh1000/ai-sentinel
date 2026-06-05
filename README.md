# 🌾 AI-Sentinel: Sugarcane Leaf Disease Detection

A full-stack AI web application that detects sugarcane leaf diseases from uploaded photos using a CNN model built with TensorFlow/Keras.

---

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the app (demo mode — no trained model needed)
python app.py

# 3. Open in browser
http://localhost:5000
```

---

## 🧬 Training Your Own Model

### Step 1 — Get the Dataset
Download from Kaggle: [Sugarcane Leaf Disease Dataset](https://www.kaggle.com/datasets/nirmalsankalana/sugarcane-leaf-disease-dataset)

Extract so the structure looks like:
```
dataset/
  Healthy/         *.jpg
  Red Rot/         *.jpg
  Rust/            *.jpg
  Leaf Scald/      *.jpg
  Yellow Leaf/     *.jpg
```

### Step 2 — Train
```bash
python train_model.py --data_dir ./dataset --epochs 25
```

The best model is auto-saved to `model/sugarcane_cnn.h5`.

### Step 3 — Launch with Trained Model
```bash
python app.py
```

---

## 🏗️ Project Structure

```
ai-sentinel/
├── app.py               # Flask backend + prediction logic
├── train_model.py       # CNN training script (MobileNetV2)
├── requirements.txt     # Python dependencies
├── templates/
│   └── index.html       # Complete frontend (HTML/CSS/JS)
├── model/
│   └── sugarcane_cnn.h5 # (generated after training)
└── static/
    └── uploads/         # Temp upload storage
```

---

## 🌿 Detectable Diseases

| Disease | Severity | Cause |
|---------|----------|-------|
| Healthy | None | — |
| Red Rot | High | *Colletotrichum falcatum* |
| Rust | Medium | *Puccinia melanocephala* |
| Leaf Scald | High | *Xanthomonas albilineans* |
| Yellow Leaf | Medium | Sugarcane Yellow Leaf Virus |

---

## 🖥️ Tech Stack

- **Frontend**: HTML5 + CSS3 + Vanilla JS (no build step)
- **Backend**: Flask (Python)
- **Model**: MobileNetV2 transfer learning via TensorFlow/Keras
- **Image processing**: Pillow + NumPy

---

## ☁️ Deploy to Production

```bash
# Gunicorn (Linux/Mac)
gunicorn app:app -w 2 -b 0.0.0.0:8000

# Docker (optional)
docker build -t ai-sentinel .
docker run -p 5000:5000 ai-sentinel
```

For cloud deployment: Render, Railway, or any VPS with Python 3.10+.

---

## 📝 Notes

- Without a trained model, the app runs in **demo mode** with mock predictions so you can explore the UI immediately.
- The training script uses **MobileNetV2** transfer learning — fast to train and mobile-friendly.
- All inference runs server-side; no data is stored permanently.
