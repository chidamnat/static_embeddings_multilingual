"""Multilingual retrieval recall test: fp32 vs q8.

Mirrors the JS runtime: tokenize with tokenizer.json + run the ONNX directly.
No model2vec dependency.
"""
import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

REPO = "minishlab/potion-multilingual-128M"
tok = Tokenizer.from_file(hf_hub_download(REPO, "tokenizer.json"))

so = ort.SessionOptions(); so.log_severity_level = 3
fp32 = ort.InferenceSession("models/model.onnx", so, providers=["CPUExecutionProvider"])
q8   = ort.InferenceSession("models/model_quantized.onnx", so, providers=["CPUExecutionProvider"])


def embed(session, texts):
    """EmbeddingBag-style feed: flat ids + per-text offsets. Returns [N, 256]."""
    flat, offsets = [], []
    for t in texts:
        offsets.append(len(flat))
        ids = tok.encode(t, add_special_tokens=False).ids
        if not ids:                      # never feed an empty sequence
            ids = [tok.token_to_id("<unk>") or 0]
        flat.extend(ids)
    out = session.run(None, {
        "input_ids": np.array(flat, dtype=np.int64),
        "offsets":   np.array(offsets, dtype=np.int64),
    })[0]
    return out  # model already L2-normalizes


# (query, gold-relevant document) pairs across languages.
PAIRS = [
    ("How do I reset my password?", "To change your account password, open settings and select 'Reset password'."),
    ("What time does the store close today?", "Our shop closes at 9 PM on weekdays and 6 PM on weekends."),
    ("¿Cuál es la capital de Francia?", "París es la capital y la ciudad más poblada de Francia."),
    ("Comment cuire des pâtes correctement ?", "Faites bouillir l'eau salée, ajoutez les pâtes et cuisez environ dix minutes."),
    ("Wie ist das Wetter morgen?", "Morgen wird es sonnig mit einer Höchsttemperatur von 22 Grad."),
    ("東京で一番高い建物は何ですか？", "東京スカイツリーは高さ634メートルで日本一高い構造物です。"),
    ("北京有哪些著名的旅游景点？", "故宫和长城是北京最受欢迎的旅游景点。"),
    ("Какая сегодня температура воздуха?", "Сегодня на улице около пятнадцати градусов тепла и небольшой ветер."),
    ("How much does shipping cost?", "Standard delivery is free for orders over fifty dollars."),
    ("What are the symptoms of the flu?", "Influenza usually causes fever, body aches, fatigue and a sore throat."),
    ("Quanto costa un biglietto del treno per Roma?", "Il prezzo del biglietto ferroviario per Roma parte da venti euro."),
    ("كيف يمكنني تعلم البرمجة؟", "ابدأ بتعلم أساسيات لغة بايثون من خلال الدورات المجانية عبر الإنترنت."),
    ("Best way to cook rice?", "Rinse the rice, add two cups of water per cup of rice, and simmer covered."),
    ("Who painted the Mona Lisa?", "The Mona Lisa was painted by Leonardo da Vinci during the Renaissance."),
    ("How long is a marathon?", "A standard marathon race covers 42.195 kilometers."),
    ("Wann wurde die Berliner Mauer gebaut?", "Die Berliner Mauer wurde im Jahr 1961 errichtet."),

    # --- Tamil weather cluster (hard negatives for each other) ---
    ("நாளை வானிலை எப்படி இருக்கும்?",
     "நாளை வெயிலுடன் கூடிய தெளிவான வானிலை இருக்கும், அதிகபட்ச வெப்பநிலை 34 டிகிரி."),
    ("இன்று மழை பெய்யுமா?",
     "இன்று மதியம் சிறிய மழைக்கு வாய்ப்புள்ளது, மாலையில் வானம் தெளிவாகும்."),

    # --- Tamil monolingual ---
    ("என் கடவுச்சொல்லை எப்படி மாற்றுவது?",
     "உங்கள் கணக்கின் கடவுச்சொல்லை மாற்ற, அமைப்புகளைத் திறந்து 'கடவுச்சொல்லை மீட்டமை' என்பதைத் தேர்ந்தெடுக்கவும்."),
    ("சாதம் எப்படி சமைப்பது?",
     "அரிசியை கழுவி, ஒரு கப் அரிசிக்கு இரண்டு கப் தண்ணீர் சேர்த்து மூடி வேகவைக்கவும்."),
    ("சென்னைக்கு ரயில் டிக்கெட் விலை எவ்வளவு?",
     "சென்னைக்கான ரயில் பயணச்சீட்டின் விலை இருநூறு ரூபாயிலிருந்து தொடங்குகிறது."),
    ("காய்ச்சலின் அறிகுறிகள் என்ன?",
     "காய்ச்சல், உடல் வலி, சோர்வு மற்றும் தொண்டை வலி ஆகியவை காய்ச்சலின் பொதுவான அறிகுறிகள்."),
    ("நிரலாக்கத்தை எப்படி கற்றுக்கொள்வது?",
     "இலவச ஆன்லைன் பாடநெறிகள் மூலம் பைதான் மொழியின் அடிப்படைகளைக் கற்றுக்கொள்ளத் தொடங்குங்கள்."),

     # --- Cross-lingual (English query <-> Tamil doc, and vice versa) ---
    ("What is the capital of France?",
     "பாரிஸ் பிரான்சின் தலைநகரம் மற்றும் மிக அதிக மக்கள்தொகை கொண்ட நகரம்."),
    ("மொனாலிசா ஓவியத்தை வரைந்தவர் யார்?",
     "The Mona Lisa was painted by Leonardo da Vinci during the Renaissance."),
]

queries = [q for q, _ in PAIRS]
docs    = [d for _, d in PAIRS]
gold    = list(range(len(PAIRS)))   # query i's relevant doc is doc i


def recall(sims, ks=(1, 3)):
    ranks = np.argsort(-sims, axis=1)
    res = {k: sum(gold[i] in ranks[i, :k] for i in range(len(gold))) / len(gold) for k in ks}
    return res, ranks[:, 0]


q_fp, d_fp = embed(fp32, queries), embed(fp32, docs)
q_q8, d_q8 = embed(q8, queries),   embed(q8, docs)

sims_fp = q_fp @ d_fp.T          # both unit-norm -> cosine
sims_q8 = q_q8 @ d_q8.T

r_fp, top1_fp = recall(sims_fp)
r_q8, top1_q8 = recall(sims_q8)
agreement = float(np.mean(top1_fp == top1_q8))

print(f"corpus: {len(PAIRS)} multilingual query/doc pairs\n")
print(f"  fp32   recall@1 = {r_fp[1]:.3f}   recall@3 = {r_fp[3]:.3f}")
print(f"  q8     recall@1 = {r_q8[1]:.3f}   recall@3 = {r_q8[3]:.3f}")
print(f"  q8-vs-fp32 top-1 agreement = {agreement:.3f}")

# Per-pair similarity scores: query <-> its gold doc, fp32 vs q8.
print("\nper-pair cosine(query, gold doc):")
print(f"  {'#':>2}  {'fp32':>6}  {'q8':>6}  {'Δ':>6}  hit  query")
deltas = []
for i in range(len(PAIRS)):
    g_fp = float(sims_fp[i, gold[i]])
    g_q8 = float(sims_q8[i, gold[i]])
    d = g_q8 - g_fp
    deltas.append(abs(d))
    hit = "ok " if top1_fp[i] == gold[i] else "MISS"
    q = queries[i] if len(queries[i]) <= 42 else queries[i][:39] + "..."
    line = f"  {i:>2}  {g_fp:6.3f}  {g_q8:6.3f}  {d:+6.3f}  {hit}  {q}"
    if top1_fp[i] != gold[i]:
        # show what fp32 actually ranked first instead of the gold doc
        wrong = top1_fp[i]
        line += f"\n        -> fp32 top-1 was doc {wrong} (cos {float(sims_fp[i, wrong]):.3f}): {docs[wrong][:50]}"
    print(line)

print(f"\n  mean |Δ cosine to gold| (fp32 vs q8): {np.mean(deltas):.4f}")
print(f"  max  |Δ cosine to gold|:               {np.max(deltas):.4f}")
