"""
evaluate_recommender.py
========================
تقييم نظام التوصية الهجين (TF-IDF + SVD)
يعمل بشكل مستقل دون Django — يحتاج فقط:
  - enhanced_places_dataset_translated.csv
  - user_ratings.csv  (الملف الذي تم توليده)

تشغيل:
    pip install pandas numpy scikit-learn scipy surprise matplotlib seaborn
    python evaluate_recommender.py
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ===================================================
# 0. تحميل البيانات
# ===================================================
print("=" * 60)
print("  تقييم نظام التوصية الهجين (TF-IDF + SVD)")
print("=" * 60)

places_df = pd.read_csv("enhanced_places_dataset_translated.csv", skipinitialspace=True)
places_df.columns = places_df.columns.str.strip()
places_df["Destination"] = places_df["Destination"].str.strip()
places_df["place_id"] = range(1, len(places_df) + 1)

ratings_df = pd.read_csv("user_ratings.csv")

print(f"\n[البيانات]")
print(f"  عدد الأماكن     : {len(places_df)}")
print(f"  عدد المستخدمين  : {ratings_df['user_id'].nunique()}")
print(f"  إجمالي التقييمات: {len(ratings_df)}")
print(f"  متوسط التقييم   : {ratings_df['rating'].mean():.2f}")
print(f"  الانحراف المعياري: {ratings_df['rating'].std():.2f}")


# ===================================================
# 1. بناء نموذج TF-IDF (Content-Based Filtering)
# ===================================================
print("\n" + "─" * 60)
print("  [1] تقييم نموذج TF-IDF (Content-Based Filtering)")
print("─" * 60)

ARABIC_STOP_WORDS = [
    "في","من","إلى","على","عن","مع","هذا","هذه","ذلك","تلك",
    "التي","الذي","هو","هي","هم","هن","كان","كانت","يكون",
    "ما","لا","لم","لن","إن","أن","كما","أو","و","ثم","حتى",
    "بعد","قبل","عند","حين","إذا","لو","كل","بعض","غير","بين",
]

# بناء نص موحد لكل مكان
def build_place_text(row):
    fields = [
        str(row.get("Destination", "")),
        str(row.get("City", "")),
        str(row.get("Country", "")),
        str(row.get("Category", "")),
        str(row.get("Interests", "")),
        str(row.get("Features", "")),
        str(row.get("semantic_description", "")),
        str(row.get("semantic_description_ar", "")),
    ]
    return " ".join(fields).lower()

places_df["text"] = places_df.apply(build_place_text, axis=1)

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
tfidf = TfidfVectorizer(
    stop_words=list(ARABIC_STOP_WORDS) + list(ENGLISH_STOP_WORDS),
    ngram_range=(1, 2),
    min_df=1
)
tfidf_matrix = tfidf.fit_transform(places_df["text"])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

print(f"  حجم مصفوفة TF-IDF : {tfidf_matrix.shape}")
print(f"  عدد المصطلحات     : {len(tfidf.vocabulary_)}")

# --- مقياس Intra-List Similarity (ILS) ---
# يقيس مدى تنوع التوصيات — القيمة المنخفضة تعني تنوعاً أعلى
def get_content_recommendations(place_idx, top_k=10):
    scores = list(enumerate(cosine_sim[place_idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    scores = [s for s in scores if s[0] != place_idx]
    return [s[0] for s in scores[:top_k]]

ils_scores = []
for idx in range(len(places_df)):
    recs = get_content_recommendations(idx, top_k=5)
    if len(recs) < 2:
        continue
    rec_vectors = tfidf_matrix[recs]
    sim_matrix = cosine_similarity(rec_vectors)
    n = len(recs)
    # متوسط التشابه بين كل زوج (بدون القطر)
    upper = [sim_matrix[i][j] for i in range(n) for j in range(i+1, n)]
    ils_scores.append(np.mean(upper))

avg_ils = np.mean(ils_scores)
diversity = 1 - avg_ils

# --- مقياس Coverage ---
all_recommended = set()
for idx in range(len(places_df)):
    recs = get_content_recommendations(idx, top_k=5)
    all_recommended.update(recs)
coverage = len(all_recommended) / len(places_df)

# --- Precision@K بناءً على الفئة ---
# نعتبر التوصية "صحيحة" إذا كانت من نفس الفئة أو الفئة مرتبطة
RELATED_CATEGORIES = {
    "Historical": ["Archaeological", "Cultural"],
    "Archaeological": ["Historical", "Cultural"],
    "Cultural": ["Historical", "Market"],
    "Beach": ["Nature", "Scenic"],
    "Nature": ["Beach", "Scenic", "Adventure"],
    "Scenic": ["Nature", "Beach"],
    "Adventure": ["Nature", "Scenic"],
    "Market": ["Cultural"],
    "Religious": ["Historical", "Cultural"],
}

def is_relevant(source_cat, target_cat):
    source_cat = source_cat.strip()
    target_cat = target_cat.strip()
    if source_cat == target_cat:
        return True
    related = RELATED_CATEGORIES.get(source_cat, [])
    return target_cat in related

precision_scores = []
for idx, row in places_df.iterrows():
    recs = get_content_recommendations(idx, top_k=5)
    source_cat = str(row["Category"]).strip()
    hits = sum(1 for r in recs if is_relevant(source_cat, str(places_df.iloc[r]["Category"])))
    precision_scores.append(hits / 5)

avg_precision = np.mean(precision_scores)

print(f"\n  النتائج:")
print(f"  ┌─────────────────────────────────────┬────────┐")
print(f"  │ المقياس                             │ القيمة │")
print(f"  ├─────────────────────────────────────┼────────┤")
print(f"  │ Precision@5 (بالفئة)                │ {avg_precision:.3f}  │")
print(f"  │ Diversity (1 - ILS)                 │ {diversity:.3f}  │")
print(f"  │ Catalog Coverage                    │ {coverage:.3f}  │")
print(f"  └─────────────────────────────────────┴────────┘")


# ===================================================
# 2. بناء نموذج SVD (Collaborative Filtering)
# ===================================================
print("\n" + "─" * 60)
print("  [2] تقييم نموذج SVD (Collaborative Filtering)")
print("─" * 60)

try:
    from surprise import Dataset, Reader, SVD, accuracy
    from surprise.model_selection import cross_validate, KFold
    SURPRISE_AVAILABLE = True
except ImportError:
    SURPRISE_AVAILABLE = False
    print("  [تحذير] مكتبة surprise غير مثبتة.")
    print("  تثبيت: pip install scikit-surprise")

if SURPRISE_AVAILABLE:
    reader = Reader(rating_scale=(1.0, 5.0))
    data = Dataset.load_from_df(
        ratings_df[["user_id", "place_id", "rating"]], reader
    )

    # Cross-Validation بـ 5 folds
    svd_model = SVD(n_factors=50, n_epochs=20, lr_all=0.005, reg_all=0.02, random_state=42)
    cv_results = cross_validate(
        svd_model, data,
        measures=["RMSE", "MAE"],
        cv=5,
        verbose=False
    )

    rmse_mean = np.mean(cv_results["test_rmse"])
    rmse_std  = np.std(cv_results["test_rmse"])
    mae_mean  = np.mean(cv_results["test_mae"])
    mae_std   = np.std(cv_results["test_mae"])

    print(f"\n  Cross-Validation (5-Fold):")
    print(f"  ┌─────────────────────────────────────┬──────────────┐")
    print(f"  │ المقياس                             │    القيمة    │")
    print(f"  ├─────────────────────────────────────┼──────────────┤")
    print(f"  │ RMSE (متوسط ± انحراف)              │ {rmse_mean:.4f} ± {rmse_std:.4f} │")
    print(f"  │ MAE  (متوسط ± انحراف)              │ {mae_mean:.4f} ± {mae_std:.4f} │")
    print(f"  └─────────────────────────────────────┴──────────────┘")

    # Hit Rate@K (HR@5) — هل المكان الصحيح ضمن أفضل 5 توصيات؟
    from surprise.model_selection import train_test_split as surprise_split
    trainset, testset = surprise_split(data, test_size=0.2, random_state=42)
    svd_model.fit(trainset)
    predictions = svd_model.test(testset)

    # تجميع التوقعات لكل مستخدم
    user_pred = {}
    for pred in predictions:
        uid = pred.uid
        if uid not in user_pred:
            user_pred[uid] = []
        user_pred[uid].append((pred.iid, pred.est, pred.r_ui))

    hit_at_5 = []
    ndcg_at_5 = []
    for uid, preds in user_pred.items():
        # المكان الحقيقي الأعلى تقييماً في مجموعة الاختبار
        true_best = max(preds, key=lambda x: x[2])[0]
        # أفضل 5 توصيات حسب التوقع
        top5 = sorted(preds, key=lambda x: x[1], reverse=True)[:5]
        top5_ids = [p[0] for p in top5]
        hit_at_5.append(1 if true_best in top5_ids else 0)

        # NDCG@5
        dcg = 0
        for rank, (iid, est, r_ui) in enumerate(top5, start=1):
            if iid == true_best:
                dcg += 1 / np.log2(rank + 1)
        idcg = 1.0  # المثالي: المكان الصحيح في المرتبة الأولى
        ndcg_at_5.append(dcg / idcg)

    hr5   = np.mean(hit_at_5)
    ndcg5 = np.mean(ndcg_at_5)

    print(f"\n  مقاييس الترتيب (Ranking Metrics):")
    print(f"  ┌─────────────────────────────────────┬────────┐")
    print(f"  │ Hit Rate@5                          │ {hr5:.3f}  │")
    print(f"  │ NDCG@5                              │ {ndcg5:.3f}  │")
    print(f"  └─────────────────────────────────────┴────────┘")


# ===================================================
# 3. تقييم النظام الهجين
# ===================================================
print("\n" + "─" * 60)
print("  [3] تقييم النظام الهجين (TF-IDF + SVD)")
print("─" * 60)

# محاكاة الدمج بثلاثة أوضاع (كما في views.py)
WEIGHT_SCENARIOS = [
    ("مستخدم جديد جداً  (<5 تقييمات)" , 1.0, 0.0),
    ("مستخدم جديد       (<15 تقييم)" , 0.8, 0.2),
    ("مستخدم نشط        (≥15 تقييم)" , 0.5, 0.5),
]

if SURPRISE_AVAILABLE:
    # بناء النموذج على كامل البيانات للاستخدام في الهجين
    full_trainset = data.build_full_trainset()
    svd_model.fit(full_trainset)

    def get_svd_score(user_id, place_id):
        try:
            pred = svd_model.predict(user_id, place_id)
            return pred.est
        except:
            return 0.0

    def get_hybrid_score(user_id, place_idx, content_score, w_content, w_collab):
        place_id = places_df.iloc[place_idx]["place_id"]
        svd_score = get_svd_score(user_id, place_id)
        collab_score = (svd_score / 5.0) * 100 if svd_score > 0 else content_score
        return (content_score * w_content) + (collab_score * w_collab)

    print(f"\n  محاكاة سيناريوهات الدمج:")
    print(f"  ┌───────────────────────────────────────┬────────┬────────┐")
    print(f"  │ السيناريو                             │ w_cont │ w_coll │")
    print(f"  ├───────────────────────────────────────┼────────┼────────┤")
    for name, wc, wcoll in WEIGHT_SCENARIOS:
        print(f"  │ {name} │  {wc:.1f}   │  {wcoll:.1f}   │")
    print(f"  └───────────────────────────────────────┴────────┴────────┘")

    # Precision@5 للنظام الهجين بكل سيناريو
    print(f"\n  Precision@5 للنظام الهجين:")
    print(f"  ┌───────────────────────────────────────┬────────┐")
    print(f"  │ السيناريو                             │  P@5   │")
    print(f"  ├───────────────────────────────────────┼────────┤")

    sample_users = ratings_df["user_id"].unique()[:20]  # عينة 20 مستخدم

    for name, wc, wcoll in WEIGHT_SCENARIOS:
        prec_list = []
        for uid in sample_users:
            user_rated = set(ratings_df[ratings_df["user_id"] == uid]["place_id"].tolist())
            hybrid_scores = []
            for idx, row in places_df.iterrows():
                if row["place_id"] in user_rated:
                    continue
                c_score = float(cosine_sim[idx].mean() * 100)
                h_score = get_hybrid_score(uid, idx, c_score, wc, wcoll)
                hybrid_scores.append((idx, h_score))

            if not hybrid_scores:
                continue
            hybrid_scores.sort(key=lambda x: x[1], reverse=True)
            top5_idx = [h[0] for h in hybrid_scores[:5]]

            # تقييم بالفئة
            user_top_cat = ratings_df[ratings_df["user_id"] == uid].merge(
                places_df[["place_id", "Category"]], on="place_id"
            ).sort_values("rating", ascending=False)["Category"].iloc[0].strip()

            hits = sum(1 for i in top5_idx
                       if is_relevant(user_top_cat, str(places_df.iloc[i]["Category"]).strip()))
            prec_list.append(hits / 5)

        avg_p = np.mean(prec_list) if prec_list else 0
        print(f"  │ {name} │ {avg_p:.3f}  │")

    print(f"  └───────────────────────────────────────┴────────┘")


# ===================================================
# 4. ملخص شامل
# ===================================================
print("\n" + "=" * 60)
print("  ملخص نتائج التقييم")
print("=" * 60)

print(f"""
  ┌─────────────────────────────────────────────────────────┐
  │  TF-IDF (Content-Based)                                 │
  │    Precision@5    : {avg_precision:.3f}                              │
  │    Diversity      : {diversity:.3f}                              │
  │    Coverage       : {coverage:.3f}                              │
  ├─────────────────────────────────────────────────────────┤""")

if SURPRISE_AVAILABLE:
    print(f"""  │  SVD (Collaborative Filtering)                          │
  │    RMSE           : {rmse_mean:.4f} ± {rmse_std:.4f}                  │
  │    MAE            : {mae_mean:.4f} ± {mae_std:.4f}                  │
  │    Hit Rate@5     : {hr5:.3f}                              │
  │    NDCG@5         : {ndcg5:.3f}                              │
  ├─────────────────────────────────────────────────────────┤""")

print(f"""  │  تفسير النتائج                                          │
  │  • RMSE < 1.0 → دقة تنبؤ جيدة                          │
  │  • Precision > 0.5 → نصف التوصيات ملائمة على الأقل     │
  │  • Diversity > 0.5 → تنوع معقول في التوصيات            │
  │  • Coverage > 0.7 → النظام يغطي معظم الأماكن           │
  └─────────────────────────────────────────────────────────┘
""")

print("  [✓] اكتمل التقييم بنجاح!")