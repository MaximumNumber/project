import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from .models import UserRating, Place

def build_user_item_matrix_svd():
    """
    بناء مصفوفة المستخدمين والأماكن بناءً على التقييمات الموجودة في قاعدة البيانات.
    تستخدم NaN للقيم المفقودة لتسهيل معالجة SVD.
    """
    ratings = UserRating.objects.all().values('user_id', 'place_id', 'rating')
    if not ratings:
        return pd.DataFrame()

    df = pd.DataFrame(list(ratings))
    # pivot لإنشاء مصفوفة: الصفوف هي المستخدمين، والأعمدة هي الأماكن
    # نستخدم NaN للقيم المفقودة هنا
    matrix = df.pivot(index='user_id', columns='place_id', values='rating')
    return matrix

def get_svd_recommendations(user_id, n_components=20, top_k=10):
    """
    حساب التوصيات باستخدام Truncated SVD.
    """
    matrix = build_user_item_matrix_svd()

    if matrix.empty or user_id not in matrix.index:
        return {}

    try:
        # ملء القيم المفقودة بالمتوسط لكل مستخدم قبل SVD
        # أو يمكن استخدام 0 إذا كانت التقييمات تبدأ من 1
        # هنا سنستخدم 0 لتبسيط المثال، ولكن المتوسط قد يكون أفضل في بعض الحالات
        # matrix_filled = matrix.fillna(matrix.mean(axis=0))
        matrix_filled = matrix.fillna(0) # ملء NaN بالصفر

        # تطبيق Truncated SVD
        # n_components هو عدد الميزات المخفية (latent features)
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        matrix_svd = svd.fit_transform(matrix_filled)

        # إعادة بناء المصفوفة الأصلية من SVD لتقدير التقييمات المفقودة
        # هذه هي المصفوفة التي تحتوي على التقييمات المتوقعة
        predicted_ratings = pd.DataFrame(svd.inverse_transform(matrix_svd), 
                                         columns=matrix.columns, 
                                         index=matrix.index)

        # جلب التقييمات المتوقعة للمستخدم الحالي
        user_predicted_ratings = predicted_ratings.loc[user_id]

        # إزالة الأماكن التي قيمها المستخدم بالفعل
        user_rated_places = matrix.loc[user_id]
        user_rated_places = user_rated_places[user_rated_places.notna()].index.tolist()

        unrated_places_predictions = user_predicted_ratings.drop(user_rated_places, errors='ignore') # إضافة errors='ignore' للتعامل مع الأماكن غير الموجودة

        # ترتيب التوصيات تنازلياً
        sorted_recs = unrated_places_predictions.sort_values(ascending=False)

        # تحويل التقييمات المتوقعة إلى مقياس من 0-100 أو 0-5 إذا لزم الأمر
        # هنا سنعيدها إلى مقياس 0-5 ليتناسب مع الكود الهجين الحالي
        # القيم المتوقعة من SVD قد تكون سالبة أو خارج نطاق 1-5، لذا نحتاج لتطبيعها
        # أبسط طريقة هي قص القيم وتطبيعها
        max_rating = 5.0 # افتراض أن أقصى تقييم هو 5
        min_rating = 1.0 # افتراض أن أدنى تقييم هو 1

        # قص القيم لتكون ضمن النطاق المتوقع
        sorted_recs = sorted_recs.apply(lambda x: max(min_rating, min(max_rating, x)))

        # إرجاع أفضل k توصيات كقاموس
        return sorted_recs.head(top_k).to_dict()

    except Exception as e:
        print(f"Error in SVD collaborative filtering: {e}")
        return {}
