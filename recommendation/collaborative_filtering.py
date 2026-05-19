import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from .models import UserRating, Place

# ----------------------------------------------------------------------------
# دالة بناء مصفوفة المستخدم-العنصر (User-Item Matrix) مع Mean Centering
# ----------------------------------------------------------------------------
def build_user_item_matrix_svd():
    """
    بناء مصفوفة المستخدمين والأماكن بناءً على التقييمات الموجودة في قاعدة البيانات.
    تطبق Mean Centering (طرح متوسط تقييمات المستخدم) لمعالجة مشكلة الندرة.
    """
    ratings = UserRating.objects.all().values('user_id', 'place_id', 'rating')
    if not ratings: return pd.DataFrame()

    df = pd.DataFrame(list(ratings))
    matrix = df.pivot(index='user_id', columns='place_id', values='rating')

    # تطبيق Mean Centering:
    # 1. حساب متوسط تقييمات كل مستخدم
    user_means = matrix.mean(axis=1)
    # 2. طرح المتوسط من التقييمات الموجودة (القيم غير الفارغة)
    matrix_centered = matrix.subtract(user_means, axis=0)
    # 3. ملء القيم المفقودة (NaN) بالصفر بعد طرح المتوسط
    matrix_filled = matrix_centered.fillna(0)

    return matrix_filled, user_means, matrix.index.tolist(), matrix.columns.tolist()

# ----------------------------------------------------------------------------
# دالة الحصول على التوصيات باستخدام SVD (Singular Value Decomposition)
# ----------------------------------------------------------------------------
def get_svd_recommendations(user_id, top_k=10):
    """
    حساب التوصيات بناءً على SVD للمستخدم المحدد.
    تستخدم Mean Centering وتتعامل مع المستخدمين الجدد.
    """
    try:
        matrix_filled, user_means, user_ids_list, place_ids_list = build_user_item_matrix_svd()

        # إذا كانت المصفوفة فارغة أو المستخدم ليس لديه تقييمات
        if matrix_filled.empty or user_id not in user_ids_list:
            # للمستخدمين الجدد أو الذين ليس لديهم تقييمات، نعود لتوصيات المحتوى
            # أو يمكن هنا استدعاء دالة توصية عامة (مثل الأكثر تقييماً)
            return {}

        # تطبيق TruncatedSVD
        # n_components: عدد الميزات المخفية. يمكن تعديلها لتحسين الدقة/الأداء.
        # قيمة 20-50 عادة ما تكون جيدة.
        svd = TruncatedSVD(n_components=min(50, matrix_filled.shape[1] - 1))
        matrix_svd = svd.fit_transform(matrix_filled)

        # إعادة بناء المصفوفة الأصلية لتقدير التقييمات المفقودة
        # (إضافة المتوسط مرة أخرى للحصول على التقييمات المتوقعة الأصلية)
        predicted_ratings_centered = pd.DataFrame(svd.inverse_transform(matrix_svd), 
                                                  columns=place_ids_list, 
                                                  index=user_ids_list)
        
        # إضافة متوسط المستخدم مرة أخرى للحصول على التقييمات المتوقعة الأصلية
        predicted_ratings = predicted_ratings_centered.add(user_means, axis=0)

        # جلب التقييمات المتوقعة للمستخدم الحالي
        user_predicted_ratings = predicted_ratings.loc[user_id]

        # استبعاد الأماكن التي قيمها المستخدم بالفعل
        user_rated_places = UserRating.objects.filter(user_id=user_id).values_list('place_id', flat=True)
        unrated_places_predictions = user_predicted_ratings.drop(list(user_rated_places), errors='ignore')

        # تطبيع القيم المتوقعة لتكون بين 1 و 5
        min_rating, max_rating = 1.0, 5.0
        unrated_places_predictions = unrated_places_predictions.apply(lambda x: max(min_rating, min(max_rating, x)))

        # ترتيب الأماكن تنازلياً حسب التقييم المتوقع
        sorted_recs = unrated_places_predictions.sort_values(ascending=False)

        # إرجاع أفضل k توصية كقاموس (place_id: predicted_rating)
        return {int(k): float(v) for k, v in sorted_recs.head(top_k).items()}

    except Exception as e:
        # معالجة أخطاء محددة لسهولة التصحيح
        print(f"Error in SVD collaborative filtering: {e}")
        # في حالة وجود خطأ، يمكن إرجاع توصيات عامة أو لا شيء
        return {}
