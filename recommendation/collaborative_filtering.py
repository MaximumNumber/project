import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from .models import UserRating, Place

def build_user_item_matrix():
    """
    بناء مصفوفة المستخدمين والأماكن بناءً على التقييمات الموجودة في قاعدة البيانات.
    """
    ratings = UserRating.objects.all().values('user_id', 'place_id', 'rating')
    if not ratings:
        return pd.DataFrame()
    
    df = pd.DataFrame(list(ratings))
    # pivot لإنشاء مصفوفة: الصفوف هي المستخدمين، والأعمدة هي الأماكن
    matrix = df.pivot(index='user_id', columns='place_id', values='rating').fillna(0)
    return matrix

def get_collaborative_recommendations(user_id, top_k=10):
    """
    حساب التوصيات بناءً على تشابه المستخدمين (User-based Collaborative Filtering).
    """
    matrix = build_user_item_matrix()
    
    # إذا كانت المصفوفة فارغة أو المستخدم ليس لديه تقييمات
    if matrix.empty or user_id not in matrix.index:
        return {}

    try:
        user_vector = matrix.loc[[user_id]]
        # حساب التشابه بين المستخدم الحالي وجميع المستخدمين الآخرين
        similarities = cosine_similarity(user_vector, matrix)[0]
        
        # جلب أفضل 5 مستخدمين مشابهين (باستثناء المستخدم نفسه)
        similar_users_indices = np.argsort(similarities)[::-1][1:6]
        similar_users_ids = matrix.index[similar_users_indices]
        
        recommendations = {}
        user_rated_places = matrix.loc[user_id]
        user_rated_places = user_rated_places[user_rated_places > 0].index.tolist()

        for sim_user_id in similar_users_ids:
            sim_user_idx = matrix.index.get_loc(sim_user_id)
            sim_score = similarities[sim_user_idx]
            
            if sim_score <= 0:
                continue
                
            sim_user_ratings = matrix.loc[sim_user_id]
            # الأماكن التي قيمها المستخدم المشابه ولم يقيمها المستخدم الحالي
            new_places = sim_user_ratings[(sim_user_ratings > 0) & (~sim_user_ratings.index.isin(user_rated_places))]
            
            for place_id, rating in new_places.items():
                # حساب الوزن: التقييم مضروب في درجة التشابه
                recommendations[place_id] = recommendations.get(place_id, 0) + (rating * sim_score)

        # ترتيب النتائج تنازلياً
        sorted_recs = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_recs[:top_k])
    except Exception as e:
        print(f"Error in collaborative filtering: {e}")
        return {}
