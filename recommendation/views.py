# recommendation/views.py
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity
from django.shortcuts import render, redirect, get_object_or_404
from django.conf import settings
from pathlib import Path
from functools import lru_cache
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.contrib import messages
from django.http import HttpResponse
from django.db.models import Q
from django.views.decorators.cache import never_cache
from django.views.decorators.vary import vary_on_cookie
from .models import Favorite, UserRating, Place
from .forms import RegisterForm
from .utils import get_weather, arabic_query_expand
from .collaborative_filtering import get_svd_recommendations as get_collaborative_recommendations_svd

# ----------------------------

ARABIC_TO_ENGLISH_DICT = {
    "الاهرامات": "pyramids",
    "اهرامات": "pyramids",
    "هرم": "pyramid",
    "متحف": "museum",
    "متاحف": "museums",
    "تاريخي": "historical",
    "تاريخ": "history",
    "شاطئ": "beach",
    "شواطئ": "beaches",
    "بحر": "sea",
    "جبل": "mountain",
    "جبال": "mountains",
    "حديقة": "park",
    "حدائق": "parks",
    "مطعم": "restaurant",
    "مطاعم": "restaurants",
    "فندق": "hotel",
    "فنادق": "hotels",
    "تسوق": "shopping",
    "مول": "mall",
    "طبيعة": "nature",
    "غوص": "diving",
    "سفاري": "safari",
    "عائلة": "family",
    "اطفال": "kids",
    "رخيص": "budget",
    "غالي": "luxury",
    "صيف": "summer",
    "شتاء": "winter",
    "اثار": "monuments",
    "قلعة": "castle",
    "برج": "tower",
    "مسجد": "mosque",
    "كنيسة": "church",
    "معبد": "temple",
}

def translate_arabic_query(query):
    """ترجمة الكلمات العربية الشائعة في الاستعلام إلى الإنجليزية لزيادة فرص التطابق"""
    if not query: return ""
    words = query.split()
    translated_words = []
    for w in words:
        translated_words.append(w)
        clean_w = w
        if w.startswith("ال") and len(w) > 3:
            clean_w = w[2:]
        
        if w in ARABIC_TO_ENGLISH_DICT:
            translated_words.append(ARABIC_TO_ENGLISH_DICT[w])
        elif clean_w in ARABIC_TO_ENGLISH_DICT:
            translated_words.append(ARABIC_TO_ENGLISH_DICT[clean_w])
            
    return " ".join(translated_words)
def is_arabic_text(text: str) -> bool:
    if not text: return False
    for ch in text:
        if "\u0600" <= ch <= "\u06FF": return True
    return False

def normalize_arabic(text):
    """تبسيط النص العربي لزيادة مرونة البحث (إزالة الهمزات، ال التعريف، والتاء المربوطة)"""
    if not text: return ""
    text = text.strip()
    text = text.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا").replace("ة", "ه").replace("ى", "ي")
    
    words = text.split()
    normalized_words = []
    for w in words:


        if w.startswith("ال") and len(w) > 3:
            normalized_words.append(w[2:])
        else:
            normalized_words.append(w)
    return " ".join(normalized_words)

# ----------------------------

# ----------------------------
ARABIC_STOP_WORDS = [
    "في", "من", "إلى", "على", "عن", "مع", "هذا", "هذه", "ذلك", "تلك",
    "التي", "الذي", "الذين", "اللواتي", "هو", "هي", "هم", "هن", "أنا",
    "أنت", "أنتم", "نحن", "كان", "كانت", "يكون", "تكون", "قد", "لقد",
    "ما", "لا", "لم", "لن", "إن", "أن", "كما", "أو", "و", "ثم", "حتى",
    "بعد", "قبل", "عند", "حين", "إذا", "لو", "كل", "بعض", "غير", "بين",
    "وهو", "وهي", "وهم", "فهو", "فهي", "فهم", "منه", "منها", "منهم",
    "عليه", "عليها", "عليهم", "به", "بها", "بهم", "له", "لها", "لهم",
    "فيه", "فيها", "فيهم", "إنه", "إنها", "إنهم", "أنه", "أنها", "أنهم",
    "كانوا", "يكونوا", "قال", "قالت", "يقول", "تقول", "جاء", "جاءت",
]

# ----------------------------

# ----------------------------
tfidf_vectorizer = None
places_tfidf_matrix = None
place_ids_in_tfidf = []

def initialize_tfidf_model():
    """
    بناء نموذج TF-IDF من جميع الأماكن في قاعدة البيانات.
    يُستدعى عند الحاجة الأولى فقط (lazy)، أو عند إضافة أماكن جديدة.

    للإنتاج: استدع هذه الدالة مرة واحدة من AppConfig.ready() في apps.py:

        # recommendation/apps.py
        from django.apps import AppConfig

        class RecommendationConfig(AppConfig):
            name = 'recommendation'

            def ready(self):
                # تأكد أن هذا لا يعمل أثناء التهجيرات أو الاختبارات
                import sys
                if 'migrate' not in sys.argv and 'test' not in sys.argv:
                    from .views import initialize_tfidf_model
                    initialize_tfidf_model()
    """
    global tfidf_vectorizer, places_tfidf_matrix, place_ids_in_tfidf

    places = Place.objects.all()
    if not places.exists():
        return

    place_texts = []
    place_ids_in_tfidf = []
    for p in places:
       
        full_text = f"{p.name} {p.city} {p.country} {p.category} {p.interests} {p.features} {p.semantic_description_ar or ''} {p.semantic_description or ''}"
        
        normalized_text = normalize_arabic(full_text) if is_arabic_text(full_text) else full_text.lower()
        
        if p.semantic_description_ar:
            translated_desc = translate_arabic_query(normalize_arabic(p.semantic_description_ar))
            normalized_text += f" {translated_desc}"
            
        place_texts.append(normalized_text)
        place_ids_in_tfidf.append(p.id)

    tfidf_vectorizer = TfidfVectorizer(
    stop_words=list(ARABIC_STOP_WORDS) + list(ENGLISH_STOP_WORDS)
)
    places_tfidf_matrix = tfidf_vectorizer.fit_transform(place_texts)



def get_content_recommendations(query=None, source_place_id=None, is_ar=True, top_k=20):
    if query:
        clean_query = query.strip()
        if is_ar:
            # تطبيع الاستعلام العربي ثم ترجمة الكلمات الشائعة فيه إلى الإنجليزية
            normalized_q = normalize_arabic(clean_query)
            query_text = translate_arabic_query(normalized_q)
        else:
            query_text = clean_query.lower()
    elif source_place_id:
        try:
            source_place = Place.objects.get(id=source_place_id)
            raw_text = f"{source_place.name} {source_place.city} {source_place.country} {source_place.category} {source_place.interests} {source_place.features} {source_place.semantic_description_ar or ''} {source_place.semantic_description or ''}"
            
            if is_ar:
                normalized_q = normalize_arabic(raw_text)
                query_text = translate_arabic_query(normalized_q)
            else:
                query_text = raw_text.lower()
        except Place.DoesNotExist:
            return []
    else:
        return []

    global tfidf_vectorizer, places_tfidf_matrix, place_ids_in_tfidf

    if tfidf_vectorizer is None or places_tfidf_matrix is None:
        initialize_tfidf_model()
        if tfidf_vectorizer is None or places_tfidf_matrix is None:
            return []

    query_vector = tfidf_vectorizer.transform([query_text])

    cosine_similarities = cosine_similarity(query_vector, places_tfidf_matrix).flatten()

    valid_indices = np.where(cosine_similarities > 0)[0]
    if len(valid_indices) == 0 and query:
        return [] 

    place_scores = []
    for i in valid_indices:
        place_id = place_ids_in_tfidf[i]
        optimistic_score = cosine_similarities[i] * 100

        place_scores.append({
            'place_id': place_id,
            'content_score': optimistic_score
        })

    if source_place_id:
        place_objects = {p.id: p for p in Place.objects.filter(id__in=place_ids_in_tfidf).exclude(id=source_place_id)}
    else:
        place_objects = {p.id: p for p in Place.objects.filter(id__in=place_ids_in_tfidf)}

    results = []
    for ps in place_scores:
        place = place_objects.get(ps['place_id'])
        if place:
            results.append({"place": place, "content_score": ps["content_score"]})

    results.sort(key=lambda x: (x["content_score"], x["place"].rating), reverse=True)
    return results[:top_k]


# ----------------------------
# صفحة البحث والنتائج الهجينة
# ----------------------------
def get_hybrid_recommendations_for_place(current_place_id, user_id=None, top_k=5):
    current_place = get_object_or_404(Place, id=current_place_id)
    is_ar = is_arabic_text(current_place.semantic_description_ar or current_place.name)

    content_results = get_content_recommendations(source_place_id=current_place_id, is_ar=is_ar, top_k=top_k * 2)

    collab_scores = {}
    if user_id:
        collab_scores = get_collaborative_recommendations_svd(user_id, top_k=top_k * 2)

    content_weight = 0.7  # الوزن الافتراضي للمحتوى
    collab_weight = 0.3   # الوزن الافتراضي للتعاوني

    if user_id:
        user_ratings_count = UserRating.objects.filter(user_id=user_id).count()
        if user_ratings_count < 5:    # مستخدم جديد جداً
            content_weight = 1.0
            collab_weight = 0.0
        elif user_ratings_count < 15: # مستخدم جديد
            content_weight = 0.8
            collab_weight = 0.2
        else:                          # مستخدم لديه خبرة كافية
            content_weight = 0.5
            collab_weight = 0.5

    hybrid_results = []
    for res in content_results:
        place = res["place"]
        c_score = res["content_score"]

        # دمج التصفية التعاونية (SVD)
        raw_coll_score = collab_scores.get(place.id, 0)
        # تحويل تقييم SVD (1-5) إلى مقياس 0-100
        coll_score = (raw_coll_score / 5.0) * 100 if raw_coll_score > 0 else c_score

        final_score = (c_score * content_weight) + (coll_score * collab_weight)

        hybrid_results.append({
            "place": place,
            "score": round(final_score, 2),
        })

    hybrid_results.sort(key=lambda x: x["score"], reverse=True)

    final_recommendations = []
    for item in hybrid_results:
        final_recommendations.append(item["place"])
        if len(final_recommendations) >= top_k:
            break

    return final_recommendations


def search(request):
    query = request.GET.get("query", "").strip()
    if not query:
        return render(request, "recommendation/results.html", {"message": "يرجى كتابة كلمة للبحث"})

    is_ar = is_arabic_text(query)
    content_results = get_content_recommendations(query, is_ar=is_ar, top_k=20)

    collab_scores = {}
    content_weight = 0.7
    collab_weight = 0.3

    if request.user.is_authenticated:
        user_id = request.user.id
        user_ratings_count = UserRating.objects.filter(user_id=user_id).count()

        if user_ratings_count < 5:
            content_weight = 1.0
            collab_weight = 0.0
        elif user_ratings_count < 15:
            content_weight = 0.8
            collab_weight = 0.2
        else:
            content_weight = 0.5
            collab_weight = 0.5

        collab_scores = get_collaborative_recommendations_svd(user_id, top_k=50)

        # *** الإضافة المهمة ***
        # أضف الأماكن من الـ Collaborative التي لم تظهر في نتائج البحث
        collab_scores = get_collaborative_recommendations_svd(user_id, top_k=50)

# أضف أماكن الـ collab التي لم تظهر في نتائج البحث
        if collab_weight > 0 and collab_scores:
            content_place_ids = {res["place"].id for res in content_results}
            extra_ids = [pid for pid in collab_scores if pid not in content_place_ids]
            if extra_ids:
                extra_places = Place.objects.filter(id__in=extra_ids[:15])
                for p in extra_places:
                    content_results.append({
                    "place": p,
                    "content_score": 0
            })

    # باقي الكود كما هو...
    hybrid_results = []
    for res in content_results:
        place = res["place"]
        c_score = res["content_score"]
        raw_coll_score = collab_scores.get(place.id, 0)
        coll_score = (raw_coll_score / 5.0) * 100 if raw_coll_score > 0 else c_score
        final_score = (c_score * content_weight) + (coll_score * collab_weight)

        hybrid_results.append({
            "name": place.name,
            "city": place.city,
            "country": place.country,
            "category": place.category,
            "rating": place.rating,
            "description": place.semantic_description_ar if is_ar else place.semantic_description,
            "score": round(final_score, 2),
            "place_id": place.id,
            "image_url": place.image_url,
        })



    if not hybrid_results:
        return render(request, "recommendation/results.html", {
            "message": "لا توجد نتائج مطابقة في قاعدة البيانات، جرب كلمات أخرى",
            "query": query
        })

    # ترتيب النتائج حسب السكور النهائي
    hybrid_results.sort(key=lambda x: x["score"], reverse=True)

    results = hybrid_results[:10]

    for item in results:
        if item["city"]:
            item["weather"] = get_weather(item["city"])

    return render(request, "recommendation/results.html", {"results": results, "query": query})


# ----------------------------
# الدوال الأساسية الأخرى
# ----------------------------
@never_cache 
@vary_on_cookie
def home(request):
    if request.user.is_authenticated:
        user_id = request.user.id
        user_ratings_count = UserRating.objects.filter(user_id=user_id).count()

        if user_ratings_count >= 3:
            # مستخدم لديه تقييمات - استخدم SVD مباشرة
            collab_scores = get_collaborative_recommendations_svd(user_id, top_k=20)
            print(collab_scores)
            if collab_scores:
                place_ids = list(collab_scores.keys())
                places = Place.objects.filter(id__in=place_ids)
                # رتب حسب SVD score
                places_dict = {p.id: p for p in places}
                top_places = sorted(
                    [places_dict[pid] for pid in place_ids if pid in places_dict],
                    key=lambda p: collab_scores[p.id],
                    reverse=True
                )[:5]
            else:
                top_places = list(Place.objects.all().order_by('-rating')[:5])
        else:
            # مستخدم جديد - أعلى تقييماً
            top_places = list(Place.objects.all().order_by('-rating')[:5])
    else:
        top_places = list(Place.objects.all().order_by('-rating')[:5])

    return render(request, "recommendation/search.html", {"top_places": top_places})

def search_page(request):
    return render(request, "recommendation/search.html")

def register(request):
    if request.method == "POST":
        form = RegisterForm(request.POST)
        if form.is_valid():
            user = form.save(commit=False)
            user.set_password(form.cleaned_data["password"])
            user.save()
            messages.success(request, "تم إنشاء الحساب بنجاح!")
            return redirect("login")
    else:
        form = RegisterForm()
    return render(request, "recommendation/signup.html", {"form": form})

def login_view(request):
    error = None
    if request.method == "POST":
        input_value = request.POST.get("username")
        password = request.POST.get("password")
        username = input_value
        if "@" in input_value:
            try:
                user_obj = User.objects.get(email=input_value)
                username = user_obj.username
            except User.DoesNotExist:
                pass
        user = authenticate(username=username, password=password)
        if user is not None:
            login(request, user)
            return redirect("home")
        else:
            error = "البيانات غير صحيحة"
    return render(request, "recommendation/login.html", {"error": error})

def logout_view(request):
    logout(request)
    return redirect("home")

def place_detail(request, place_id):
    place = get_object_or_404(Place, id=place_id)
    user_id = request.user.id if request.user.is_authenticated else None
    user_rating = None
    if request.user.is_authenticated:
        try:
            user_rating = UserRating.objects.get(user=request.user, place=place)
        except UserRating.DoesNotExist:
            pass
    similar_places = get_hybrid_recommendations_for_place(place_id, user_id=user_id, top_k=5)

    context = {
        "place": place,
        "similar_places": similar_places,
        "user_rating": user_rating,
    }
    return render(request, "recommendation/place_detail.html", context)

def profile_view(request):
    user = request.user
    favorites = Favorite.objects.filter(user=user).select_related("place")
    ratings = UserRating.objects.filter(user=user).select_related("place")
    context = {
        "user": user,                  
        "favorites": favorites,
        "ratings": ratings,            
    }
    return render(request, "recommendation/profile.html", context) 

def add_favorite(request, destination):
    if not request.user.is_authenticated:
        messages.error(request, "يجب تسجيل الدخول أولاً")
        return redirect("login")

    place = Place.objects.filter(Q(name__iexact=destination) | Q(name__icontains=destination)).first()
    if not place:
        messages.error(request, "المكان غير موجود")
        return redirect("home")

    already_exists = Favorite.objects.filter(user=request.user, place=place).exists()
    if already_exists:
        messages.warning(request, "هذا المكان موجود بالفعل في المفضلة")
    else:
        Favorite.objects.create(user=request.user, place=place)
        messages.success(request, "تمت إضافة المكان إلى المفضلة بنجاح!")

    return redirect("place_detail", place_id=place.id)

def remove_favorite(request, place_id):
    if not request.user.is_authenticated:
        return redirect("login")
    Favorite.objects.filter(user=request.user, place_id=place_id).delete()
    return redirect("profile")

@login_required
def rate_place(request, place_id):
    if request.method == "POST":
        rating_value = request.POST.get("rating")
        if not rating_value:
            return redirect("place_detail", place_id=place_id)

        place = get_object_or_404(Place, id=place_id)

        UserRating.objects.update_or_create(
            user=request.user,
            place=place,
            defaults={'rating': float(rating_value)}
        )

        from django.db.models import Avg
        avg_rating = UserRating.objects.filter(place=place).aggregate(Avg('rating'))['rating__avg']

        if avg_rating:
            place.rating = round(avg_rating, 1)
            place.save()

        messages.success(request, "تم تسجيل تقييمك وتحديث تقييم المكان بنجاح!")
        return redirect("place_detail", place_id=place_id)
    return redirect("home")

@login_required
def favorites_page(request):
    favs = Favorite.objects.filter(user=request.user).order_by("-added_at")
    return render(request, "recommendation/favorites.html", {"favorites": favs})