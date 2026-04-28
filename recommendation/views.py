# recommendation/views.py
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
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

from .models import Favorite, UserRating, Place
from .forms import RegisterForm
from .utils import get_weather, arabic_query_expand
from .collaborative_filtering import get_svd_recommendations as get_collaborative_recommendations_svd

# ----------------------------
# كشف اللغة ومعالجة النصوص العربية
# ----------------------------
def is_arabic_text(text: str) -> bool:
    if not text: return False
    for ch in text:
        if "\u0600" <= ch <= "\u06FF": return True
    return False

def normalize_arabic(text):
    """تبسيط النص العربي لزيادة مرونة البحث (إزالة الهمزات، ال التعريف، والتاء المربوطة)"""
    if not text: return ""
    text = text.strip().lower()
    # توحيد الحروف المتشابهة
    text = text.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا").replace("ة", "ه").replace("ى", "ي")
    # إزالة ال التعريف في بداية الكلمات
    words = text.split()
    normalized_words = []
    for w in words:
        if w.startswith("ال") and len(w) > 3:
            normalized_words.append(w[2:])
        normalized_words.append(w)
    return " ".join(set(normalized_words))

# ----------------------------
# منطق الـ Content-based Recommendation (قاعدة البيانات فقط)
# ----------------------------
tfidf_vectorizer = None
places_tfidf_matrix = None
place_ids_in_tfidf = []

def initialize_tfidf_model():
    global tfidf_vectorizer, places_tfidf_matrix, place_ids_in_tfidf
    places = Place.objects.all()
    if not places:
        return

    place_texts = []
    place_ids_in_tfidf = []
    for p in places:
        # دمج جميع النصوص ذات الصلة في سلسلة واحدة
        full_text = f"{p.name} {p.city} {p.category} {p.semantic_description_ar or ''} {p.semantic_description or ''}"
        place_texts.append(full_text)
        place_ids_in_tfidf.append(p.id)

    tfidf_vectorizer = TfidfVectorizer(stop_words='arabic' if is_arabic_text('dummy') else 'english') # افتراض أن الدالة is_arabic_text موجودة
    places_tfidf_matrix = tfidf_vectorizer.fit_transform(place_texts)

# استدعاء تهيئة النموذج عند بدء تشغيل التطبيق أو عند الحاجة
# For simplicity, we'll call it here, but in a real app, you might want to call it once on startup
initialize_tfidf_model()

def get_content_recommendations(query=None, source_place_id=None, is_ar=True, top_k=20):
    if query:
        clean_query = query.strip().lower()
        norm_query = normalize_arabic(clean_query) if is_ar else clean_query
        query_text = clean_query
    elif source_place_id:
        source_place = Place.objects.get(id=source_place_id)
        query_text = f"{source_place.name} {source_place.city} {source_place.category} {source_place.semantic_description_ar or ''} {source_place.semantic_description or ''}"
    else:
        return []

    
    global tfidf_vectorizer, places_tfidf_matrix, place_ids_in_tfidf
    if tfidf_vectorizer is None or places_tfidf_matrix is None:
        initialize_tfidf_model()
        if tfidf_vectorizer is None or places_tfidf_matrix is None:
            return [] # لا توجد أماكن لإنشاء نموذج TF-IDF

    # تحويل الاستعلام أو نص المكان المصدر إلى متجه TF-IDF
    query_vector = tfidf_vectorizer.transform([query_text])

    # حساب تشابه جيب التمام بين الاستعلام وجميع الأماكن
    cosine_similarities = cosine_similarity(query_vector, places_tfidf_matrix).flatten()

    # ربط درجات التشابه بالأماكن
    place_scores = []
    for i, place_id in enumerate(place_ids_in_tfidf):
        place_scores.append({
            'place_id': place_id,
            'content_score': cosine_similarities[i] * 100 # تحويل إلى مقياس 0-100
        })

    # جلب تفاصيل الأماكن
    # جلب تفاصيل الأماكن، مع استبعاد المكان المصدر إذا كان موجوداً
    if source_place_id:
        place_objects = {p.id: p for p in Place.objects.filter(id__in=place_ids_in_tfidf).exclude(id=source_place_id)}
    else:
        place_objects = {p.id: p for p in Place.objects.filter(id__in=place_ids_in_tfidf)}

    results = []
    for ps in place_scores:
        place = place_objects.get(ps['place_id'])
        if place:
            results.append({"place": place, "content_score": ps["content_score"]})

    # ترتيب النتائج حسب السكور والتقييم
    results.sort(key=lambda x: (x["content_score"], x["place"].rating), reverse=True)
    return results[:top_k]

# ----------------------------
# صفحة البحث والنتائج الهجينة
# ----------------------------
def get_hybrid_recommendations_for_place(current_place_id, user_id=None, top_k=5):
    # جلب المكان الحالي
    current_place = get_object_or_404(Place, id=current_place_id)
    is_ar = is_arabic_text(current_place.semantic_description_ar or current_place.name)

    # 1. توصيات المحتوى (TF-IDF) بناءً على المكان الحالي
    content_results = get_content_recommendations(source_place_id=current_place_id, is_ar=is_ar, top_k=top_k*2) # جلب عدد أكبر ثم تصفية

    collab_scores = {}
    if user_id:
        # 2. توصيات الترشيح التعاوني (SVD) للمستخدم الحالي
        collab_scores = get_collaborative_recommendations_svd(user_id, top_k=top_k*2) # جلب عدد أكبر ثم تصفية

    hybrid_results = []
    for res in content_results:
        place = res["place"]
        c_score = res["content_score"]
        
        # تحديد الأوزان ديناميكياً بناءً على عدد تقييمات المستخدم
    content_weight = 0.7 # الوزن الافتراضي للمحتوى
    collab_weight = 0.3  # الوزن الافتراضي للتعاوني

    if user_id:
        user_ratings_count = UserRating.objects.filter(user_id=user_id).count()
        if user_ratings_count < 5: # مستخدم جديد جداً
            content_weight = 1.0
            collab_weight = 0.0
        elif user_ratings_count < 15: # مستخدم جديد
            content_weight = 0.8
            collab_weight = 0.2
        else: # مستخدم لديه خبرة كافية
            content_weight = 0.5
            collab_weight = 0.5

    for res in content_results:
        place = res["place"]
        c_score = res["content_score"]
        
        # دمج التصفية التعاونية (SVD)
        raw_coll_score = collab_scores.get(place.id, 0)
        # تحويل تقييم SVD (1-5) إلى مقياس 0-100
        coll_score = (raw_coll_score / 5.0) * 100 if raw_coll_score > 0 else c_score # إذا لم يكن هناك تقييم تعاوني، استخدم درجة المحتوى
        
        final_score = (c_score * content_weight) + (coll_score * collab_weight)
        
        hybrid_results.append({
            "place": place,
            "score": round(final_score, 2),
        })

    # ترتيب النتائج حسب السكور النهائي
    hybrid_results.sort(key=lambda x: x["score"], reverse=True)
    
    # إرجاع أفضل k مكان
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
    
    # جلب نتائج قاعدة البيانات فقط (تم إلغاء ملفات pkl لضمان نظافة البيانات)
    content_results = get_content_recommendations(query, is_ar=is_ar)
    
    collab_scores = {}
    if request.user.is_authenticated:
        collab_scores = get_collaborative_recommendations_svd(request.user.id)

    hybrid_results = []
    for res in content_results:
        place = res["place"]
        c_score = res["content_score"]
        
        # دمج التصفية التعاونية (Collaborative Filtering) بنسبة 30%
        raw_coll_score = collab_scores.get(place.id, 0)
        coll_score = (raw_coll_score / 5.0) * 100 if raw_coll_score > 0 else c_score
        
        final_score = (c_score * 0.7) + (coll_score * 0.3)
        
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
# الدوال الأساسية الأخرى (باقي الدوال كما هي)
# ----------------------------
def home(request):
    top_places = Place.objects.all().order_by('-rating')[:5]
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
    context = {
        "username": user.username,
        "email": user.email,
        "favorites": favorites,
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
        place = get_object_or_404(Place, id=place_id)
        UserRating.objects.update_or_create(
            user=request.user,
            place=place,
            defaults={'rating': float(rating_value)}
        )
        messages.success(request, "تم تسجيل تقييمك بنجاح!")
        return redirect("place_detail", place_id=place_id)
    return redirect("home")

@login_required
def favorites_page(request):
    favs = Favorite.objects.filter(user=request.user).order_by("-added_at")
    return render(request, "recommendation/favorites.html", {"favorites": favs})
