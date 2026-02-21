# recommendation/views.py
import os
import pickle
import numpy as np
import pandas as pd
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
from .collaborative_filtering import get_collaborative_recommendations

# ----------------------------
# مسارات البيانات
# ----------------------------
APP_DIR = Path(__file__).resolve().parent
BASE_DIR = APP_DIR.parent

DATA_AR_PATH  = BASE_DIR / "tourism_data_ar.pkl"
DATA_EN_PATH  = BASE_DIR / "tourism_data_en.pkl"

# ----------------------------
# تحميل البيانات
# ----------------------------
@lru_cache(maxsize=1)
def load_data():
    try:
        with open(DATA_AR_PATH, "rb") as f:
            data_ar = pickle.load(f)
        with open(DATA_EN_PATH, "rb") as f:
            data_en = pickle.load(f)
        return data_ar, data_en
    except Exception as e:
        print(f"Error loading pkl files: {e}")
        return [], []

data_ar, data_en = load_data()

# ----------------------------
# كشف اللغة
# ----------------------------
def is_arabic_text(text: str) -> bool:
    if not text:
        return False
    for ch in text:
        if "\u0600" <= ch <= "\u06FF":
            return True
    return False

# ----------------------------
# منطق الـ Content-based Recommendation المحسن
# ----------------------------
def get_content_recommendations(query, is_ar=True, top_k=20):
    """
    توصية مبنية على المحتوى مع دعم البحث المرن وتوسيع الكلمات.
    """
    # 1. توسيع الاستعلام (إذا كان عربياً)
    processed_query = arabic_query_expand(query) if is_ar else query
    search_terms = processed_query.lower().split()
    
    # 2. البحث في قاعدة البيانات باستخدام Q objects للبحث المرن
    query_filter = Q()
    for term in search_terms:
        query_filter |= (
            Q(name__icontains=term) | 
            Q(city__icontains=term) | 
            Q(category__icontains=term) |
            Q(semantic_description__icontains=term) |
            Q(semantic_description_ar__icontains=term) |
            Q(features__icontains=term) |
            Q(interests__icontains=term)
        )
    
    places = Place.objects.filter(query_filter).distinct()
    
    results = []
    for p in places:
        score = 0
        p_name = p.name.lower()
        p_city = p.city.lower()
        p_desc = (p.semantic_description_ar + p.semantic_description).lower()
        
        # حساب النقاط بناءً على مطابقة الكلمات
        for term in search_terms:
            if term in p_name: score += 1.0
            if term in p_city: score += 0.5
            if term in p_desc: score += 0.2
        
        results.append({
            "place": p,
            "content_score": score
        })
            
    results.sort(key=lambda x: x["content_score"], reverse=True)
    return results[:top_k]

# ----------------------------
# صفحة البحث والنتائج الهجينة
# ----------------------------
def search(request):
    query = request.GET.get("query", "").strip()
    if not query:
        return render(request, "recommendation/results.html", {"message": "يرجى كتابة كلمة للبحث"})

    is_ar = is_arabic_text(query)
    
    # 1. جلب نتائج المحتوى (Content-based)
    content_results = get_content_recommendations(query, is_ar=is_ar)
    
    # 2. جلب نتائج التصفية التعاونية (Collaborative)
    collab_scores = {}
    if request.user.is_authenticated:
        collab_scores = get_collaborative_recommendations(request.user.id)

    # 3. دمج النتائج (Hybrid)
    hybrid_results = []
    for res in content_results:
        place = res["place"]
        c_score = res["content_score"]
        coll_score = collab_scores.get(place.id, 0)
        
        # دمج النقاط: 70% محتوى + 30% تعاوني
        final_score = (c_score * 0.7) + (coll_score * 0.3)
        
        hybrid_results.append({
            "name": place.name,
            "city": place.city,
            "country": place.country,
            "category": place.category,
            "rating": place.rating,
            "description": place.semantic_description_ar if is_ar else place.semantic_description,
            "score": final_score,
            "place_id": place.id,
            "image_url": place.image_url,
        })

    # خطة احتياطية: إذا لم توجد نتائج في DB، ابحث في ملفات pkl
    if not hybrid_results:
        data = data_ar if is_ar else data_en
        search_terms = query.lower().split()
        for item in data:
            item_text = str(item).lower()
            if any(term in item_text for term in search_terms):
                hybrid_results.append({
                    "name": item.get("Destination") or item.get("name"),
                    "city": item.get("City", ""),
                    "country": item.get("Country", ""),
                    "category": item.get("Category", ""),
                    "rating": item.get("Rating", 0),
                    "description": item.get("semantic_description_ar") if is_ar else item.get("semantic_description"),
                    "score": 0.1,
                    "place_id": None,
                    "image_url": item.get("Image url", ""),
                })

    if not hybrid_results:
        return render(request, "recommendation/results.html", {"message": "لا توجد نتائج مطابقة، جرب كلمات أخرى"})

    hybrid_results.sort(key=lambda x: x["score"], reverse=True)
    results = hybrid_results[:10] # زيادة عدد النتائج المعروضة قليلاً

    for item in results:
        if item["city"]:
            item["weather"] = get_weather(item["city"])

    return render(request, "recommendation/results.html", {"results": results})

# ----------------------------
# الدوال الأساسية الأخرى
# ----------------------------
def home(request):
    # جلب أفضل 5 أماكن بناءً على التقييم من قاعدة البيانات
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
    return render(request, "recommendation/place_detail.html", {"place": place})

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
