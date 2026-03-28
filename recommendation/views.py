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
# مسارات البيانات (للاستخدامات الأخرى إن وجدت)
# ----------------------------
APP_DIR = Path(__file__).resolve().parent
BASE_DIR = APP_DIR.parent

# ----------------------------
# كشف اللغة ومعالجة النصوص العربية
# ----------------------------
def is_arabic_text(text: str) -> bool:
    if not text: return False
    for ch in text:
        if "\u0600" <= ch <= "\u06FF": return True
    return False

def normalize_arabic(text):
    """تبسيط النص العربي لزيادة مرونة البحث"""
    if not text: return ""
    text = text.strip().lower()
    # توحيد الحروف المتشابهة
    text = text.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا").replace("ة", "ه").replace("ى", "ي")
    # إزالة ال التعريف في بداية الكلمات لزيادة مرونة المطابقة
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
def get_content_recommendations(query, is_ar=True, top_k=20):
    clean_query = query.strip().lower()
    norm_query = normalize_arabic(clean_query) if is_ar else clean_query
    query_terms = norm_query.split()
    
    # البحث في قاعدة البيانات فقط (التي قام المستخدم بتنظيفها)
    query_filter = Q()
    for term in query_terms:
        if len(term) > 1:
            query_filter |= (Q(name__icontains=term) | Q(city__icontains=term) | Q(category__icontains=term))
            if is_ar:
                query_filter |= Q(semantic_description_ar__icontains=term)
    
    # محاولة إضافية للبحث بالكلمة الأصلية لضمان عدم ضياع النتائج
    query_filter |= Q(name__icontains=clean_query)
    
    places = Place.objects.filter(query_filter).distinct()
    
    results = []
    for p in places:
        p_name = p.name.lower()
        p_name_norm = normalize_arabic(p.name) if is_ar else p_name
        
        # حساب النسبة المئوية بطريقة مرنة
        score = 0.0
        # 1. تطابق تام أو شبه تام في الاسم (100%)
        if clean_query in p_name or p_name in clean_query or norm_query in p_name_norm:
            score = 100.0
        else:
            # 2. تطابق جزئي بناءً على الكلمات المشتركة في الاسم والوصف
            p_full_text = f"{p.name} {p.city} {p.category} {p.semantic_description_ar if is_ar else p.semantic_description}".lower()
            p_full_norm = normalize_arabic(p_full_text) if is_ar else p_full_text
            
            matches = sum(1 for term in query_terms if term in p_full_norm)
            score = (matches / len(query_terms)) * 100 if query_terms else 0
            
        results.append({"place": p, "content_score": score})
            
    results.sort(key=lambda x: (x["content_score"], x["place"].rating), reverse=True)
    return results[:top_k]

# ----------------------------
# صفحة البحث والنتائج الهجينة
# ----------------------------
def search(request):
    query = request.GET.get("query", "").strip()
    if not query:
        return render(request, "recommendation/results.html", {"message": "يرجى كتابة كلمة للبحث"})

    is_ar = is_arabic_text(query)
    
    # جلب نتائج قاعدة البيانات فقط (تم إلغاء ملفات pkl لضمان نظافة البيانات)
    content_results = get_content_recommendations(query, is_ar=is_ar)
    
    collab_scores = {}
    if request.user.is_authenticated:
        collab_scores = get_collaborative_recommendations(request.user.id)

    hybrid_results = []
    for res in content_results:
        place = res["place"]
        c_score = res["content_score"]
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
        return render(request, "recommendation/results.html", {"message": "لا توجد نتائج مطابقة في قاعدة البيانات، جرب كلمات أخرى", "query": query})

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
    similar_places = Place.objects.filter(
        Q(category=place.category) | Q(city=place.city)
    ).exclude(id=place.id).order_by('-rating')[:2]
    
    context = {
        "place": place,
        "similar_places": similar_places,
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
