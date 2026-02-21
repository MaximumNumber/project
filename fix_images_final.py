import os
import django
import time
import random
from duckduckgo_search import DDGS

# 1. إعداد بيئة Django
# تأكد من وضع هذا الملف في المجلد الرئيسي للمشروع (بجانب manage.py)
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'proj.settings')
try:
    django.setup()
    from recommendation.models import Place
    print("✅ Django environment setup successfully.")
except Exception as e:
    print(f"❌ Error setting up Django: {e}")
    print("Make sure this script is in the root folder of your project.")
    exit()

def get_real_image_url(query):
    """
    جلب رابط صورة حقيقية ومختلفة باستخدام DuckDuckGo Search
    """
    try:
        search_query = f"{query} landmark tourism photo"
        with DDGS() as ddgs:
            results = ddgs.images(
                keywords=search_query,
                region="wt-wt",
                safesearch="moderate",
                max_results=5
            )
            if results:
                # نختار النتيجة الأولى لضمان الدقة، أو عشوائي من أول 3 للتنوع
                return results[0]['image']
    except Exception as e:
        print(f"   ⚠️ Error fetching for {query}: {e}")
    
    # حل احتياطي: Unsplash مع رقم عشوائي فريد جداً لكسر التخزين المؤقت (Cache)
    return f"https://images.unsplash.com/photo-1501785888041-af3ef285b470?auto=format&fit=crop&w=1000&q=80&sig={random.randint(1, 999999)}"

def update_images():
    places = Place.objects.all()
    total = places.count()
    
    if total == 0:
        print("❌ No places found in the database. Please make sure your database is populated.")
        return

    print(f"--- Starting Image Update for {total} places ---")
    
    updated_count = 0
    for i, place in enumerate(places):
        print(f"[{i+1}/{total}] Updating: {place.name} ({place.city})...")
        
        # جلب الصورة الجديدة
        query = f"{place.name} {place.city}"
        new_url = get_real_image_url(query)
        
        # التحديث والحفظ
        place.image_url = new_url
        place.save()
        
        print(f"   ✅ New URL: {new_url[:70]}...")
        updated_count += 1
        
        # تأخير لتجنب الحظر
        time.sleep(1.5)

    print(f"\n--- Finished! Updated {updated_count} places. ---")
    print("🚀 Please refresh your browser. If images are still the same, try clearing browser cache (Ctrl+F5).")

if __name__ == "__main__":
    update_images()
