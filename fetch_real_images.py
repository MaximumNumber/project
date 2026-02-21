import os
# import django
import time
import random
from duckduckgo_search import DDGS

# إعداد بيئة Django (تأكد من أن هذا الملف في مجلد المشروع الرئيسي)
# os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'proj.settings')
# django.setup()
# from recommendation.models import Place

def get_real_image_url(query):
    """
    جلب رابط صورة حقيقية ومختلفة باستخدام DuckDuckGo Search
    """
    try:
        # إضافة كلمات مفتاحية لتحسين النتائج
        search_query = f"{query} landmark tourism photo"
        
        with DDGS() as ddgs:
            # جلب نتائج الصور
            results = ddgs.images(
                keywords=search_query,
                region="wt-wt",
                safesearch="moderate",
                max_results=10
            )
            
            if results:
                # اختيار صورة عشوائية من أول 5 نتائج لضمان التنوع في كل مرة تشغل فيها السكريبت
                # أو يمكنك دائماً أخذ النتيجة الأولى results[0]['image']
                choice = random.choice(results[:5])
                return choice['image']
                
    except Exception as e:
        print(f"Error fetching image for {query}: {e}")
    
    # صورة افتراضية في حال الفشل (تأكد من تغيير الـ sig لضمان عدم التكرار)
    return f"https://images.unsplash.com/photo-1501785888041-af3ef285b470?auto=format&fit=crop&w=1000&q=80&sig={random.randint(1, 1000)}"

def update_all_images():
    # ملاحظة: قم بإلغاء التعليق عن الأسطر التالية عند التشغيل داخل مشروع Django
    # places = Place.objects.all()
    
    # محاكاة للبيانات للتوضيح
    class MockPlace:
        def __init__(self, name, city):
            self.name = name
            self.city = city
            self.image_url = ""
        def save(self):
            print(f"   Saved to DB: {self.image_url}")

    places = [
        MockPlace("Burj Khalifa", "Dubai"),
        MockPlace("Pyramids", "Giza"),
        MockPlace("Eiffel Tower", "Paris"),
        MockPlace("Great Wall", "China"),
        MockPlace("Petra", "Jordan")
    ]
    
    total = len(places)
    print(f"--- Starting Image Update for {total} places ---")
    
    for i, place in enumerate(places):
        print(f"[{i+1}/{total}] Fetching image for: {place.name} ({place.city})...")
        
        query = f"{place.name} {place.city}"
        new_image_url = get_real_image_url(query)
        
        place.image_url = new_image_url
        place.save()
        
        # تأخير بسيط لتجنب الحظر
        time.sleep(2)

    print(f"\n--- Finished! Updated {total} places with real images. ---")

if __name__ == "__main__":
    update_all_images()
