import os
import pandas as pd
import django

# إعداد بيئة Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'proj.settings')
django.setup()

from recommendation.models import Place

def import_csv_to_db(file_path):
    print(f"--- Starting Clean Import from {file_path} ---")
    
    try:
        # قراءة الملف وتجاوز السطر الأول إذا كان يحتوي على العناوين
        df = pd.read_csv(file_path)
        
        # إعادة تسمية الأعمدة لضمان التوافق
        expected_columns = [
            'Destination', 'City', 'Country', 'Continent', 'Category', 
            'Rating', 'Interests', 'Season', 'Budget', 'Family', 
            'Features', 'Lat', 'Long', 'Desc_EN', 'Desc_AR'
        ]
        
        if len(df.columns) == len(expected_columns):
            df.columns = expected_columns
        else:
            print(f"Note: CSV has {len(df.columns)} columns. Adjusting...")
            # في حال كان الملف بدون هيدر أصلاً، سنعيد قراءته
            df = pd.read_csv(file_path, header=None)
            df.columns = expected_columns

    except Exception as e:
        print(f"Critical Error reading CSV: {e}")
        return

    # خيار: مسح البيانات القديمة لضمان استيراد نظيف (اختياري، لكنه يحل مشكلة الـ 0 places)
    # print("Cleaning old data...")
    # Place.objects.all().delete()

    count = 0
    for index, row in df.iterrows():
        try:
            name = str(row['Destination']).strip()
            if name.lower() == 'destination' or pd.isna(row['Destination']):
                continue
                
            # معالجة التقييم
            try:
                rating_val = float(row['Rating'])
            except:
                rating_val = 0.0

            # استخدام update_or_create لضمان التحديث إذا وجد الاسم
            place, created = Place.objects.update_or_create(
                name=name,
                city=str(row['City']).strip() if pd.notna(row['City']) else '',
                defaults={
                    'country': str(row['Country']).strip() if pd.notna(row['Country']) else '',
                    'continent': str(row['Continent']).strip() if pd.notna(row['Continent']) else '',
                    'category': str(row['Category']).strip() if pd.notna(row['Category']) else '',
                    'rating': rating_val,
                    'interests': str(row['Interests']).strip() if pd.notna(row['Interests']) else '',
                    'best_visit_season': str(row['Season']).strip() if pd.notna(row['Season']) else '',
                    'budget_range': str(row['Budget']).strip() if pd.notna(row['Budget']) else '',
                    'family_friendly': str(row['Family']).strip() if pd.notna(row['Family']) else '',
                    'features': str(row['Features']).strip() if pd.notna(row['Features']) else '',
                    'latitude': float(row['Lat']) if pd.notna(row['Lat']) else None,
                    'longitude': float(row['Long']) if pd.notna(row['Long']) else None,
                    'semantic_description': str(row['Desc_EN']).strip() if pd.notna(row['Desc_EN']) else '',
                    'semantic_description_ar': str(row['Desc_AR']).strip() if pd.notna(row['Desc_AR']) else '',
                    'image_url': 'https://images.unsplash.com/photo-1501785888041-af3ef285b470' 
                }
            )
            
            if created:
                print(f"Added: {name}")
                count += 1
            else:
                print(f"Updated: {name}")
                count += 1 # نحسبه أيضاً للتأكد من وجوده في القاعدة

        except Exception as e:
            print(f"Error at row {index}: {e}")

    print(f"\n--- Import Finished! Total processed: {count} ---")

if __name__ == "__main__":
    csv_path = 'enhanced_places_dataset_translated.csv'
    if os.path.exists(csv_path):
        import_csv_to_db(csv_path)
    else:
        print(f"Error: {csv_path} not found in the current directory.")
