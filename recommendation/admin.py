from django.contrib import admin
from django.utils.html import format_html
from .models import Place, Favorite, UserRating

@admin.register(Place)
class PlaceAdmin(admin.ModelAdmin):
    # 1. تحسين عرض القائمة الرئيسية (List View)
    list_display = ('display_image', 'name', 'city', 'category', 'rating', 'country')
    list_filter = ('category', 'city', 'country')
    search_fields = ('name', 'city', 'semantic_description_ar')
    ordering = ('-rating',)
    list_per_page = 20

    # 2. تقسيم صفحة التعديل إلى أقسام (Fieldsets) لتكون مريحة ومنظمة
    fieldsets = (
        ('المعلومات الأساسية', {
            'fields': ('name', 'category', 'rating'),
            'description': 'قم بإدخال الاسم والنوع والتقييم العام للمكان.'
        }),
        ('الموقع الجغرافي', {
            'fields': ('city', 'country'),
            'classes': ('collapse',), # يمكن طي هذا القسم لتقليل الزحمة
        }),
        ('المحتوى والوصف', {
            'fields': ('semantic_description', 'semantic_description_ar', 'features', 'interests'),
        }),
        ('الوسائط والصور', {
            'fields': ('image_url', 'preview_image'),
        }),
    )

    # 3. جعل بعض الحقول للقراءة فقط (مثل معاينة الصورة)
    readonly_fields = ('preview_image',)

    # 4. دالة لعرض صورة مصغرة في القائمة الرئيسية
    def display_image(self, obj):
        if obj.image_url:
            return format_html('<img src="{}" style="width: 50px; height: 50px; border-radius: 5px; object-fit: cover;" />', obj.image_url)
        return "لا توجد صورة"
    display_image.short_description = 'الصورة'

    # 5. دالة لعرض معاينة كبيرة للصورة داخل صفحة التعديل
    def preview_image(self, obj):
        if obj.image_url:
            return format_html('<img src="{}" style="max-width: 300px; border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);" />', obj.image_url)
        return "سيتم عرض المعاينة هنا عند إضافة رابط الصورة"
    preview_image.short_description = 'معاينة الصورة الحالية'

@admin.register(Favorite)
class FavoriteAdmin(admin.ModelAdmin):
    list_display = ('user', 'place', 'added_at')
    list_filter = ('user', 'added_at')
    search_fields = ('user__username', 'place__name')

@admin.register(UserRating)
class UserRatingAdmin(admin.ModelAdmin):
    list_display = ('user', 'place', 'rating', 'created_at')
    list_filter = ('rating', 'created_at')
    search_fields = ('user__username', 'place__name')

# تخصيص عنوان لوحة الإدارة بالكامل
admin.site.site_header = "لوحة تحكم نظام التوصية السياحي"
admin.site.site_title = "إدارة السياحة"
admin.site.index_title = "مرحباً بك في بوابة الإدارة"
