from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('search/', views.search, name='search'),
    path('search-page/', views.search_page, name='search_page'),

    path("register/", views.register, name="register"),
    path("login/", views.login_view, name="login"),
    path("logout/", views.logout_view, name="logout"),

    # إضافة للمفضلة (يستخدم اسم الوجهة)
    path("favorite/add/<str:destination>/", views.add_favorite, name="add_favorite"),
    path("favorite/remove/<int:place_id>/", views.remove_favorite, name="remove_favorite"), 
    path("favorites/", views.favorites_page, name="favorites"),

    # تفاصيل المكان والتقييم (يستخدم ID المكان)
    path("place/<int:place_id>/", views.place_detail, name="place_detail"),
    path("place/rate/<int:place_id>/", views.rate_place, name="rate_place"),

    path("profile/", views.profile_view, name="profile"),
]
