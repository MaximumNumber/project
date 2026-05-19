from django.db import models
from django.conf import settings
from django.core.validators import MinValueValidator, MaxValueValidator

class Place(models.Model):
    id= models.AutoField(primary_key=True)
    name = models.CharField(max_length=255)
    city = models.CharField(max_length=255, blank=True)
    country = models.CharField(max_length=255, blank=True)
    continent = models.CharField(max_length=255, blank=True)
    category = models.CharField(max_length=255, blank=True)
    rating = models.FloatField(default=0.0)

    interests = models.CharField(max_length=500, blank=True)
    best_visit_season = models.CharField(max_length=255, blank=True)
    budget_range = models.CharField(max_length=100, blank=True)
    family_friendly = models.CharField(max_length=10, blank=True)
    features = models.CharField(max_length=500, blank=True)

    image_url = models.URLField(blank=True)

    # إحداثيات
    latitude = models.FloatField(null=True, blank=True)
    longitude = models.FloatField(null=True, blank=True)

    # التضمينات الدلالية
    semantic_description = models.TextField(blank=True)
    semantic_description_ar = models.TextField(blank=True)

    def __str__(self):
        return f"{self.name} ({self.city}, {self.country})"


class UserRating(models.Model):
    id= models.AutoField(primary_key=True)

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="ratings"
    )
    place = models.ForeignKey(
        Place,
        on_delete=models.CASCADE,
        related_name="user_ratings"
    )
    rating = models.FloatField(
        validators=[MinValueValidator(1.0), MaxValueValidator(5.0)]
    )  # يدعم أنصاف الدرجات
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ("user", "place")
        ordering = ["-created_at"]

    def __str__(self):
        return f"{self.user} → {self.place.name}: {self.rating}"


class Favorite(models.Model):
    id= models.AutoField(primary_key=True)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="favorites"
    )
    place = models.ForeignKey(
        Place,
        on_delete=models.CASCADE,
        related_name="favorited_by",
        null=True,      

        blank=True
    )

    added_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ("user", "place")
        ordering = ["-added_at"]

    def __str__(self):
        return f"{self.user} ♥ {self.place.name}"

class UserProfile(models.Model):
    user = models.OneToOneField(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="profile"
    )
    preferred_categories = models.CharField(max_length=500, blank=True)  # مثل "museum,beach,nature"
    last_active = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"Profile of {self.user.username}"

class Interaction(models.Model):
    INTERACTION_TYPES = [
    ('view', 'View'),
    ('search', 'Search'),
    ('favorite', 'Favorite'),
    ]
    user = models.ForeignKey(
    settings.AUTH_USER_MODEL,
    on_delete=models.CASCADE,
    related_name="interactions"
    )
    place = models.ForeignKey(
    Place,
    on_delete=models.CASCADE,
    related_name="interactions",
    null=True, blank=True
    )
    interaction_type = models.CharField(max_length=20, choices=INTERACTION_TYPES)
    search_query = models.CharField(max_length=255, blank=True)  # لحفظ كلمة البحث
    created_at = models.DateTimeField(auto_now_add=True)