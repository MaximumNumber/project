from django.db import models
from django.contrib.auth.models import AbstractBaseUser, BaseUserManager, PermissionsMixin


# ─────────────────────────────────────────────
# Custom User Manager
# ─────────────────────────────────────────────

class UserManager(BaseUserManager):
    def create_user(self, email, password=None, **extra_fields):
        if not email:
            raise ValueError("Email is required")
        email = self.normalize_email(email)
        user = self.model(email=email, **extra_fields)
        user.set_password(password)
        user.save(using=self._db)
        return user

    def create_superuser(self, email, password=None, **extra_fields):
        extra_fields.setdefault("is_staff", True)
        extra_fields.setdefault("is_superuser", True)
        return self.create_user(email, password, **extra_fields)


# ─────────────────────────────────────────────
# Users
# ─────────────────────────────────────────────

class User(AbstractBaseUser, PermissionsMixin):
    """
    Maps to: Users
    PK      → id (AutoField via AbstractBaseUser)
    UNIQUE  → email
    Fields  → password (handled by AbstractBaseUser), full_name,
               profile_picture, account_creation_date, last_login, status
    """
    id                    = models.AutoField(primary_key=True)
    email                 = models.EmailField(unique=True)
    # `password` is inherited from AbstractBaseUser
    full_name             = models.TextField()
    profile_picture       = models.TextField(blank=True)
    account_creation_date = models.DateTimeField(auto_now_add=True)
    last_login            = models.DateTimeField(null=True, blank=True)
    status                = models.TextField(blank=True)

    # Required by Django admin / permission system
    is_active  = models.BooleanField(default=True)
    is_staff   = models.BooleanField(default=False)

    objects = UserManager()

    USERNAME_FIELD  = "email"
    REQUIRED_FIELDS = ["full_name"]

    class Meta:
        db_table = "users"

    def __str__(self):
        return self.email


# ─────────────────────────────────────────────
# Categories
# ─────────────────────────────────────────────

class Category(models.Model):
    """
    Maps to: Categories
    PK     → id
    UNIQUE → name
    """
    id   = models.AutoField(primary_key=True)
    name = models.TextField(unique=True)

    class Meta:
        db_table = "categories"

    def __str__(self):
        return self.name


# ─────────────────────────────────────────────
# Tourist Places
# ─────────────────────────────────────────────

class TouristPlace(models.Model):
    """
    Maps to: Tourist Places
    PK     → id
    Fields → name, city, country, category, average_rating,
              description_ar, description_en, address,
              opening_hours, phone, date_added
    """
    id             = models.AutoField(primary_key=True)
    name           = models.TextField()
    city           = models.TextField(blank=True)
    country        = models.TextField(blank=True)
    category       = models.TextField(blank=True)          # denormalized text field (as in schema)
    average_rating = models.DecimalField(
                         max_digits=3, decimal_places=2,
                         null=True, blank=True
                     )
    description_ar = models.TextField(blank=True)
    description_en = models.TextField(blank=True)
    address        = models.TextField(blank=True)
    opening_hours  = models.TextField(blank=True)
    phone          = models.TextField(blank=True)
    date_added     = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "tourist_places"

    def __str__(self):
        return f"{self.name} ({self.city}, {self.country})"


# ─────────────────────────────────────────────
# Place Categories  (junction: TouristPlace ↔ Category)
# ─────────────────────────────────────────────

class PlaceCategory(models.Model):
    """
    Maps to: Place Categories
    PK  → id
    FKs → place_id → TouristPlace, category_id → Category
    Relationship: Tourist Places ──< Place Categories >── Categories
    """
    id       = models.AutoField(primary_key=True)
    place    = models.ForeignKey(
                   TouristPlace,
                   on_delete=models.CASCADE,
                   related_name="place_categories",
                   db_column="place_id"
               )
    category = models.ForeignKey(
                   Category,
                   on_delete=models.CASCADE,
                   related_name="place_categories",
                   db_column="category_id"
               )

    class Meta:
        db_table = "place_categories"

    def __str__(self):
        return f"{self.place.name} → {self.category.name}"


# ─────────────────────────────────────────────
# Reviews
# ─────────────────────────────────────────────

class Review(models.Model):
    """
    Maps to: Reviews
    PK  → id
    FKs → user_id → User, place_id → TouristPlace
    Fields → rating (smallint), comment, review_date
    Relationships:
        Users ──< Reviews (makes)
        Tourist Places ──< Reviews (has reviews)
    """
    id          = models.AutoField(primary_key=True)
    user        = models.ForeignKey(
                      User,
                      on_delete=models.CASCADE,
                      related_name="reviews",
                      db_column="user_id"
                  )
    place       = models.ForeignKey(
                      TouristPlace,
                      on_delete=models.CASCADE,
                      related_name="reviews",
                      db_column="place_id"
                  )
    rating      = models.SmallIntegerField()
    comment     = models.TextField(blank=True)
    review_date = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "reviews"

    def __str__(self):
        return f"Review by {self.user.email} on {self.place.name} ({self.rating}★)"


# ─────────────────────────────────────────────
# Favorites
# ─────────────────────────────────────────────

class Favorite(models.Model):
    """
    Maps to: Favorites
    PK  → id
    FKs → user_id → User, place_id → TouristPlace
    Fields → added_date
    Relationships:
        Users ──< Favorites (has favorites)
        Tourist Places ──< Favorites (within)
    """
    id         = models.AutoField(primary_key=True)
    user       = models.ForeignKey(
                     User,
                     on_delete=models.CASCADE,
                     related_name="favorites",
                     db_column="user_id"
                 )
    place      = models.ForeignKey(
                     TouristPlace,
                     on_delete=models.CASCADE,
                     related_name="favorited_by",
                     db_column="place_id"
                 )
    added_date = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "favorites"

    def __str__(self):
        return f"{self.user.email} ♥ {self.place.name}"


# ─────────────────────────────────────────────
# Recommendations
# ─────────────────────────────────────────────

class Recommendation(models.Model):
    """
    Maps to: Recommendations
    PK  → id
    FKs → user_id → User, place_id → TouristPlace
    Fields → recommendation_type, recommendation_score (smallint),
              recommendation_date
    Relationships:
        Users ──< Recommendations (receives recommendations)
        Tourist Places ──< Recommendations (recommended for)
    """
    id                  = models.AutoField(primary_key=True)
    user                = models.ForeignKey(
                              User,
                              on_delete=models.CASCADE,
                              related_name="recommendations",
                              db_column="user_id"
                          )
    place               = models.ForeignKey(
                              TouristPlace,
                              on_delete=models.CASCADE,
                              related_name="recommendations",
                              db_column="place_id"
                          )
    recommendation_type  = models.TextField(blank=True)
    recommendation_score = models.SmallIntegerField(null=True, blank=True)
    recommendation_date  = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "recommendations"

    def __str__(self):
        return f"Rec for {self.user.email} → {self.place.name} (score: {self.recommendation_score})"
