from django.apps import AppConfig
import sys
import logging

logger = logging.getLogger(__name__)


class RecommendationConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'recommendation'

    def ready(self):
        skip_commands = {'migrate', 'makemigrations', 'test', 'collectstatic'}
        running_command = sys.argv[1] if len(sys.argv) > 1 else ''

        if running_command not in skip_commands:
            try:
                from .views import initialize_tfidf_model
                initialize_tfidf_model()
            except Exception as e:
                logger.warning(f"تعذر بناء نموذج TF-IDF: {e}")