"""
WSGI config for supermining project.

It exposes the WSGI callable as a recommondation_module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/1.11/howto/deployment/wsgi/
"""

import os

from django.core.wsgi import get_wsgi_application

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "supermining.settings")

application = get_wsgi_application()
