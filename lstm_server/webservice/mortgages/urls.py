from . import views
from django.urls import path
from django.conf.urls import url, include
from mortgages.views import PredictionViewSet
from rest_framework import routers
from mortgages.prediction import Predictor

router = routers.DefaultRouter()

predictor = Predictor()

print(predictor.instance)

router.register(r'predictions', PredictionViewSet, base_name='predictions')

urlpatterns = [
    url(r'^$', views.index, name='index'),
]
urlpatterns += router.urls

