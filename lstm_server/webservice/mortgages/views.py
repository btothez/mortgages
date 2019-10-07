from django.shortcuts import render
from django.http import HttpResponse
from rest_framework import viewsets, views
from rest_framework.response import Response
from django.http.response import JsonResponse
from mortgages.serializers import PredictionSerializer
from mortgages.prediction import Predictor

def index(request):
    context = {}
    return render(request, 'mortgages/index.html', context)

class PredictionViewSet(viewsets.ViewSet):
    serializer_class = PredictionSerializer

    def list(self, request):
        try:
            words = request.GET['words']
            predictor = Predictor()
            result, conf = predictor.predict(words)
            label = "...Could not solve... :("
            probability = 0.0
            if type(result) == str:
                label = result
            if type(conf) == str:
                probability = conf
            serializer = PredictionSerializer({
                'prediction': label,
                'confidence': probability
            })
            return Response(serializer.data)
        except Exception:
           return JsonResponse(
               {
                   'status':'false',
                   'message':'Unable to process request'
               },
               status=500
           )


