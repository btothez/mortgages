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
            print(words)
            print(words.split(' '))
            predictor = Predictor()
            result, conf = predictor.predict(words)
            label = "...Could not solve... :("
            probability = 0.0
            if type(result) == list and len(result):
                label = result[0]
            if type(conf) == list and len(conf):
                probability = conf[0]
            print(label, probability)
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


