from django.shortcuts import render
from django.http import HttpResponse
from rest_framework import viewsets, views
from rest_framework.response import Response
from mortgages.serializers import PredictionSerializer
from rest_framework.response import Response
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
            result = predictor.predict(words)
            label = "...Could not solve... :("
            print('reust')
            print(result)
            print(type(result))
            if type(result) == list and len(result):
                label = result[0]
            print(label)
            print('!!!!!!')
            serializer = PredictionSerializer({
                'prediction': label,
                'confidence': .77
            })
            return Response(serializer.data)
        except Exception:
            return None
