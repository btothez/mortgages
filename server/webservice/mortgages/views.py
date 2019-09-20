from django.shortcuts import render
from django.http import HttpResponse
from rest_framework import viewsets, views
from rest_framework.response import Response
from mortgages.serializers import PredictionSerializer
from rest_framework.response import Response

def index(request):
    context = {}
    return render(request, 'mortgages/index.html', context)
"""
class PredictionView(views.APIView):
    def __init__(self):
        self.get_extra_actions = []
        super().__init__() 

    def get(self, request):
        print('!!!!!!!!!!!!!!')
        data = {'prediciton': 10}
        results = PredictionSerializer(data).data
        return Response(results)
"""

class PredictionViewSet(viewsets.ViewSet):
    # Required for the Browsable API renderer to have a nice form.
    serializer_class = PredictionSerializer

    def list(self, request):
        serializer = PredictionSerializer({
            'prediction': 40
        })
        return Response(serializer.data)
