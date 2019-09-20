from rest_framework import serializers

class PredictionSerializer(serializers.Serializer):
    prediction = serializers.IntegerField()
    confidence = serializers.DecimalField(max_digits=5, decimal_places=2)
