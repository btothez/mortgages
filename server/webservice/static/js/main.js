var App = angular.module('App', []);

App.controller('PredictionController', function PredictionController($scope, $http) {
    $scope.prediction = null;
    $scope.submitText = () => {
        console.log('getting ...')
        $http.get('/predictions', {
            words: $scope.textString
        }).then((result) => {
            console.log('result', result);
            $scope.prediction = result.data;
        }).catch((err) => {
            console.log('err', err);
        });
    };
});
