var App = angular.module('App', []);

App.controller('PredictionController', function PredictionController($scope, $http) {
    $scope.prediction = null;
    $scope.textString = '';
    $scope.submitText = () => {
        console.log('getting ...')
        $http.get('/predictions', {
            params: {
                words: $scope.textString
            }
        }).then((result) => {
            console.log('result', result);
            $scope.prediction = result.data;
        }).catch((err) => {
            console.log('caught an error');
            if (err.data && err.data.message) {
                console.error(err.data.message);
            }
        });
    };
});
