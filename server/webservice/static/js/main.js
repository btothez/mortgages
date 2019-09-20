var App = angular.module('App', []);

App.controller('PredictionController', function PredictionController($scope) {
    $scope.hello = 'World!!!!';
    console.log($scope.hello);
    console.log('Boaz');
});
