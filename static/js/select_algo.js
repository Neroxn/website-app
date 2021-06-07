$(document).ready(function(){
$('#selector').change(function(){
   if($(this).val() == "Adaboost"){
       $('#parameters').remove();
        $('#deneme').append("<label id = \'parameters\'><h1>Please enter parameters of Adaboost algorithm</h1><form action=\"\" method=post enctype=multipart/form-data><p>Number of estimator<input type = \"text\" name = \"numberEstimator\" /></p><p>Learning rate<input type = \"text\" name = \"learningRate\" /></p><p>Loss<input type = \"text\" name = \"loss\" /></p><input type=\"checkbox\" name=\"selector\" style=\"display:none\" value = \"AdaBoost\" checked><input type=submit value=Sumbit></form></label>");
   }
   else if($(this).val() == "SVM"){
       $('#parameters').remove();
              $('#deneme').append("<label id = \'parameters\'><h1>Please enter parameters of SVM algorithm</h1><form action=\"\" method=post enctype=multipart/form-data><p>Kernel<input type = \"text\" name = \"kernel\" /></p><p>C<input type = \"text\" name = \"C\" /></p><p>(if kernel is NOT linear)Gamma <input type = \"text\" name = \"gamma\" /></p><p>(if kernel is poly)Degree<input type = \"text\" name = \"degree\" /></p><input type=\"checkbox\" name=\"selector\" style=\"display:none\" value = \"SVM\" checked><input type=submit value=Sumbit></form></label>");
   }
    else if($(this).val() == "RandomForest"){
        $('#parameters').remove();
        $('#deneme').append("<label id = \'parameters\'><h1>Please enter parameters of RandomForest algorithm</h1><form action=\"\" method=post enctype=multipart/form-data><p>Number of estimator<input type = \"text\" name = \"numberEstimator\" /></p><p>maxDepth(int) <input type = \"text\" name = \"maxDepth\" /></p><p>Minimum Samples Leaf<input type = \"text\" name = \"minSamplesLeaf\" /></p><input type=\"checkbox\" name=\"selector\" style=\"display:none\" value = \"RandomForest\" checked><input type=submit value=Sumbit></form></label>");
   }
});
});