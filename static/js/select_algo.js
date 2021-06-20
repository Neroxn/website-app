$(document).ready(function(){
$('#selector').change(function(){

   if ($(this).val() == "SVR"){
       $('#parameters').remove();
        $('#deneme').append("<label id = \'parameters\'><h1>Please enter parameters of SVR algorithm.</h1> <form action=\"\" method=post enctype=multipart/form-data><p> Kernel <select name = \"kernel\"><option value=\"linear\">Linear</option><option value=\"poly\">Poly</option><option value=\"rbf\">RBF</option><option value=\"sigmoid\">Sigmoid</option></select></p> <p>C<input type = \"text\" placeholder = \"1.0\"name = \"C\" /></p><p>Degree <input type = \"text\" placeholder = \"3\" name = \"degree\" /></p><input type=\"checkbox\" name=\"selected_model\" style=\"display:none\" value = \"SVR\" checked><input type=submit value=Sumbit></form></label>");

    }
    else if($(this).val() == "SVC"){
        $('#parameters').remove();
         $('#deneme').append("<label id = \'parameters\'><h1>Please enter parameters of SVC algorithm.</h1> <form action=\"\" method=post enctype=multipart/form-data><p> Kernel <select name = \"kernel\"><option value=\"linear\">Linear</option><option value=\"poly\">Poly</option><option value=\"rbf\">RBF</option><option value=\"sigmoid\">Sigmoid</option></select></p> <p>C<input type = \"text\" placeholder = \"1.0\"name = \"C\" /></p><p>Degree <input type = \"text\" placeholder = \"3\" name = \"degree\" /></p><input type=\"checkbox\" name=\"selected_model\" style=\"display:none\" value = \"SVC\" checked><input type=submit value=Sumbit></form></label>");
 
     }
     else if($(this).val() == "LinearRegression"){
        $('#parameters').remove();
        $('#deneme').append("<label id = \'parameters\'><h1>Linear Regression model.</h1> <form action=\"\" method=post enctype=multipart/form-data><input type=\"checkbox\" name=\"selected_model\" style=\"display:none\" value = \"LinearRegression\" checked><input type=submit value=Sumbit></form></label>"); 
     }
     else if($(this).val() == "LogisticRegression"){
        $('#parameters').remove();
        $('#deneme').append("<label id = \'parameters\'><h1>Please enter parameters of Logistic Regression algorithm.</h1> <form action=\"\" method=post enctype=multipart/form-data><p> Penalty <select name = \"penalty\"><option value=\"l1\">L1</option><option value=\"poly\">L2</option><option value=\"elasticnet\">ElasticNet</option><option value=\"none\">No Penalty</option></select></p> <p>C<input type = \"text\" placeholder=\"1.0\" name = \"C\" /></p><p>L1 Ratio <input type = \"text\" placeholder=\"0.5\" name = \"l1_ratio\" /></p><input type=\"checkbox\" name=\"selected_model\" style=\"display:none\" value = \"LogisticRegression\" checked><input type=submit value=Sumbit></form></label>");
 
     }
    else if($(this).val () == "RandomForestRegressor"){
        $('#parameters').remove();
        $('#deneme').append("<label id = \'parameters\'><h1>Please enter parameters of Random Forest Regression algorithm.</h1> <form action=\"\" method=post enctype=multipart/form-data><p>Number of estimators<input type = \"text\" placeholder=\"100\" name = \"n_estimators\" /></p><p>Max depth<input type = \"text\" placeholder=\"None\" name = \"max_depth\" /></p><p>Min samples for splitting<input type = \"text\" placeholder=\"2\" name = \"min_samples_split\" /></p><p>Min samples in leaf <input type = \"text\" placeholder=\"1\" name = \"min_samples_leaf\" /></p><input type=\"checkbox\" name=\"selected_model\" style=\"display:none\" value = \"RandomForestRegressor\" checked><input type=submit value=Submit></form></label>");
    }
   else if($(this).val () == "RandomForestClassifier"){
        $('#parameters').remove();
        $('#deneme').append("<label id = \'parameters\'> <h1>Please enter parameters of Random Forest Classifier algorithm.</h1> <form action=\"\" method=post enctype=multipart/form-data><p>Number of estimators<input type = \"text\" placeholder=\"100\" name = \"n_estimators\" /></p><p>Max depth<input type = \"text\" placeholder=\"None\" name = \"max_depth\" /></p><p>Min samples for splitting<input type = \"text\" placeholder=\"2\" name = \"min_samples_split\" /></p><p>Min samples in leaf <input type = \"text\" placeholder=\"1\" name = \"min_samples_leaf\" /></p><input type=\"checkbox\" name=\"selected_model\" style=\"display:none\" value = \"RandomForestClassifier\" checked><input type=submit value=Submit></form></label>");
    }
    else if($(this).val () == "AdaBoostRegressor"){
        $('#parameters').remove();
        $('#deneme').append("<label id = \'parameters\'> <h1>Please enter parameters of AdaBoost Regressor algorithm.</h1> <form action=\"\" method=post enctype=multipart/form-data> <p>Number of estimators<input type = \"text\" placeholder=\"50\" name = \"n_estimators\" /></p> <p>Learning rate<input type = \"text\" placeholder=\"1.0\" name = \"learning_rate\" /></p> <input type=\"checkbox\" name=\"selected_model\" style=\"display:none\" value = \"AdaBoostRegressor\" checked> <input type=submit value=Submit></form></label>");
    }
else if($(this).val () == "AdaBoostClassifier"){
    $('#parameters').remove();
    $('#deneme').append("<label id = \'parameters\'> <h1>Please enter parameters of AdaBoost Classifier algorithm.</h1> <form action=\"\" method=post enctype=multipart/form-data> <p>Number of estimators<input type = \"text\" placeholder=\"50\" name = \"n_estimators\" /></p> <p>Learning rate<input type = \"text\" placeholder=\"1.0\" name = \"learning_rate\" /></p> <input type=\"checkbox\" name=\"selected_model\" style=\"display:none\" value = \"AdaBoostRegressor\" checked> <input type=submit value=Submit></form></label>");
}
else if($(this).val () == "ElasticNet"){
    $('#parameters').remove();
    $('#deneme').append("<label id = \'parameters\'> <h1>Please enter parameters of Elastic Net Regression algorithm.</h1> <form action=\"\" method=post enctype=multipart/form-data> <p>Alpha<input type = \"text\" name = \"alpha\" /></p> <p>L1 Ratio <input type = \"text\" name = \"l1_ratio\" /></p> <input type=\"checkbox\" name=\"selected_model\" style=\"display:none\" value = \"LogisticRegression\" checked> <input type=submit value=Submit></form></label>");
}

});
});
