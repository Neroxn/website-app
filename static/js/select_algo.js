$(document).ready(function(){
$('#selector').change(function(){
   if($(this).val() == "Adaboost"){
       $('#parameters').remove();
        $('#deneme').append("<label id = \'parameters\'>
        <div class="form-group">
			<h1>Please enter parameters of Adaboost algorithm</h1>
			<form action=\"\" method=post enctype=multipart/form-data>
				<label>Number of estimator</label>
				<input type = \"text\"  class="form-control" name = \"numberEstimator\" />
				<label>Learning rate</label>
				<input type = \"text\" class="form-control" name = \"learningRate\" />
				<label>Loss</label>
				<input type = \"text\" class="form-control" name = \"loss\" />
				<input type=\"checkbox\" class="form-control"name=\"selector\" style=\"display:none\" value = \"AdaBoost\" checked>
				<input type=submit class="btn btn-primary active mt-2" value=Sumbit>
			</form>
		</div>
		</label>");
   }
   else if($(this).val() == "SVM"){
       $('#parameters').remove();
        $('#deneme').append("<label id = \'parameters\'>
		<div class="form-group">
			<h1>Please enter parameters of SVM algorithm</h1>
			<form action=\"\" method=post enctype=multipart/form-data>
        			<label>Kernel</label>
            		<input type = \"text\" class="form-control" name = \"kernel\" /></p>
        			<label>C</label>
            		<input type = \"text\" class="form-control" name = \"C\" />
				<label>(if kernel is NOT linear)Gamma </label>
            		<input type = \"text\" class="form-control" name = \"gamma\" />
				<label>(if kernel is poly)Degree</label>
            		<input type = \"text\" class="form-control" name = \"degree\" />
				<input type=\"checkbox\" class="form-control" name=\"selector\" style=\"display:none\" value = \"SVM\" checked>
				<input type=submit class="btn btn-primary active mt-2" value=Sumbit>
			</form>
		</div>
		</label>");
   }
    else if($(this).val() == "RandomForest"){
        $('#parameters').remove();
         $('#deneme').append("<label id = \'parameters\'>
		 <div class="form-group">
		 	<h1>Please enter parameters of RandomForest algorithm</h1>
         	<form action=\"\" method=post enctype=multipart/form-data>
        	 		<label>Number of estimator</label>
            		<input type = \"text\" class="form-control" name = \"numberEstimator\" />
        			<label>maxDepth(int)</label> 
            		<input type = \"text\" class="form-control" name = \"maxDepth\" />
        			<label>Minimum Samples Leaf</label>
            		<input type = \"text\" class="form-control" name = \"minSamplesLeaf\" />
        			<input type=\"checkbox\" class="form-control" name=\"selector\" style=\"display:none\" value = \"RandomForest\" checked>				
            		<input type=submit class="btn btn-primary active mt-2" value=Sumbit>
    			</form>
		 </div>
		 </label>");
   }
});
});