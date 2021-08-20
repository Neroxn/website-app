function toggle(source) {
  let checkboxes = document.getElementsByName('hello');
  for(let i = 0; i < checkboxes.length; i++){
    checkboxes[i].checked = !checkboxes[i].checked;
  }
}

$(document).ready(function(){
  $("#myInput").on("keyup", function() {
    var value = $(this).val().toLowerCase();
    $("#myTable tr").filter(function() {
      $(this).toggle($(this).text().toLowerCase().indexOf(value) > -1)
    });
  });
});