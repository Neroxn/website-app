   function myFunction() {
      var input, filter, ul, li, la, i, txtValue;
      input = document.getElementById("myInput");
      filter = input.value.toUpperCase();
      ul = document.getElementById("myUL");
      li = ul.getElementsByTagName("li");
    
      for (i = 0; i < li.length; i++) { 
        la = li[i].getElementsByTagName("label")[0];
        txtValue = la.textContent || la.innerText; 
        if (txtValue.toUpperCase().indexOf(filter) > -1) {  
          li[i].style.display = "";
        } else {
          li[i].style.display = "none";
        }
      }
    }
    function myFunction2() { 
    var checkBox = document.getElementsByClassName("myCheck");
    var text = document.getElementsByClassName("list-group m1");
    var i;

    for(i = 0; i < checkBox.length; i++)
    {
      if (checkBox[i].checked == true){    
        text[i].style.display = "block";
      } else {
        text[i].style.display = "none";
      }
    }
  }

    function toggle(source) {
          let checkboxes = document.getElementsByName('selected_parameters');
          for(let i = 0; i < checkboxes.length; i++){
            checkboxes[i].checked = !checkboxes[i].checked;
          }
        }
    