function toggle(source) {
  let checkboxes = document.getElementsByName('hello');
  for(let i = 0; i < checkboxes.length; i++){
    checkboxes[i].checked = !checkboxes[i].checked;
  }
}