'use strict';
console.log('reading js');


// Variables
var about = document.querySelector('#about');


// Pages
about.addEventListener('click', function() {
  document.querySelector('#projectPage').style.display = 'none';
  document.querySelector('#AboutPage').style.display = 'block';
});
