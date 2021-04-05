$(document).ready(function(){
    $("#about").click(function(){
        $("#aboutPage").show();
        $("#banner").css("background-image", "linear-gradient(to right, #00c0ff , #2ad8a6)");
        $("#homepage, #stockAnalysis, #projectbckgrnd, #pdfEx, #pivotTable, #flexDash, #randomForest, #wildfireOne, #wildfireTwo, #youtube").hide();
        $("#banner").css("padding", "2em");
    });
});

// BITCOIN REGRESSION
$(document).ready(function(){
    $(".bitcoinProj").click(function(){
        $("#bitcoinRegression, #projectbckgrnd").show();
        $("#banner").css("background-image", "linear-gradient(to right, #fff, #fff)");
        $("#banner").css("padding-right", "0");
        $("#homepage, #aboutPage, #pdfEx, #pivotTable, #flexDash, #randomForest, #wildfireOne, #wildfireTwo, #youtube, #stockAnalysis").hide();
    });
});

// STOCK ANALYSIS
$(document).ready(function(){
    $("#featured, .stockProj").click(function(){
        $("#stockAnalysis, #projectbckgrnd").show();
        $("#banner").css("background-image", "linear-gradient(to right, #fff, #fff)");
        $("#banner").css("padding-right", "0");
        $("#homepage, #aboutPage, #pdfEx, #pivotTable, #flexDash, #randomForest, #wildfireOne, #wildfireTwo, #youtube, #bitcoinRegression").hide();
    });
});

// PDF EXPRESSIONS
$(document).ready(function(){
    $(".pdfExProj").click(function(){
        $("#pdfEx, #projectbckgrnd").show();
        $("#banner").css("background-image", "linear-gradient(to right, #fff, #fff)");
        $("#banner").css("padding-right", "0");
        $("#homepage, #aboutPage, #stockAnalysis, #pivotTable, #flexDash, #randomForest, #wildfireOne, #wildfireTwo, #youtube, #bitcoinRegression").hide();
    });
});

// PIVOT TABLES
$(document).ready(function(){
    $(".ptProj").click(function(){
        $("#pivotTable, #projectbckgrnd").show();
        $("#banner").css("background-image", "linear-gradient(to right, #fff, #fff)");
        $("#banner").css("padding-right", "0");
        $("#homepage, #aboutPage, #stockAnalysis, #pdfEx, #flexDash, #randomForest,  #wildfireOne, #wildfireTwo, #youtube, #bitcoinRegression").hide();
    });
});

// FLEX DASHBOARD
$(document).ready(function(){
    $(".flexDashProj").click(function(){
        $("#flexDash, #projectbckgrnd").show();
        $("#banner").css("background-image", "linear-gradient(to right, #fff, #fff)");
        $("#banner").css("padding-right", "0");
        $("#homepage, #aboutPage, #stockAnalysis, #pdfEx, #pivotTable, #randomForest, #wildfireOne, #wildfireTwo, #youtube, #bitcoinRegression").hide();
    });
});

// MACHINE LEARNING RANDOM FOREST
$(document).ready(function(){
    $(".randomForestProj").click(function(){
        $("#randomForest, #projectbckgrnd").show();
        $("#banner").css("background-image", "linear-gradient(to right, #fff, #fff)");
        $("#banner").css("padding-right", "0");
        $("#homepage, #aboutPage, #stockAnalysis, #pdfEx, #pivotTable, #flexDash, #wildfireOne, #wildfireTwo, #youtube, #bitcoinRegression").hide();
    });
});

// WILDFIRE TWO
$(document).ready(function(){
    $(".wf2Proj").click(function(){
        $("#wildfireTwo, #projectbckgrnd").show();
        $("#banner").css("background-image", "linear-gradient(to right, #fff, #fff)");
        $("#banner").css("padding-right", "0");
        $("#homepage, #aboutPage, #stockAnalysis, #pdfEx, #pivotTable, #flexDash, #randomForest, #wildfireOne, #youtube, #bitcoinRegression").hide();
    });
});

// WILDFIRE ONE
$(document).ready(function(){
    $(".wf1Proj").click(function(){
        $("#wildfireOne, #projectbckgrnd").show();
        $("#banner").css("background-image", "linear-gradient(to right, #fff, #fff)");
        $("#banner").css("padding-right", "0");
        $("#homepage, #aboutPage, #stockAnalysis, #pdfEx, #pivotTable, #flexDash, #randomForest, #wildfireTwo, #youtube, #bitcoinRegression").hide();
    });
});

// YOUTUBE
$(document).ready(function(){
    $(".ytProj").click(function(){
        $("#youtube, #projectbckgrnd").show();
        $("#banner").css("background-image", "linear-gradient(to right, #fff, #fff)");
        $("#banner").css("padding-right", "0");
        $("#homepage, #aboutPage, #stockAnalysis, #pdfEx, #pivotTable, #flexDash, #randomForest, #wildfireTwo, #wildfireOne, #bitcoinRegression").hide();
    });
});

// CLOSE TO HOMEPAGE
$("button").click(function(){
  $("#aboutPage, #stockAnalysis, #projectbckgrnd, #pdfEx").hide();
  $("#banner").css("background-image", "linear-gradient(to right, #00c0ff , #2ad8a6)");
  $("#banner").css("padding", "2em");
  $("#homepage").show();
});
