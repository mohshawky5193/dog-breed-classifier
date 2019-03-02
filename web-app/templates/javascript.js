$(document).ready(function(){
    $(".upload-button").click(function(){
        $("#fileId").click();
    })
    $("#fileId").change(function(){
        $("#fileId").click();
        var value = $("#fileId").val();
        var text1=$("#labelId").text();
        if (value != ""){
            $("#labelId").html(value.split('\\').pop());
        } else {
            $("#labelId").html("Choose a file");
        }
    })
});