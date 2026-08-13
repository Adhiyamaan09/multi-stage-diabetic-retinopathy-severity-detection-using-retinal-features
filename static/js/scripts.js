var testButton = document.getElementById("testbutton");
var img = document.getElementById("imageinput");
var vesselimage = document.getElementById("vesselimage");
var exudateimage = document.getElementById("exudateimage");
var resultclass = document.getElementById("resultclass");
var resultbox = document.getElementById("resultbox");
var resultaccuracy = document.getElementById("resultaccuracy");
var predgraphimage = document.getElementById("predgraphimage");
var graphviewButton = document.getElementById("graphviewButton");
var graphcontainer = document.getElementById("graphcontainer");
var refreshButton = document.getElementById("tryagainButton");


function readURL(input){
	if(input.files && input.files[0]){
		let reader = new FileReader();
		reader.onload = function(e) {
			var retinaimage = document.getElementById("retinaimage");
			retinaimage.src = e.target.result;
			retinaimage.style.height = "75%";
			retinaimage.style.width = "75%";
		}
		reader.readAsDataURL(input.files[0]);
	}	
}


document.addEventListener("DOMContentLoaded", function() {
    // Add event listener for the button click
    testButton.addEventListener("click", function(event) {
        event.preventDefault(); // Prevent the default form submission
        
        var file = img.files[0];

        if (file) {
            var formData = new FormData();
            formData.append("image", file);

            fetch("http://localhost:5000/process_image", {
                method: "POST",
                body: formData,
            })
            .then(response => {
                if (!response.ok) {
                    throw new Error("Network response was not ok " + response.statusText);
                }
                return response.json();
            })
            .then(data => {
                vesselimage.src = "data:image/png;base64," + data.vesselimage;
				vesselimage.style.height = "75%";
				vesselimage.style.width = "75%";
                
                exudateimage.src = "data:image/png;base64," + data.exudateimage;
				exudateimage.style.height = "75%";
				exudateimage.style.width = "75%";
            })
            .catch(error => {
                console.error("Error:", error);
                alert("Error: " + error.message);
            });
        } else {
            alert("Please select an image file first.");
        }
    });
});


document.addEventListener("DOMContentLoaded", function() {
    // Add event listener for the button click
    testButton.addEventListener("click", function(event) {
        event.preventDefault(); // Prevent the default form submission

        var file = img.files[0];

        if (file) {
            var formData = new FormData();
            formData.append("image", file);

            fetch("http://localhost:5000/get_prediction", {
                method: "POST",
                body: formData,
            })
            .then(response => {
                if (!response.ok) {
                    throw new Error("Network response was not ok " + response.statusText);
                }
                return response.json();
            })
            .then(data => {
                resultclass.innerHTML = "Result: " + data.predclass;
                resultaccuracy.innerHTML = "Accuracy: " + data.predprob.toFixed(2) + "%";
                
                predgraphimage.src = "data:image/png;base64," + data.predgraphimage;
                predgraphimage.style.height = "120%";
				predgraphimage.style.width = "120%";

                resultbox.style.display = "block";
            })
            .catch(error => {
                console.error("Error:", error);
                alert("Error: " + error.message);
            });
        } else {
            alert("Please select an image file first.");
        }
    });
});


document.addEventListener("DOMContentLoaded", function() {
    graphviewButton.addEventListener("click", function(event) {
        if (graphcontainer.className == "hide") {
            graphcontainer.className = "show";
        } else {
            graphcontainer.className = "hide";
        }
    });
});


document.addEventListener("DOMContentLoaded", function() {
    refreshButton.addEventListener("click", function(event) {
        location.reload(true);
    });
});


document.addEventListener("DOMContentLoaded", function() {
    refreshButton.addEventListener("click", function(event) {
        location.reload(true);
    });
});
