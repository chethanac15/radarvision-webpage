document.addEventListener('DOMContentLoaded', function() {  
    document.body.classList.add('loaded');  
 
    const imageElement = document.getElementById('image');  
    const resultsBox = document.getElementById('results');  
    const spinner = document.getElementById('spinner');  
    const predictionElement = document.getElementById('prediction');  
    const confidenceElement = document.getElementById('confidence');     
    const sections = document.querySelectorAll('.oneOne, .twoTwo, .threeThree');  

    window.addEventListener('scroll', function() {
        const backToTopButton = document.getElementById('back-to-top');
        // Show the button when scrolled down more than 300px, and hide it when scrolled up
        if (window.scrollY > 300) {
            backToTopButton.style.visibility = 'visible'; // Show button suddenly
        } else {
            backToTopButton.style.visibility = 'hidden'; // Hide button suddenly
        }
    });
    
    // Smoothly scroll to the top when the button is clicked
    document.getElementById('back-to-top').addEventListener('click', function() {
        window.scrollTo({ top: 0, behavior: 'smooth' });
    });
    

    // Update the image preview  
    const updateImagePreview = (imageSrc) => {  
        const imgElement = document.createElement('img');  
        imgElement.src = imageSrc;  
        imgElement.style.width = '50%';  
        imgElement.onload = () => imgElement.classList.add('visible');  
        document.getElementById('preview').innerHTML = '';  
        document.getElementById('preview').appendChild(imgElement);  
    };  

    // Event listeners for image input and prediction form submission  
    imageElement.addEventListener('change', (event) => {
        const file = event.target.files[0];
        if (file && file.type.startsWith('image/')) {
            const reader = new FileReader();
            reader.onload = (e) => updateImagePreview(e.target.result);
            reader.readAsDataURL(file);
        }
    });
    document.getElementById('prediction-form').addEventListener('submit', (event) => {  
        event.preventDefault();  
        const formData = new FormData(event.target);  
        spinner.style.display = 'block';  
        resultsBox.classList.remove('visible');  

        fetch('/predict', {  
            method: 'POST',
            body: formData  
        })  
        .then(response => response.json())  
        .then(data => {  
            predictionElement.innerText = `Predicted Class: ${data.prediction}`;  
            confidenceElement.innerText = `Confidence Score: ${data.confidence.toFixed(2)*100}%`;  
            spinner.style.display = 'none';  
            resultsBox.classList.add('visible');  
        })  
        .catch(error => {  
            console.error('Error:', error);  
            spinner.style.display = 'none';  
        });  
    });  
});