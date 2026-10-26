const sliders = ['ph', 'temp', 'moist', 'n', 'p', 'k'];

sliders.forEach(id => {
    document.getElementById(id).addEventListener('input', function() {
        document.getElementById(id + 'Val').innerText = this.value;
    });
});

async function getPrediction() {
    const payload = {
        Soil: document.getElementById('soilType').value,
        Crop: document.getElementById('cropType').value,
        PH: parseFloat(document.getElementById('ph').value),
        Temperature: parseFloat(document.getElementById('temp').value),
        Moisture: parseFloat(document.getElementById('moist').value),
        Nitrogen: parseFloat(document.getElementById('n').value),
        Phosphorous: parseFloat(document.getElementById('p').value),
        Potassium: parseFloat(document.getElementById('k').value)
    };

    const resultText = document.getElementById('fertResult');
    resultText.innerText = "Processing...";
    resultText.style.color = "#333";

    try {
        const response = await fetch("http://127.0.0.1:8000/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });
        const result = await response.json();
        resultText.innerText = result.predicted_fertilizer;
        resultText.style.color = "#2e7d32";
    } catch (error) {
        resultText.innerText = "API Offline / Error";
        resultText.style.color = "#d32f2f";
    }
}