const dropArea = document.getElementById('drop-area');
const loadingState = document.getElementById('loading-state');
const resultsSection = document.getElementById('results-section');
const resultImage = document.getElementById('result-image');
const carCount = document.getElementById('car-count');
const detectionsUl = document.getElementById('detections-ul');

// Prevent default drag behaviors
['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
    dropArea.addEventListener(eventName, preventDefaults, false);
    document.body.addEventListener(eventName, preventDefaults, false);
});

function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
}

// Highlight drop area when item is dragged over it
['dragenter', 'dragover'].forEach(eventName => {
    dropArea.addEventListener(eventName, () => {
        dropArea.querySelector('.upload-box').classList.add('highlight');
    }, false);
});

['dragleave', 'drop'].forEach(eventName => {
    dropArea.addEventListener(eventName, () => {
        dropArea.querySelector('.upload-box').classList.remove('highlight');
    }, false);
});

// Handle dropped files
dropArea.addEventListener('drop', (e) => {
    const dt = e.dataTransfer;
    const files = dt.files;
    handleFiles(files);
}, false);

function handleFiles(files) {
    if (files.length === 0) return;
    const file = files[0];
    
    if (!file.type.startsWith('image/')) {
        alert('Please upload an image file.');
        return;
    }

    uploadImage(file);
}

async function uploadImage(file) {
    // Show loading state
    dropArea.classList.add('hidden');
    resultsSection.classList.add('hidden');
    loadingState.classList.remove('hidden');

    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`Server error: ${response.statusText}`);
        }

        const data = await response.json();
        
        if (data.success) {
            displayResults(data);
        } else {
            throw new Error(data.error || "Unknown error occurred");
        }
    } catch (error) {
        console.error('Error:', error);
        alert('An error occurred during analysis. Please try again.');
        resetApp();
    }
}

function displayResults(data) {
    // Hide loading, show results
    loadingState.classList.add('hidden');
    resultsSection.classList.remove('hidden');
    
    // Set image
    resultImage.src = data.image;
    
    // Set stats
    carCount.textContent = data.car_count;
    
    // Populate detections list
    detectionsUl.innerHTML = '';
    
    if (data.detections.length === 0) {
        const li = document.createElement('li');
        li.innerHTML = '<span class="det-name" style="color: var(--text-muted)">No objects detected</span>';
        detectionsUl.appendChild(li);
    } else {
        // Sort by confidence
        const sortedDetections = data.detections.sort((a, b) => b.confidence - a.confidence);
        
        sortedDetections.forEach(det => {
            const li = document.createElement('li');
            const percent = (det.confidence * 100).toFixed(1);
            li.innerHTML = `
                <span class="det-name">${det.class}</span>
                <span class="det-conf">${percent}%</span>
            `;
            detectionsUl.appendChild(li);
        });
    }
}

function resetApp() {
    resultsSection.classList.add('hidden');
    loadingState.classList.add('hidden');
    dropArea.classList.remove('hidden');
    
    // Clear input
    document.getElementById('fileElem').value = '';
}
