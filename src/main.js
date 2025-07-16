// Initialize the application
document.addEventListener('DOMContentLoaded', async () => {
    // Much faster initialization
    const loadingElement = document.getElementById('loading');
    const appElement = document.getElementById('app');
    
    // Show loading briefly for smooth UX
    loadingElement.style.display = 'flex';
    appElement.style.display = 'none';
    
    // Quick model loading
    await model.loadModel();
    
    // Hide loading and show app
    setTimeout(() => {
        loadingElement.style.display = 'none';
        appElement.style.display = 'block';
        
        // Initialize drag and drop
        initializeDragAndDrop();
    }, 800); // Smooth transition
});