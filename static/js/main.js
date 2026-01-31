// Main JavaScript functionality for Machine Vision Plus

document.addEventListener('DOMContentLoaded', function() {
    // Initialize the application
    initializeApp();
    
    // Set up real-time stats updates
    updateStatsPeriodically();
    
    // Add smooth scrolling for navigation links
    setupSmoothScrolling();
    
    // Add loading states and animations
    setupLoadingStates();
});

function initializeApp() {
    console.log('Machine Vision Plus initialized');
    
    // Add any global initialization code here
    setupImagePreview();
    setupDragAndDrop();
}

function updateStatsPeriodically() {
    // Update visitor stats every 30 seconds
    setInterval(async () => {
        try {
            const response = await fetch('/api/stats');
            const stats = await response.json();
            
            // Update visitor counts in footer
            updateVisitorCounts(stats);
            
        } catch (error) {
            console.log('Failed to update stats:', error);
        }
    }, 30000);
}

function updateVisitorCounts(stats) {
    // Update total visitors
    const totalVisitors = document.querySelector('.stat-item:nth-child(1) span');
    if (totalVisitors) {
        totalVisitors.textContent = `Total Visitors: ${stats.total_visitors}`;
    }
    
    // Update unique visitors
    const uniqueVisitors = document.querySelector('.stat-item:nth-child(2) span');
    if (uniqueVisitors) {
        uniqueVisitors.textContent = `Unique Visitors: ${stats.unique_visitors}`;
    }
    
    // Update countries count
    const countriesCount = document.querySelector('.stat-content h3');
    if (countriesCount && stats.countries) {
        countriesCount.textContent = stats.countries.length;
    }
}

function setupSmoothScrolling() {
    // Add smooth scrolling behavior for anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
}

function setupLoadingStates() {
    // Add loading animation for buttons
    document.querySelectorAll('.btn').forEach(button => {
        button.addEventListener('click', function() {
            if (!this.disabled) {
                this.style.position = 'relative';
                this.style.overflow = 'hidden';
                
                // Add ripple effect
                const ripple = document.createElement('span');
                ripple.style.position = 'absolute';
                ripple.style.borderRadius = '50%';
                ripple.style.background = 'rgba(255, 255, 255, 0.3)';
                ripple.style.transform = 'scale(0)';
                ripple.style.animation = 'ripple 0.6s linear';
                ripple.style.left = '50%';
                ripple.style.top = '50%';
                ripple.style.width = '20px';
                ripple.style.height = '20px';
                ripple.style.marginLeft = '-10px';
                ripple.style.marginTop = '-10px';
                
                this.appendChild(ripple);
                
                setTimeout(() => {
                    ripple.remove();
                }, 600);
            }
        });
    });
}

function setupImagePreview() {
    // Enhanced image preview functionality
    const imageInputs = document.querySelectorAll('input[type="file"][accept*="image"]');
    
    imageInputs.forEach(input => {
        input.addEventListener('change', function(e) {
            const file = e.target.files[0];
            if (file) {
                previewImage(file, this);
            }
        });
    });
}

function previewImage(file, input) {
    const reader = new FileReader();
    reader.onload = function(e) {
        // Find the preview container
        const previewContainer = input.closest('.upload-area') || 
                                document.querySelector('.preview-box') ||
                                document.querySelector('.image-container');
        
        if (previewContainer) {
            // Create or update preview image
            let previewImg = previewContainer.querySelector('img');
            if (!previewImg) {
                previewImg = document.createElement('img');
                previewImg.style.maxWidth = '100%';
                previewImg.style.borderRadius = '10px';
                previewContainer.appendChild(previewImg);
            }
            
            previewImg.src = e.target.result;
            previewImg.alt = 'Preview';
            
            // Hide upload text if present
            const uploadText = previewContainer.querySelector('.upload-content');
            if (uploadText) {
                uploadText.style.display = 'none';
            }
        }
    };
    reader.readAsDataURL(file);
}

function setupDragAndDrop() {
    // Enhanced drag and drop functionality
    const dropAreas = document.querySelectorAll('.upload-area');
    
    dropAreas.forEach(area => {
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            area.addEventListener(eventName, preventDefaults, false);
        });
        
        ['dragenter', 'dragover'].forEach(eventName => {
            area.addEventListener(eventName, highlight, false);
        });
        
        ['dragleave', 'drop'].forEach(eventName => {
            area.addEventListener(eventName, unhighlight, false);
        });
        
        area.addEventListener('drop', handleDrop, false);
    });
}

function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
}

function highlight(e) {
    e.currentTarget.classList.add('dragover');
}

function unhighlight(e) {
    e.currentTarget.classList.remove('dragover');
}

function handleDrop(e) {
    const dt = e.dataTransfer;
    const files = dt.files;
    
    if (files.length > 0) {
        const file = files[0];
        if (file.type.startsWith('image/')) {
            // Find the associated file input
            const input = e.currentTarget.querySelector('input[type="file"]');
            if (input) {
                // Create a new FileList-like object
                const dataTransfer = new DataTransfer();
                dataTransfer.items.add(file);
                input.files = dataTransfer.files;
                
                // Trigger change event
                const event = new Event('change', { bubbles: true });
                input.dispatchEvent(event);
            }
        } else {
            showNotification('Please select a valid image file.', 'error');
        }
    }
}

function showNotification(message, type = 'info') {
    // Create notification element
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.textContent = message;
    
    // Style the notification
    notification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        padding: 15px 20px;
        border-radius: 8px;
        color: white;
        font-weight: 500;
        z-index: 10000;
        transform: translateX(100%);
        transition: transform 0.3s ease;
        max-width: 300px;
        word-wrap: break-word;
    `;
    
    // Set background color based on type
    const colors = {
        'info': '#667eea',
        'success': '#28a745',
        'error': '#dc3545',
        'warning': '#ffc107'
    };
    notification.style.backgroundColor = colors[type] || colors.info;
    
    // Add to page
    document.body.appendChild(notification);
    
    // Animate in
    setTimeout(() => {
        notification.style.transform = 'translateX(0)';
    }, 100);
    
    // Remove after 5 seconds
    setTimeout(() => {
        notification.style.transform = 'translateX(100%)';
        setTimeout(() => {
            if (notification.parentNode) {
                notification.parentNode.removeChild(notification);
            }
        }, 300);
    }, 5000);
}

// Add CSS for ripple animation
const style = document.createElement('style');
style.textContent = `
    @keyframes ripple {
        to {
            transform: scale(4);
            opacity: 0;
        }
    }
    
    .notification {
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    }
`;
document.head.appendChild(style);

// Utility functions
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

function validateImageFile(file) {
    const validTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/gif', 'image/webp'];
    const maxSize = 10 * 1024 * 1024; // 10MB
    
    if (!validTypes.includes(file.type)) {
        return { valid: false, message: 'Please select a valid image file (JPEG, PNG, GIF, or WebP).' };
    }
    
    if (file.size > maxSize) {
        return { valid: false, message: 'File size must be less than 10MB.' };
    }
    
    return { valid: true };
}

// Export functions for use in other scripts
window.MachineVisionPlus = {
    showNotification,
    formatFileSize,
    validateImageFile,
    updateVisitorCounts
};
