// Copy BibTeX to clipboard
function copyBibTeX() {
    const bibtexElement = document.getElementById('bibtex-code');
    const button = document.querySelector('.copy-bibtex-btn');
    const copyText = button.querySelector('.copy-text');

    if (!bibtexElement) return;

    function showCopied() {
        // The CSS appends "ied!" so the label reads "Copied!"
        button.classList.add('copied');
        copyText.textContent = 'Cop';
        setTimeout(function() {
            button.classList.remove('copied');
            copyText.textContent = 'Copy';
        }, 2000);
    }

    function fallbackCopy() {
        const textArea = document.createElement('textarea');
        textArea.value = bibtexElement.textContent;
        document.body.appendChild(textArea);
        textArea.select();
        document.execCommand('copy');
        document.body.removeChild(textArea);
        showCopied();
    }

    if (navigator.clipboard && window.isSecureContext) {
        navigator.clipboard.writeText(bibtexElement.textContent).then(showCopied).catch(fallbackCopy);
    } else {
        fallbackCopy();
    }
}

// Scroll to top functionality
function scrollToTop() {
    window.scrollTo({
        top: 0,
        behavior: 'smooth'
    });
}

// Show/hide scroll to top button
window.addEventListener('scroll', function() {
    const scrollButton = document.querySelector('.scroll-to-top');
    if (window.pageYOffset > 300) {
        scrollButton.classList.add('visible');
    } else {
        scrollButton.classList.remove('visible');
    }
});

// Only play the comparison video while it is on screen
document.addEventListener('DOMContentLoaded', function() {
    const videos = document.querySelectorAll('video[autoplay]');
    if (!('IntersectionObserver' in window) || videos.length === 0) return;

    const observer = new IntersectionObserver(function(entries) {
        entries.forEach(function(entry) {
            if (entry.isIntersecting) {
                entry.target.play().catch(function() {});
            } else {
                entry.target.pause();
            }
        });
    }, { threshold: 0.25 });

    videos.forEach(function(video) { observer.observe(video); });
});
