let slideIndex = 0;

window.onload = function() {
    showSlides(slideIndex);

    // Get elements
    const prev = document.querySelector('.prev');
    const next = document.querySelector('.next');
    const dots = Array.from(document.querySelectorAll('.dot'));

    // Add event listeners
    prev.addEventListener('click', () => changeSlide(-1));
    next.addEventListener('click', () => changeSlide(1));
    dots.forEach((dot, index) => {
        dot.addEventListener('click', () => currentSlide(index));
    });
}

function changeSlide(n) {
    showSlides(slideIndex += n);
}

function currentSlide(n) {
    showSlides(slideIndex = n);
}

function showSlides(n) {
    const slides = document.querySelectorAll('.mySlides');
    const dots = document.querySelectorAll('.dot');

    if (n >= slides.length) {slideIndex = 0} 
    if (n < 0) {slideIndex = slides.length - 1}

    slides.forEach((slide, index) => {
        slide.style.display = (index === slideIndex) ? "block" : "none"; 
    });

    dots.forEach((dot, index) => {
        dot.className = dot.className.replace(" active", "");
        if(index === slideIndex) dot.className += " active";
    });
}