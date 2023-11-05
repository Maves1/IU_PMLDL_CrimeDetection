document.addEventListener('DOMContentLoaded', function () {
    var completeJokeBtn = document.getElementById('completeJokeBtn');
    completeJokeBtn.addEventListener('click', function (event) {
        event.preventDefault(); // Prevent the default form submission
        var jokeStart = document.getElementById('jokeStart').value;
        var formData = new FormData();
        formData.append('start', jokeStart);

        fetch('/joke/new_joke', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.text(); // Expecting an HTML response
        })
        .then(html => {
            // Update the page with the new HTML content
            document.open();
            document.write(html);
            document.close();
        })
        .catch(error => {
            document.getElementById('jokeResult').innerText = 'Sorry, there was an error completing your joke.';
        });
    });
});
