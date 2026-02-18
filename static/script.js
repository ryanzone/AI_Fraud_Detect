function analyze() {
    fetch("/analyze", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({
            text: document.getElementById("text").value,
            url: document.getElementById("url").value
        })
    })
    .then(response => response.json())
    .then(data => {
        let output = "";

        if (data.text_result) {
            output += "Text: " + data.text_result +
                      " (Confidence: " + data.text_confidence + "%)<br>";
        }

        if (data.url_result) {
            output += "URL: " + data.url_result;
        }

        document.getElementById("result").innerHTML = output;
    });
}
