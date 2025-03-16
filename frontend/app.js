async function uploadSketch() {
    const file = document.getElementById("uploadSketch").files[0];
    const formData = new FormData();
    formData.append("sketch", file);

    try {
        const response = await fetch("http://localhost:5000/upload_sketch", {
            method: "POST",
            body: formData,
        });
        const data = await response.json();
        if (data.refined_image) {
            document.getElementById("result").innerHTML = `<img src="data:image/png;base64,${data.refined_image}" alt="Refined Sketch"/>`;
        } else {
            document.getElementById("result").innerText = data.message;
        }
    } catch (error) {
        console.error("Error uploading sketch:", error);
    }
}

async function generateStory() {
    const prompt = document.getElementById("storyPrompt").value;

    try {
        const response = await fetch("http://localhost:5000/generate_dialogue", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ prompt }),
        });
        const data = await response.json();
        document.getElementById("result").innerText = data.response;
    } catch (error) {
        console.error("Error generating story:", error);
    }
}
