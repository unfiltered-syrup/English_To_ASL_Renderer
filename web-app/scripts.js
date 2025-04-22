const sentenceInput = document.getElementById('sentence-input');
const translateButton = document.getElementById('translate-button');
const gestureDisplay = document.getElementById('gesture-display');
const statusMessage = document.getElementById('status-message');

const API_URL = 'http://localhost:5000';

translateButton.addEventListener('click', translateSentence);

async function translateSentence() {
    const sentence = sentenceInput.value.trim();
    if (!sentence) {
        statusMessage.textContent = 'Please enter some text.';
        gestureDisplay.src = "";
        return;
    }

    statusMessage.textContent = 'Processing sentence...';
    gestureDisplay.src = "";

    try {
        const response = await fetch(`${API_URL}/translate`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ sentence: sentence }),
        });

        const data = await response.json();

        if (data.error) {
             statusMessage.textContent = `Error: ${data.error}`;
        } else if (data.glosses_to_render && data.glosses_to_render.length > 0) {
            await renderGlossesSequentially(data.glosses_to_render);
            statusMessage.textContent = 'Translation complete.';
        } else {
            statusMessage.textContent = 'No gestures found or needed for the input.';
        }

    } catch (error) {
        console.error('Error during translation:', error);
        statusMessage.textContent = 'Failed to translate. Check console or server.';
        gestureDisplay.src = "";
    }
}

async function renderGlossesSequentially(glosses) {
    gestureDisplay.src = "";
    await sleep(15);

    for (const gloss of glosses) {
        statusMessage.textContent = `Rendering: ${gloss}`;
        try {
            const response = await fetch(`${API_URL}/get_gesture_frames/${gloss}`);
            if (!response.ok) {
                 console.warn(`Could not fetch frames for "${gloss}", status: ${response.status}. Skipping.`);
                 statusMessage.textContent = `Skipping: ${gloss} (not found)`;
                 await sleep(100);
                 continue;
            }

            const data = await response.json();

            if (data.error) {
                console.warn(`Server error fetching frames for "${gloss}": ${data.error}. Skipping.`);
                statusMessage.textContent = `Skipping: ${gloss} (error)`;
                await sleep(100);
                continue;
            }


            if (data.frames && data.frames.length > 0) {
                for (const frameData of data.frames) {
                    gestureDisplay.src = `data:image/jpeg;base64,${frameData}`;
                    await sleep(40);
                }
                await sleep(100);
            } else {
                 console.warn(`No frames returned for "${gloss}". Skipping.`);
                 statusMessage.textContent = `Skipping: ${gloss} (no frames)`;
                 await sleep(100);
            }

        } catch (error) {
            console.error(`Error fetching/rendering frames for "${gloss}":`, error);
            statusMessage.textContent = `Error rendering: ${gloss}`;
            await sleep(100);

        }
    }
     gestureDisplay.src = "";
}


function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}