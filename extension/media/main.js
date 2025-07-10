// main.js

//@ts-check

/**
 * @type {Array<{ role: 'user' | 'assistant' | 'tool', content: string }>}
 */
let chatHistory = [];
let isProcessing = false;
const vscode = acquireVsCodeApi();

function addMessage(text, role) {
    const chatContainer = document.getElementById('chat-container');
    const messageDiv = document.createElement('div');

    if (role === 'user') {
        messageDiv.className = 'user-message';
    } else if (role === 'assistant') {
        messageDiv.className = 'bot-message';
    } else if (role === 'tool') {
        messageDiv.className = 'tool-message';
    }

    const iconSpan = document.createElement('span');
    iconSpan.className = 'message-icon';

    if (role === 'user') {
        iconSpan.textContent = '🧑';
    } else if (role === 'assistant') {
        iconSpan.textContent = '🤖';
    } else if (role === 'tool') {
        iconSpan.textContent = '🛠️';
    }

    const textSpan = document.createElement('span');
    textSpan.className = 'message-text';
    textSpan.textContent = text;
    //textSpan.innerHTML = marked.parse(text);

    messageDiv.appendChild(iconSpan);
    messageDiv.appendChild(textSpan);
    chatContainer.appendChild(messageDiv);
    chatContainer.scrollTop = chatContainer.scrollHeight;

    chatHistory.push({
        role: role,
        content: text
    });
}

async function sendMessage() {
    if (isProcessing) return;

    const userInput = document.getElementById('user-input');
    const message = userInput.value.trim();
    if (!message) {
        isProcessing = false;
        return;
    }

    addMessage(message, 'user');
    userInput.value = '';

    vscode.postMessage({
        command: 'sendMessage',
        value: JSON.stringify({ 
            request: message,
            history: chatHistory
        }) 
    });
}

window.addEventListener('message', (event) => {
    const response = event.data;
    if (response.command === 'getResponse') {
        const { content, history } = response.payload;
        if (history && history.length > 0)
        {
            const lastMessage = history[history.length - 1];
            if (lastMessage.role === 'tool') {
                addMessage(lastMessage.content, 'tool');
            }
        }
        if (content) {
            addMessage(content, 'assistant');
        }

        isProcessing = false;
    } else if (response.command === 'errorMessage') {
        addMessage('Ошибка: ' + response.text, false);
    }
});

document.getElementById('send-button').addEventListener('click', sendMessage);

document.addEventListener('key', (event) => {
    if (event.code === 'Enter')
    {
        sendMessage();
    }
});