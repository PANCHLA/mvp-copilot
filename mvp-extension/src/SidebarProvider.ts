import * as vscode from 'vscode';
const fetch = require('node-fetch');

export class SidebarProvider implements vscode.WebviewViewProvider {
    _view?: vscode.WebviewView;
    _doc?: vscode.TextDocument;

    constructor(private readonly _extensionUri: vscode.Uri) { }

    public resolveWebviewView(webviewView: vscode.WebviewView) {
        this._view = webviewView;

        webviewView.webview.options = {
            enableScripts: true,
            localResourceRoots: [this._extensionUri],
        };

        webviewView.webview.html = this._getHtmlForWebview(webviewView.webview);

        webviewView.webview.onDidReceiveMessage(async (data) => {
            switch (data.type) {
                case 'ask': {
                    if (!data.value) {
                        return;
                    }
                    this._askBackend(data.value, data.mode);
                    break;
                }
                case 'onInfo': {
                    if (!data.value) {
                        return;
                    }
                    vscode.window.showInformationMessage(data.value);
                    break;
                }
                case 'onError': {
                    if (!data.value) {
                        return;
                    }
                    vscode.window.showErrorMessage(data.value);
                    break;
                }
            }
        });
    }

    private async _askBackend(query: string, mode: string) {
        if (!this._view) {
            return;
        }

        // Get active file context
        let context = "";
        const editor = vscode.window.activeTextEditor;
        if (editor && editor.document.uri.scheme === 'file') {
            const fileName = editor.document.fileName;
            const fileContent = editor.document.getText();
            context = `Active File: ${fileName}\n\n${fileContent}`;
        }

        try {
            const response = await fetch('http://localhost:8000/query', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    q: query,
                    k: 5,
                    mode: mode, // 'local' or 'openrouter'
                    context: context
                })
            });

            if (!response.ok) {
                throw new Error(`API Error: ${response.statusText}`);
            }

            const data = await response.json();
            const answer = data.answer || 'No answer provided.';
            const sources = data.sources || [];

            this._view.webview.postMessage({
                type: 'addResponse',
                value: answer,
                sources: sources
            });

        } catch (error: any) {
            this._view.webview.postMessage({
                type: 'addError',
                value: `Error: ${error.message}`
            });
        }
    }

    public revive(panel: vscode.WebviewView) {
        this._view = panel;
    }

    private _getHtmlForWebview(webview: vscode.Webview) {
        const styleResetUri = webview.asWebviewUri(vscode.Uri.joinPath(this._extensionUri, "media", "reset.css"));
        const styleVSCodeUri = webview.asWebviewUri(vscode.Uri.joinPath(this._extensionUri, "media", "vscode.css"));

        const nonce = getNonce();

        return `<!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>MVP Copilot Chat</title>
                <style>
                    :root {
                        --container-paddding: 20px;
                        --input-padding: 10px;
                        --button-bg: var(--vscode-button-background);
                        --button-fg: var(--vscode-button-foreground);
                        --button-hover: var(--vscode-button-hoverBackground);
                        --chat-bg: var(--vscode-editor-background);
                        --user-msg-bg: var(--vscode-button-background);
                        --user-msg-fg: var(--vscode-button-foreground);
                        --bot-msg-bg: var(--vscode-editor-inactiveSelectionBackground);
                        --border-radius: 6px;
                    }
                    body {
                        font-family: var(--vscode-font-family);
                        padding: 0;
                        margin: 0;
                        background-color: var(--chat-bg);
                        color: var(--vscode-editor-foreground);
                        height: 100vh;
                        display: flex;
                        flex-direction: column;
                    }
                    .header {
                        padding: 15px 20px;
                        border-bottom: 1px solid var(--vscode-widget-border);
                        display: flex;
                        justify-content: space-between;
                        align-items: center;
                        background: var(--vscode-sideBar-background);
                    }
                    .header h2 {
                        margin: 0;
                        font-size: 14px;
                        font-weight: 600;
                        text-transform: uppercase;
                        letter-spacing: 0.5px;
                    }
                    .model-select {
                        background: var(--vscode-dropdown-background);
                        color: var(--vscode-dropdown-foreground);
                        border: 1px solid var(--vscode-dropdown-border);
                        padding: 4px 8px;
                        border-radius: 4px;
                        font-size: 12px;
                        outline: none;
                    }
                    .messages {
                        flex: 1;
                        overflow-y: auto;
                        padding: 20px;
                        display: flex;
                        flex-direction: column;
                        gap: 16px;
                    }
                    .message {
                        max-width: 85%;
                        padding: 12px 16px;
                        border-radius: var(--border-radius);
                        font-size: 13px;
                        line-height: 1.5;
                        position: relative;
                        word-wrap: break-word;
                    }
                    .message.user {
                        align-self: flex-end;
                        background: var(--user-msg-bg);
                        color: var(--user-msg-fg);
                        border-bottom-right-radius: 2px;
                    }
                    .message.bot {
                        align-self: flex-start;
                        background: var(--bot-msg-bg);
                        border-bottom-left-radius: 2px;
                    }
                    .input-area {
                        padding: 20px;
                        border-top: 1px solid var(--vscode-widget-border);
                        background: var(--vscode-sideBar-background);
                        display: flex;
                        flex-direction: column;
                        gap: 10px;
                    }
                    textarea {
                        background: var(--vscode-input-background);
                        color: var(--vscode-input-foreground);
                        border: 1px solid var(--vscode-input-border);
                        border-radius: var(--border-radius);
                        padding: 10px;
                        resize: none;
                        min-height: 60px;
                        font-family: inherit;
                        font-size: 13px;
                        outline: none;
                    }
                    textarea:focus {
                        border-color: var(--vscode-focusBorder);
                    }
                    button {
                        background: var(--button-bg);
                        color: var(--button-fg);
                        border: none;
                        border-radius: var(--border-radius);
                        padding: 8px 16px;
                        cursor: pointer;
                        font-weight: 500;
                        align-self: flex-end;
                        transition: background 0.2s;
                    }
                    button:hover {
                        background: var(--button-hover);
                    }
                    .sources {
                        margin-top: 10px;
                        padding-top: 10px;
                        border-top: 1px solid rgba(128, 128, 128, 0.2);
                        font-size: 11px;
                    }
                    .sources-title {
                        font-weight: 600;
                        margin-bottom: 4px;
                        opacity: 0.8;
                    }
                    .source-link {
                        display: block;
                        color: var(--vscode-textLink-foreground);
                        text-decoration: none;
                        margin-bottom: 2px;
                        white-space: nowrap;
                        overflow: hidden;
                        text-overflow: ellipsis;
                    }
                    .source-link:hover {
                        text-decoration: underline;
                    }
                    pre {
                        background: rgba(0, 0, 0, 0.2);
                        padding: 10px;
                        border-radius: 4px;
                        overflow-x: auto;
                        margin: 8px 0;
                    }
                    code {
                        font-family: var(--vscode-editor-font-family, monospace);
                        font-size: 0.9em;
                    }
                    /* Scrollbar styling */
                    ::-webkit-scrollbar {
                        width: 8px;
                    }
                    ::-webkit-scrollbar-track {
                        background: transparent;
                    }
                    ::-webkit-scrollbar-thumb {
                        background: var(--vscode-scrollbarSlider-background);
                        border-radius: 4px;
                    }
                    ::-webkit-scrollbar-thumb:hover {
                        background: var(--vscode-scrollbarSlider-hoverBackground);
                    }
                </style>
            </head>
            <body>
                <div class="header">
                    <h2>MVP Copilot</h2>
                    <select id="model-selector" class="model-select">
                        <option value="openrouter">OpenRouter</option>
                        <option value="local">Local</option>
                    </select>
                </div>
                <div class="messages" id="messages">
                    <div class="message bot">Hello! I'm MVP Copilot. How can I help you with your code today?</div>
                </div>
                <div class="input-area">
                    <textarea id="prompt-input" placeholder="Ask something... (Cmd+Enter to send)"></textarea>
                    <button id="send-btn">Send</button>
                </div>
                <script nonce="${nonce}">
                    const vscode = acquireVsCodeApi();
                    const messagesContainer = document.getElementById('messages');
                    const input = document.getElementById('prompt-input');
                    const sendBtn = document.getElementById('send-btn');
                    const modelSelector = document.getElementById('model-selector');

                    // Load state
                    const previousState = vscode.getState();
                    if (previousState && previousState.model) {
                        modelSelector.value = previousState.model;
                    }

                    modelSelector.addEventListener('change', () => {
                        vscode.setState({ model: modelSelector.value });
                    });

                    function addMessage(text, sender, sources = []) {
                        const div = document.createElement('div');
                        div.className = 'message ' + sender;
                        
                        let content = text;
                        // Basic markdown
                        content = content.replace(/\\*\\*(.*?)\\*\\*/g, '<b>$1</b>');
                        content = content.replace(/\\\`(.*?)\\\`/g, '<code>$1</code>');
                        content = content.replace(/\\n/g, '<br>');
                        
                        // Handle code blocks
                        if (content.includes('\`\`\`')) {
                             content = content.replace(/\`\`\`([\\s\\S]*?)\`\`\`/g, '<pre><code>$1</code></pre>');
                        }

                        div.innerHTML = content;

                        if (sources && sources.length > 0) {
                            const sourcesDiv = document.createElement('div');
                            sourcesDiv.className = 'sources';
                            const title = document.createElement('div');
                            title.className = 'sources-title';
                            title.innerText = 'Sources:';
                            sourcesDiv.appendChild(title);
                            
                            sources.forEach(s => {
                                const link = document.createElement('a');
                                link.className = 'source-link';
                                // Show file name only for brevity, full path on hover
                                const fileName = s.file.split(/[\\\\/]/).pop();
                                link.innerText = fileName + ':' + s.start + '-' + s.end;
                                link.title = s.file;
                                link.href = '#'; 
                                sourcesDiv.appendChild(link);
                            });
                            div.appendChild(sourcesDiv);
                        }

                        messagesContainer.appendChild(div);
                        messagesContainer.scrollTop = messagesContainer.scrollHeight;
                    }

                    function sendMessage() {
                        const text = input.value.trim();
                        if (!text) return;
                        
                        addMessage(text, 'user');
                        input.value = '';
                        
                        vscode.postMessage({
                            type: 'ask',
                            value: text,
                            mode: modelSelector.value
                        });

                        // Add loading indicator
                        const loadingId = 'loading-' + Date.now();
                        const loadingDiv = document.createElement('div');
                        loadingDiv.id = loadingId;
                        loadingDiv.className = 'message bot';
                        loadingDiv.innerHTML = '<i>Thinking...</i>';
                        messagesContainer.appendChild(loadingDiv);
                    }

                    sendBtn.addEventListener('click', sendMessage);

                    input.addEventListener('keydown', (e) => {
                        if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
                            e.preventDefault();
                            sendMessage();
                        }
                    });

                    window.addEventListener('message', event => {
                        const message = event.data;
                        
                        // Remove loading indicators
                        const loadings = document.querySelectorAll('.message.bot');
                        loadings.forEach(el => {
                            if (el.innerHTML === '<i>Thinking...</i>') {
                                el.remove();
                            }
                        });

                        switch (message.type) {
                            case 'addResponse':
                                addMessage(message.value, 'bot', message.sources);
                                break;
                            case 'addError':
                                addMessage(message.value, 'bot');
                                break;
                        }
                    });
                </script>
            </body>
            </html>`;
    }
}

function getNonce() {
    let text = '';
    const possible = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
    for (let i = 0; i < 32; i++) {
        text += possible.charAt(Math.floor(Math.random() * possible.length));
    }
    return text;
}
