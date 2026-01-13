import * as vscode from 'vscode';
import { SidebarProvider } from './SidebarProvider';
const fetch = require('node-fetch');

export function activate(context: vscode.ExtensionContext) {
    console.log('MVP Copilot extension is now active!');

    const sidebarProvider = new SidebarProvider(context.extensionUri);
    context.subscriptions.push(
        vscode.window.registerWebviewViewProvider(
            "mvpCopilot.sidebar",
            sidebarProvider
        )
    );

    let disposable = vscode.commands.registerCommand('mvpCopilot.ask', () => {
        vscode.commands.executeCommand('workbench.view.extension.mvp-copilot-sidebar');
    });

    context.subscriptions.push(disposable);
}



export function deactivate() { }
