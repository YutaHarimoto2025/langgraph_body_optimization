// bodyopt doclinks: @docs/... の参照を VS Code のエディタタブで開く拡張。
//
// 1. HTML プレビュー (doclinks.openPreview): HTML を webview に表示し、
//    ページ内リンク (.md/.html) のクリックを横取りして showTextDocument で開く。
//    既に開いているファイルならそのタブへフォーカスする (preview: false)。
// 2. DocumentLinkProvider: エディタ本文の @docs/... を Ctrl+クリックで開ける。
// 3. configurationDefaults: markdown.preview.openMarkdownLinks = inEditor
//    (Markdown preview のリンクもエディタタブで開く)。
"use strict";

const vscode = require("vscode");
const fs = require("fs");
const path = require("path");

const DOC_PATTERN = /@(docs\/[A-Za-z0-9_\-\/.]+?\.(?:md|html))/g;

function workspaceRoot(uri) {
  const folder = vscode.workspace.getWorkspaceFolder(uri);
  return folder ? folder.uri.fsPath : path.dirname(uri.fsPath);
}

/** webview 内のリンク URL を実ファイルパスへ戻す */
function webviewUrlToFsPath(url) {
  try {
    const parsed = vscode.Uri.parse(url);
    if (parsed.scheme === "file") return parsed.fsPath;
    // vscode-webview-resource://<uuid>/<abs path>
    return decodeURIComponent(parsed.path);
  } catch (_) {
    return null;
  }
}

class DocLinkProvider {
  provideDocumentLinks(document) {
    const text = document.getText();
    const links = [];
    let m;
    DOC_PATTERN.lastIndex = 0;
    while ((m = DOC_PATTERN.exec(text)) !== null) {
      const rel = m[1];
      // コードブロック・インラインコード内は除外 (行内の ` 数を数える簡易判定)
      const lineStart = text.lastIndexOf("\n", m.index) + 1;
      const lineEnd = text.indexOf("\n", m.index);
      const line = text.slice(lineStart, lineEnd === -1 ? undefined : lineEnd);
      if ((line.match(/`/g) || []).length % 2 === 1) continue;
      const root = workspaceRoot(document.uri);
      const abs = path.join(root, rel);
      if (!fs.existsSync(abs)) continue;
      const start = document.positionAt(m.index);
      const end = document.positionAt(m.index + m[0].length);
      const link = new vscode.DocumentLink(new vscode.Range(start, end), vscode.Uri.file(abs));
      link.tooltip = rel;
      links.push(link);
    }
    return links;
  }
}

async function openFile(fsPath) {
  try {
    const doc = await vscode.workspace.openTextDocument(vscode.Uri.file(fsPath));
    await vscode.window.showTextDocument(doc, { preview: false, viewColumn: vscode.ViewColumn.Active });
  } catch (err) {
    vscode.window.showErrorMessage(`doclinks: ファイルを開けませんでした: ${fsPath}`);
  }
}

function buildPreviewHtml(webview, sourcePath, source) {
  const dir = path.dirname(sourcePath);
  const baseUri = webview.asWebviewUri(vscode.Uri.file(dir)).toString().replace(/\/?$/, "/");
  const baseTag = `<base href="${baseUri}">`;
  const bridge = `<script>
    (function () {
      const vscode = acquireVsCodeApi();
      document.addEventListener("click", function (e) {
        const a = e.target && e.target.closest ? e.target.closest("a[href]") : null;
        if (!a) return;
        const href = a.getAttribute("href");
        if (!href || href.startsWith("#")) return;
        if (/^(https?:|mailto:)/.test(href)) return;
        e.preventDefault();
        e.stopPropagation();
        vscode.postMessage({ type: "open", href: a.href });
      }, true);
    })();
  <\/script>`;
  // <head> に base と橋渡しスクリプトを注入する
  if (/<head[^>]*>/i.test(source)) {
    return source.replace(/<head([^>]*)>/i, (m) => m + baseTag + bridge);
  }
  return baseTag + bridge + source;
}

async function openPreview(commandUri) {
  let target = commandUri;
  if (!target && vscode.window.activeTextEditor) {
    target = vscode.window.activeTextEditor.document.uri;
  }
  if (!target) {
    vscode.window.showWarningMessage("doclinks: プレビューする HTML を選んでください。");
    return;
  }
  const fsPath = target.fsPath;
  const root = workspaceRoot(target);
  const webviewPanel = vscode.window.createWebviewPanel(
    "doclinks.preview",
    `preview: ${path.basename(fsPath)}`,
    vscode.ViewColumn.Beside,
    { enableScripts: true, localResourceRoots: [vscode.Uri.file(root)], retainContextWhenHidden: true }
  );
  webviewPanel.webview.html = buildPreviewHtml(webviewPanel.webview, fsPath, fs.readFileSync(fsPath, "utf8"));
  webviewPanel.webview.onDidReceiveMessage(async (msg) => {
    if (msg && msg.type === "open") {
      const target = webviewUrlToFsPath(msg.href);
      if (target && fs.existsSync(target)) await openFile(target);
    }
  });
}

function activate(context) {
  context.subscriptions.push(
    vscode.languages.registerDocumentLinkProvider(
      [{ language: "markdown" }, { language: "html" }],
      new DocLinkProvider()
    )
  );
  context.subscriptions.push(
    vscode.commands.registerCommand("doclinks.openPreview", openPreview)
  );
}

function deactivate() {}

module.exports = { activate, deactivate };
