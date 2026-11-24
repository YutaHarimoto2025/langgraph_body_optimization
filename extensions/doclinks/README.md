# bodyopt doclinks (VS Code 拡張)

@docs/... への参照を VS Code のエディタタブで開くための拡張。

## 機能

- **HTML プレビュー**: エクスプローラで HTML を右クリック →「bodyopt: HTML プレビューを開く」。プレビュー内の `.md`/`.html` リンクをクリックすると、外部ブラウザではなく **VS Code のエディタタブ** で開く。既に開いていればそのタブへフォーカスする。
- **Markdown preview**: `markdown.preview.openMarkdownLinks` を `inEditor` に設定するので、preview のリンクはエディタタブで開く。
- **DocumentLink**: エディタ本文の `@docs/...` を Ctrl+クリックで開ける。

## インストール (開発用)

```bash
scripts/install_doclinks.sh
```

VS Code で Reload Window (Ctrl+Shift+P → Developer: Reload Window) する。

## 制限

- HTML プレビューは webview なので、外部 CDN のスクリプト (three.js など) と一部の CSS は動かない。報告書のような自己完結 HTML を想定する。
- 動画は APNG (img) を推奨。H.264 の video は webview で再生できないことがある。
