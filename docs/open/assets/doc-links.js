/* ドキュメント間リンク (`target="bodyopt-doc"`) を名前付きウィンドウで開く共通スクリプト。
 *
 * ブラウザの Ctrl+クリック・中クリックは target を無視して新しいタブを増やすため、
 * クリックを横取りして window.open(url, "bodyopt-doc") を呼ぶ。同じ名前のタブが
 * 既にあればそこへ読み込み (ジャンプ)、無ければ新しいタブで開く。
 *
 * 使い方: HTML 末尾で <script src="assets/doc-links.js" defer></script> を読み込む。
 * リンクは <a href="../agents/sim/physics.md" target="bodyopt-doc">@docs/agents/sim/physics.md</a>。
 */
(() => {
  const TARGET = "bodyopt-doc";
  const openNamed = (url) => {
    const win = window.open(url, TARGET);
    if (win) win.focus();
  };
  document.addEventListener("click", (e) => {
    const link = e.target instanceof Element ? e.target.closest(`a[target="${TARGET}"]`) : null;
    if (!(link instanceof HTMLAnchorElement)) return;
    e.preventDefault();
    openNamed(link.href);
  });
  document.addEventListener("auxclick", (e) => {
    const link = e.target instanceof Element ? e.target.closest(`a[target="${TARGET}"]`) : null;
    if (!(link instanceof HTMLAnchorElement) || e.button !== 1) return;
    e.preventDefault();
    openNamed(link.href);
  });
})();
