/* HTML 操作状態の保存・復元 (localStorage)。
 *
 * 使い方 (AGENTS.md「HTML ドキュメントの操作状態」):
 *   1. 状態を保存したい入力群の祖先 (通常 form) に data-persist を付ける。
 *      属性値は省略可 (要素が form なら自動対象)。
 *   2. HTML 末尾にこのスクリプトを読み込む:
 *        <script src="assets/html-state.js" defer></script>
 *   3. reset ボタンを押すと保存内容も消える。
 *
 * 対象: checkbox / radio / text 系 input / textarea / select / range。
 * file input はブラウザの制約上、復元できないため対象外。
 * 保存キーは URL パス + コンテナ id + 要素名。別ページ・別ブラウザとは共有されない。
 */
(() => {
  const PREFIX = "htmlstate:";
  const pageKey = () => PREFIX + location.pathname;

  const containers = () =>
    [...document.querySelectorAll("[data-persist]")].filter(
      (el) => el instanceof HTMLElement
    );

  const inputKey = (container, el) => {
    const cid = container.id || container.getAttribute("name") || "";
    let name = el.name || el.id || "";
    // 同名の checkbox / radio グループは value まで含めて一意にする
    if (el instanceof HTMLInputElement && (el.type === "checkbox" || el.type === "radio") && el.value) {
      name += ":" + el.value;
    }
    if (!name) {
      // name も id も無い要素は文書順の位置で一意化する
      name = "ix" + [...container.querySelectorAll("input, textarea, select")].indexOf(el);
    }
    return `${cid}|${name}`;
  };

  const collect = () => {
    const data = {};
    for (const c of containers()) {
      for (const el of c.querySelectorAll("input, textarea, select")) {
        if (el instanceof HTMLInputElement && el.type === "file") continue;
        const k = inputKey(c, el);
        if (!k.endsWith("|")) {
          data[k] =
            el instanceof HTMLInputElement && (el.type === "checkbox" || el.type === "radio")
              ? { c: el.checked }
              : { v: el.value };
        }
      }
    }
    return data;
  };

  const save = () => {
    try {
      localStorage.setItem(pageKey(), JSON.stringify(collect()));
    } catch (_) {
      /* プライベートモード等で保存できない場合は何もしない */
    }
  };

  const restore = () => {
    let data = {};
    try {
      data = JSON.parse(localStorage.getItem(pageKey()) || "{}");
    } catch (_) {
      return;
    }
    for (const c of containers()) {
      for (const el of c.querySelectorAll("input, textarea, select")) {
        if (el instanceof HTMLInputElement && el.type === "file") continue;
        const rec = data[inputKey(c, el)];
        if (!rec) continue;
        if ("c" in rec && el instanceof HTMLInputElement && (el.type === "checkbox" || el.type === "radio")) {
          el.checked = rec.c;
          el.dispatchEvent(new Event("change", { bubbles: true }));
        } else if ("v" in rec) {
          el.value = rec.v;
          el.dispatchEvent(new Event("input", { bubbles: true }));
        }
      }
    }
  };

  document.addEventListener("input", (e) => {
    if (e.target instanceof Element && e.target.closest("[data-persist]")) save();
  });
  document.addEventListener("change", (e) => {
    if (e.target instanceof Element && e.target.closest("[data-persist]")) save();
  });
  // reset ボタンは保存内容も消す
  document.addEventListener(
    "reset",
    () => {
      try {
        localStorage.removeItem(pageKey());
      } catch (_) {}
    },
    true
  );

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", restore);
  } else {
    restore();
  }
})();
