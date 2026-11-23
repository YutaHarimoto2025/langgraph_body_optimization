// ページ内のフォーム入力（チェック・ラジオ・テキスト・選択）を localStorage に保存し、再読み込み時に復元する。
// 使い方: 保存したい要素を含む祖先（通常は form）に data-persist を付け、
//   <script src="assets/html-state.js" defer></script>
// を読み込む。保存先はブラウザのプロファイルで、HTML ファイル自体は書き換えない。
(() => {
  const init = () => {
    const scope = document.querySelector('[data-persist]') || document.querySelector('form');
    if (!scope) return;

    const key = 'html-state:' + location.pathname;
    const fields = [...scope.querySelectorAll('input, textarea, select')].filter(
      (el) => !['submit', 'reset', 'button', 'file', 'image'].includes(el.type)
    );

    // チェックボックスとラジオは同じ name を共有するため value まで含めて識別する。
    const idOf = (el, i) => {
      const base = el.name || el.id || '#' + i;
      return el.type === 'checkbox' || el.type === 'radio' ? base + '\t' + (el.value || i) : base;
    };
    const read = (el) =>
      el.type === 'checkbox' || el.type === 'radio' ? (el.checked ? '1' : '0') : el.value;

    function save() {
      const state = {};
      fields.forEach((el, i) => {
        state[idOf(el, i)] = read(el);
      });
      try {
        localStorage.setItem(key, JSON.stringify(state));
      } catch (_) {}
    }

    try {
      const state = JSON.parse(localStorage.getItem(key) || 'null');
      if (state) {
        fields.forEach((el, i) => {
          const value = state[idOf(el, i)];
          if (value === undefined) return;
          if (el.type === 'checkbox' || el.type === 'radio') el.checked = value === '1';
          else el.value = value;
        });
      }
    } catch (_) {}

    scope.addEventListener('change', save);
    scope.addEventListener('input', (e) => {
      if (e.target.matches('textarea, select, input:not([type=checkbox]):not([type=radio])')) save();
    });
    scope.addEventListener('reset', () => {
      try {
        localStorage.removeItem(key);
      } catch (_) {}
      setTimeout(save, 0);
    });
  };

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();
})();
