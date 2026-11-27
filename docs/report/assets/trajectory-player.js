"use strict";

for (const figure of document.querySelectorAll("[data-trajectory-player]")) {
  const { times, pos, spring, contact, edges } = JSON.parse(figure.querySelector("script[type='application/json']").textContent);
  const canvas = figure.querySelector("canvas");
  const ctx = canvas.getContext("2d");
  const button = figure.querySelector("button");
  const seek = figure.querySelector("input[type='range']");
  const clock = figure.querySelector("output");
  const end = times.at(-1);
  const centers = pos.map((points) => points.reduce(
    ([x, y], [px, py]) => [x + px / points.length, y + py / points.length], [0, 0],
  ));
  const width = canvas.width;
  const height = canvas.height;
  let minX = Infinity, maxX = -Infinity, minY = 0, maxY = 0;
  for (const points of pos) {
    for (const [x, y] of points) {
      minX = Math.min(minX, x);
      maxX = Math.max(maxX, x);
      minY = Math.min(minY, y);
      maxY = Math.max(maxY, y);
    }
  }
  const scale = Math.min(width / (maxX - minX + 0.8), height / (maxY - minY + 0.8));
  const left = (minX + maxX - width / scale) / 2;
  const bottom = (minY + maxY - height / scale) / 2;
  const right = left + width / scale;
  const top = bottom + height / scale;
  const x = (px) => (px - left) * scale;
  const y = (py) => height - (py - bottom) * scale;
  seek.max = String(end);
  let time = 0;
  let start = 0;
  let animation = 0;

  function draw() {
    const w = Math.round(canvas.clientWidth * devicePixelRatio);
    const h = Math.round(canvas.clientHeight * devicePixelRatio);
    if (!w || !h) return;
    if (canvas.width !== w) canvas.width = w;
    if (canvas.height !== h) canvas.height = h;
    ctx.setTransform(w / width, 0, 0, h / height, 0, 0);
    const i = Math.max(0, times.findLastIndex((t) => t <= time));
    const [comX] = centers[i];
    const line = (points, color, weight) => {
      ctx.beginPath();
      points.forEach(([px, py], j) => j ? ctx.lineTo(x(px), y(py)) : ctx.moveTo(x(px), y(py)));
      ctx.lineWidth = weight;
      ctx.strokeStyle = color;
      ctx.stroke();
    };
    const dot = (px, py, radius, color) => {
      ctx.beginPath();
      ctx.arc(x(px), y(py), radius, 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.fill();
    };
    ctx.fillStyle = "#26282c";
    ctx.fillRect(0, 0, width, height);
    for (let gx = Math.ceil(left); gx <= right; gx++) line([[gx, bottom], [gx, top]], "#34383d", 1);
    for (let gy = Math.ceil(bottom); gy <= top; gy++) line([[left, gy], [right, gy]], "#34383d", 1);
    line([[left, 0], [right, 0]], "#8a9099", 3);
    line(centers.slice(Math.max(0, i - 45), i + 1), "#9eceff", 2);
    const maxForce = Math.max(...spring[i].map(Math.abs)) || 1;
    edges.forEach(([a, b], e) => {
      const force = spring[i][e] / maxForce;
      const hue = force < 0 ? 170 - 55 * force : 170 * (1 - force);
      line([pos[i][a], pos[i][b]], "hsl(" + hue + " 74% 65%)", 5 + 2 * Math.abs(force));
    });
    pos[i].forEach(([px, py], j) => {
      if (contact[i][j]) dot(px, py, 10, "#e0b685");
      dot(px, py, 6, "#e8ecef");
    });
    ctx.fillStyle = "#e8ecef";
    ctx.font = "16px system-ui";
    ctx.fillText("COM dx = " + (comX - centers[0][0]).toFixed(3) + " m", 14, 24);
  }

  function render(seconds) {
    time = seconds;
    seek.value = String(seconds);
    clock.value = seconds.toFixed(2) + " / " + end.toFixed(2) + " s";
    draw();
  }

  function pause() {
    cancelAnimationFrame(animation);
    animation = 0;
    button.textContent = "再生";
  }

  function tick(now) {
    render(Math.min(Math.max(0, (now - start) / 1000), end));
    if (time === end) pause();
    else animation = requestAnimationFrame(tick);
  }

  button.addEventListener("click", () => {
    if (animation) {
      pause();
      return;
    }
    if (time === end) render(0);
    start = performance.now() - time * 1000;
    button.textContent = "一時停止";
    animation = requestAnimationFrame(tick);
  });
  seek.addEventListener("input", () => {
    render(seek.valueAsNumber);
    if (animation) start = performance.now() - time * 1000;
  });
  new ResizeObserver(draw).observe(canvas);
  render(0);
  button.disabled = false;
  seek.disabled = false;
}
