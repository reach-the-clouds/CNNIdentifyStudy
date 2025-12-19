let API_BASE = null;
const HOSTS = ["127.0.0.1", "127.0.0.2", "localhost"];
const PORTS = [8000, 8001, 8002, 9000, 8080, 5000];
const fileInput = document.getElementById("fileInput");
const previewImg = document.getElementById("previewImg");
const predictBtn = document.getElementById("predictBtn");
const resultText = document.getElementById("resultText");

let idx2name = {};
predictBtn.disabled = true;
async function detectApiBase() {
  for (const h of HOSTS) {
    for (const p of PORTS) {
      try {
        const r = await fetch(`http://${h}:${p}/labels`);
        if (r.ok) {
          const data = await r.json();
          const names = data.names || [];
          idx2name = {};
          names.forEach((n, i) => { idx2name[i] = n; });
          API_BASE = `http://${h}:${p}`;
          return true;
        }
      } catch (e) {}
    }
  }
  return false;
}
(async () => { const ok = await detectApiBase(); predictBtn.disabled = !ok; })();

fileInput.addEventListener("change", () => {
  const file = fileInput.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = e => { previewImg.src = e.target.result; resultText.textContent = ""; };
  reader.readAsDataURL(file);
});

predictBtn.addEventListener("click", async () => {
  const file = fileInput.files[0];
  if (!file) { resultText.textContent = "请先选择图片"; return; }
  predictBtn.disabled = true;
  resultText.textContent = "识别中...";
  try {
    const fd = new FormData();
    fd.append("file", file);
    if (!API_BASE) { resultText.textContent = "后端未连接"; return; }
    const resp = await fetch(`${API_BASE}/predict`, { method: "POST", body: fd });
    if (!resp.ok) throw new Error(`接口错误 ${resp.status}`);
    const data = await resp.json();
    let name = data.label_name || data.class_name;
    if (!name) {
      const idx = data.label_index ?? data.class_index ?? data.label ?? data.class;
      if (idx != null && idx2name[idx] != null) name = idx2name[idx];
    }
    resultText.textContent = name ? `识别结果：${name}` : "无法解析接口返回";
  } catch (e) {
    resultText.textContent = `错误：${e.message}`;
  } finally {
    predictBtn.disabled = false;
  }
});
