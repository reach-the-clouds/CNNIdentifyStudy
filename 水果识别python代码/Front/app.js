let API_BASE = null;
const HOSTS = ["127.0.0.1", "127.0.0.2", "localhost"];
const PORTS = [8001, 8002, 8003, 9001, 8081, 5001];
const fileInput = document.getElementById("fileInput");
const previewImg = document.getElementById("previewImg");
const predictBtn = document.getElementById("predictBtn");
const resultText = document.getElementById("resultText");
const resultTags = document.getElementById("resultTags");
const nutritionCard = document.getElementById("nutritionCard");
const nutritionName = document.getElementById("nutritionName");
const nutritionGrade = document.getElementById("nutritionGrade");
const nutCal = document.getElementById("nut_cal");
const nutCarbs = document.getElementById("nut_carbs");
const nutFiber = document.getElementById("nut_fiber");
const nutVitC = document.getElementById("nut_vitc");
const nutProtein = document.getElementById("nut_protein");
const nutFat = document.getElementById("nut_fat");
const nutSugar = document.getElementById("nut_sugar");
const nutPotassium = document.getElementById("nut_potassium");
const nutCalcium = document.getElementById("nut_calcium");
const nutIron = document.getElementById("nut_iron");
const nutSodium = document.getElementById("nut_sodium");

let idx2name = {};
let selectionToken = 0;
const en2zh = {
    "apple": "苹果",
    "banana": "香蕉",
    "beetroot": "甜菜根",
    "cabbage": "卷心菜",
    "capsicum": "灯笼椒",
    "carrot": "胡萝卜",
    "cauliflower": "花椰菜",
    "chilli pepper": "辣椒",
    "corn": "玉米",
    "cucumber": "黄瓜",
    "eggplant": "茄子",
    "garlic": "大蒜",
    "ginger": "姜",
    "grape": "葡萄",
    "grapes": "葡萄",
    "jalepeno": "墨西哥辣椒",
    "kiwi": "猕猴桃",
    "lemon": "柠檬",
    "lettuce": "生菜",
    "mango": "芒果",
    "onion": "洋葱",
    "orange": "橙子",
    "paprika": "红椒",
    "pear": "梨",
    "peas": "豌豆",
    "pineapple": "菠萝",
    "pomegranate": "石榴",
    "potato": "土豆",
    "raddish": "小红萝卜",
    "soy beans": "黄豆",
    "spinach": "菠菜",
    "sweetcorn": "甜玉米",
    "sweetpotato": "红薯",
    "tomato": "西红柿",
    "turnip": "芜菁",
    "watermelon": "西瓜",
    "hami": "哈密瓜",
    "hami melon": "哈密瓜",
    "cantaloupe": "哈密瓜",
    "muskmelon": "香瓜",
    "honeydew": "蜜瓜",
    "honeydew melon": "蜜瓜",
    "papaya": "木瓜",
    "strawberry": "草莓",
    "blueberry": "蓝莓",
    "cherry": "樱桃",
    "peach": "桃子",
    "plum": "李子",
    "apricot": "杏",
    "dragonfruit": "火龙果",
    "pitaya": "火龙果",
    "lychee": "荔枝",
    "longan": "龙眼"
};
const aliasMap = { "hami melon": "hami", "cantaloupe": "hami", "muskmelon": "hami", "honeydew melon": "honeydew", "pitaya": "dragonfruit" };
predictBtn.disabled = true;
const nutritionData = {
    "apple": {calories: 52, carbs: 14, fiber: 2.4, vitaminC: 4.6, protein: 0.3, fat: 0.2, sugar: 10, potassium: 107, calcium: 6, iron: 0.12, sodium: 1},
    "banana": {calories: 89, carbs: 23, fiber: 2.6, vitaminC: 8.7, protein: 1.1, fat: 0.3, sugar: 12.2, potassium: 358, calcium: 5, iron: 0.26, sodium: 1},
    "beetroot": {calories: 43, carbs: 10, fiber: 2.8, vitaminC: 4.9},
    "cabbage": {calories: 25, carbs: 6, fiber: 2.5, vitaminC: 36.6},
    "capsicum": {calories: 31, carbs: 6, fiber: 2.1, vitaminC: 80.4},
    "carrot": {calories: 41, carbs: 9.6, fiber: 2.8, vitaminC: 5.9},
    "cauliflower": {calories: 25, carbs: 5, fiber: 2, vitaminC: 48.2},
    "chilli pepper": {calories: 40, carbs: 9, fiber: 1.5, vitaminC: 143},
    "corn": {calories: 86, carbs: 19, fiber: 2.7, vitaminC: 6.8},
    "cucumber": {calories: 15, carbs: 3.6, fiber: 0.5, vitaminC: 2.8},
    "eggplant": {calories: 25, carbs: 6, fiber: 3, vitaminC: 2.2},
    "garlic": {calories: 149, carbs: 33, fiber: 2.1, vitaminC: 31.2},
    "ginger": {calories: 80, carbs: 18, fiber: 2, vitaminC: 5},
    "grape": {calories: 69, carbs: 18, fiber: 0.9, vitaminC: 3.2, protein: 0.6, fat: 0.4, sugar: 15.5, potassium: 191, calcium: 10, iron: 0.36, sodium: 2},
    "grapes": {calories: 69, carbs: 18, fiber: 0.9, vitaminC: 3.2, protein: 0.6, fat: 0.4, sugar: 15.5, potassium: 191, calcium: 10, iron: 0.36, sodium: 2},
    "jalepeno": {calories: 29, carbs: 6.5, fiber: 2.8, vitaminC: 118.6},
    "kiwi": {calories: 61, carbs: 15, fiber: 3, vitaminC: 92.7, protein: 1.1, fat: 0.5, sugar: 9, potassium: 312, calcium: 34, iron: 0.31, sodium: 2},
    "lemon": {calories: 29, carbs: 9, fiber: 2.8, vitaminC: 53, protein: 1.1, fat: 0.3, sugar: 2.5, potassium: 138, calcium: 26, iron: 0.6, sodium: 2},
    "lettuce": {calories: 15, carbs: 2.9, fiber: 1.3, vitaminC: 9.2},
    "mango": {calories: 60, carbs: 15, fiber: 1.6, vitaminC: 36.4, protein: 0.8, fat: 0.4, sugar: 14, potassium: 168, calcium: 11, iron: 0.16, sodium: 1},
    "onion": {calories: 40, carbs: 9.3, fiber: 1.7, vitaminC: 7.4},
    "orange": {calories: 47, carbs: 12, fiber: 2.4, vitaminC: 53.2, protein: 0.9, fat: 0.2, sugar: 8.5, potassium: 181, calcium: 40, iron: 0.1, sodium: 0},
    "paprika": {calories: 31, carbs: 6, fiber: 2.1, vitaminC: 127},
    "pear": {calories: 57, carbs: 15, fiber: 3.1, vitaminC: 4.3, protein: 0.4, fat: 0.1, sugar: 9.8, potassium: 119, calcium: 9, iron: 0.17, sodium: 1},
    "peas": {calories: 81, carbs: 14, fiber: 5.1, vitaminC: 40},
    "pineapple": {calories: 50, carbs: 13, fiber: 1.4, vitaminC: 47.8, protein: 0.5, fat: 0.1, sugar: 10, potassium: 109, calcium: 13, iron: 0.29, sodium: 1},
    "pomegranate": {calories: 83, carbs: 19, fiber: 4, vitaminC: 10.2},
    "potato": {calories: 77, carbs: 17, fiber: 2.2, vitaminC: 19.7},
    "raddish": {calories: 16, carbs: 3.4, fiber: 1.6, vitaminC: 14.8},
    "soy beans": {calories: 446, carbs: 30, fiber: 9.3, vitaminC: 6},
    "spinach": {calories: 23, carbs: 3.6, fiber: 2.2, vitaminC: 28.1},
    "sweetcorn": {calories: 86, carbs: 19, fiber: 2.7, vitaminC: 6.8},
    "sweetpotato": {calories: 86, carbs: 20, fiber: 3, vitaminC: 2.4},
    "tomato": {calories: 18, carbs: 3.9, fiber: 1.2, vitaminC: 13.7, protein: 0.9, fat: 0.2, sugar: 2.6, potassium: 237, calcium: 10, iron: 0.27, sodium: 5},
    "turnip": {calories: 28, carbs: 6.4, fiber: 1.8, vitaminC: 21},
    "watermelon": {calories: 30, carbs: 8, fiber: 0.4, vitaminC: 8.1, protein: 0.6, fat: 0.2, sugar: 6.2, potassium: 112, calcium: 7, iron: 0.24, sodium: 1},
    "hami": {calories: 34, carbs: 8.2, fiber: 0.9, vitaminC: 7.9, protein: 0.8, fat: 0.2, sugar: 7.9, potassium: 267, calcium: 9, iron: 0.21, sodium: 15},
    "honeydew": {calories: 36, carbs: 9.1, fiber: 0.8, vitaminC: 18, protein: 0.5, fat: 0.1, sugar: 8.2, potassium: 228, calcium: 6, iron: 0.17, sodium: 18},
    "papaya": {calories: 43, carbs: 11, fiber: 1.7, vitaminC: 60.9, protein: 0.5, fat: 0.3, sugar: 7.8, potassium: 182, calcium: 20, iron: 0.25, sodium: 8},
    "strawberry": {calories: 32, carbs: 7.7, fiber: 2, vitaminC: 58.8, protein: 0.7, fat: 0.3, sugar: 4.9, potassium: 153, calcium: 16, iron: 0.41, sodium: 1},
    "blueberry": {calories: 57, carbs: 14.5, fiber: 2.4, vitaminC: 9.7, protein: 0.7, fat: 0.3, sugar: 10, potassium: 77, calcium: 6, iron: 0.28, sodium: 1},
    "cherry": {calories: 50, carbs: 12, fiber: 1.6, vitaminC: 7, protein: 1, fat: 0.3, sugar: 8, potassium: 173, calcium: 13, iron: 0.36, sodium: 3},
    "peach": {calories: 39, carbs: 9.5, fiber: 1.5, vitaminC: 6.6, protein: 0.9, fat: 0.3, sugar: 8.4, potassium: 190, calcium: 6, iron: 0.25, sodium: 0},
    "plum": {calories: 46, carbs: 11.4, fiber: 1.4, vitaminC: 9.5, protein: 0.7, fat: 0.3, sugar: 9.9, potassium: 157, calcium: 6, iron: 0.17, sodium: 0},
    "apricot": {calories: 48, carbs: 11.1, fiber: 2, vitaminC: 10, protein: 1.4, fat: 0.4, sugar: 9, potassium: 259, calcium: 13, iron: 0.39, sodium: 1},
    "dragonfruit": {calories: 50, carbs: 11, fiber: 3, vitaminC: 20.5, protein: 1.1, fat: 0.1, sugar: 8, potassium: 168, calcium: 18, iron: 0.55, sodium: 1},
    "lychee": {calories: 66, carbs: 16.5, fiber: 1.3, vitaminC: 71.5, protein: 0.8, fat: 0.4, sugar: 15.2, potassium: 171, calcium: 5, iron: 0.31, sodium: 1},
    "longan": {calories: 60, carbs: 15, fiber: 1.1, vitaminC: 84, protein: 1.3, fat: 0.1, sugar: 13, potassium: 266, calcium: 1, iron: 0.13, sodium: 0}
};

function computeGrade(v) {
    let score = 0;
    if (v.fiber >= 2.5) score += 2; else if (v.fiber >= 1.5) score += 1;
    if (v.vitaminC >= 30) score += 2; else if (v.vitaminC >= 15) score += 1;
    if (v.calories <= 60) score += 2; else if (v.calories <= 90) score += 1;
    if (score >= 5) return "A";
    if (score >= 3) return "B";
    return "C";
}
function setGradeBadge(grade) {
    if (!grade || grade === "-") {
        nutritionGrade.textContent = "营养级 -";
        nutritionGrade.className = "grade-badge";
        return;
    }
    nutritionGrade.textContent = "营养级 " + grade;
    nutritionGrade.className = "grade-badge " + (grade === "A" ? "grade-a" : grade === "B" ? "grade-b" : "grade-c");
}
function renderNutrition(en, zh) {
    const v = nutritionData[en] || nutritionData[en + "s"];
    nutritionName.textContent = zh;
    const set = (el, val, unit) => { el.textContent = val != null ? (val + " " + unit) : "—"; };
    set(nutCal, v?.calories, "kcal");
    set(nutCarbs, v?.carbs, " g");
    set(nutFiber, v?.fiber, " g");
    set(nutVitC, v?.vitaminC, " mg");
    set(nutProtein, v?.protein, " g");
    set(nutFat, v?.fat, " g");
    set(nutSugar, v?.sugar, " g");
    set(nutPotassium, v?.potassium, " mg");
    set(nutCalcium, v?.calcium, " mg");
    set(nutIron, v?.iron, " mg");
    set(nutSodium, v?.sodium, " mg");
    const g = v ? computeGrade(v) : "-";
    setGradeBadge(g);
    nutritionCard.classList.remove("hidden");
}
function resolveLabel(raw) {
    if (!raw) return null;
    let en = String(raw).toLowerCase().trim();
    if (aliasMap[en]) en = aliasMap[en];
    const tries = [en];
    if (en.endsWith("ies")) tries.push(en.slice(0, -3) + "y");
    if (en.endsWith("es")) tries.push(en.slice(0, -2));
    if (en.endsWith("s")) tries.push(en.slice(0, -1));
    for (let k of tries) {
        if (aliasMap[k]) k = aliasMap[k];
        if (en2zh[k]) return {en: k, zh: en2zh[k]};
    }
    return {en, zh: en};
}
function setResultTags(zh, en) {
    resultTags.innerHTML = "";
    const same = zh === en;
    const zhBadge = document.createElement("span");
    zhBadge.className = "badge badge-zh";
    zhBadge.textContent = zh;
    resultTags.appendChild(zhBadge);
    if (!same) {
        const enBadge = document.createElement("span");
        enBadge.className = "badge badge-en";
        enBadge.textContent = en;
        resultTags.appendChild(enBadge);
    }
}

async function detectApiBase() {
    for (const h of HOSTS) {
        for (const p of PORTS) {
            try {
                const r = await fetch(`http://${h}:${p}/labels`);
                if (r.ok) {
                    const data = await r.json();
                    const names = data.names || [];
                    idx2name = {};
                    names.forEach((n, i) => {
                        idx2name[i] = n;
                    });
                    API_BASE = `http://${h}:${p}`;
                    return true;
                }
            } catch (e) {
            }
        }
    }
    return false;
}

(async () => {
    const ok = await detectApiBase();
    predictBtn.disabled = !ok;
})();

fileInput.addEventListener("change", () => {
    const file = fileInput.files[0];
    if (!file) return;
    selectionToken++;
    const reader = new FileReader();
reader.onload = e => {
    previewImg.src = e.target.result;
    resultText.textContent = "";
    resultTags.innerHTML = "";
    nutritionCard.classList.add("hidden");
    previewImg.onload = () => {};
};
    reader.readAsDataURL(file);
});

predictBtn.addEventListener("click", async () => {
    const file = fileInput.files[0];
    if (!file) {
        resultText.textContent = "请先选择图片";
        return;
    }
    const token = selectionToken;
    predictBtn.disabled = true;
    resultText.textContent = "识别中...";
    try {
        const fd = new FormData();
        fd.append("file", file);
        if (!API_BASE) {
            throw new Error("后端未连接");
        }
        const resp = await fetch(`${API_BASE}/predict`, {method: "POST", body: fd});
        if (!resp.ok) throw new Error(`接口错误 ${resp.status}`);
        const data = await resp.json();
        let en = data.label_name || data.class_name;
        if (!en) {
            const idx = data.label_index ?? data.class_index ?? data.label ?? data.class;
            if (idx != null && idx2name[idx] != null) en = idx2name[idx];
        }
    if (en) {
        const r = resolveLabel(en);
        const zh = r.zh;
        const enNorm = r.en;
        if (token !== selectionToken) return;
        resultText.textContent = "识别结果：";
        setResultTags(zh, enNorm);
        renderNutrition(enNorm, zh);
    } else {
        resultText.textContent = "无法解析接口返回";
    }
    } catch (e) {
        resultText.textContent = `错误：${e.message}`;
        nutritionCard.classList.add("hidden");
    } finally {
        predictBtn.disabled = false;
    }
});
function rgbToHsl(r,g,b){ r/=255; g/=255; b/=255; const max=Math.max(r,g,b), min=Math.min(r,g,b); let h,s,l=(max+min)/2; if(max===min){ h=s=0; } else { const d=max-min; s=l>0.5? d/(2-max-min) : d/(max+min); switch(max){ case r: h=(g-b)/d+(g<b?6:0); break; case g: h=(b-r)/d+2; break; case b: h=(r-g)/d+4; break; } h/=6; } return [h,s,l]; }
function hslToHex(h,s,l){ h*=360; const a=s*Math.min(l,1-l); const f=n=>{ const k=(n+h/30)%12; const c=l-a*Math.max(Math.min(k-3,9-k,1),-1); return Math.round(255*c); }; const r=f(0), g=f(8), b=f(4); return "#"+[r,g,b].map(x=>x.toString(16).padStart(2,"0")).join(""); }
function shadeHex(hex,dl,ds=0){ const m=hex.match(/^#?([\da-f]{2})([\da-f]{2})([\da-f]{2})$/i); if(!m) return hex; const r=parseInt(m[1],16), g=parseInt(m[2],16), b=parseInt(m[3],16); let [h,s,l]=rgbToHsl(r,g,b); l=Math.min(1,Math.max(0,l+dl)); s=Math.min(1,Math.max(0,s+ds)); return hslToHex(h,s,l); }
function extractDominantColor(img){ const w=64,h=64; const c=document.createElement("canvas"); c.width=w; c.height=h; const ctx=c.getContext("2d"); ctx.drawImage(img,0,0,w,h); const data=ctx.getImageData(0,0,w,h).data; const step=16; const map=new Map(); for(let i=0;i<data.length;i+=4){ const r=data[i], g=data[i+1], b=data[i+2], a=data[i+3]; if(a<128) continue; const k=((r/step)|0)+"_"+((g/step)|0)+"_"+((b/step)|0); const v=map.get(k)||{count:0, r:0, g:0, b:0}; v.count++; v.r+=r; v.g+=g; v.b+=b; map.set(k,v); } let best=null; for(const [k,v] of map){ if(!best || v.count>best.count) best=v; } if(!best) return null; const r=(best.r/best.count)|0, g=(best.g/best.count)|0, b=(best.b/best.count)|0; return "#"+[r,g,b].map(x=>x.toString(16).padStart(2,"0")).join(""); }
function applyTheme(hex){ const root=document.documentElement.style; const strong=shadeHex(hex,-0.12); const weak=shadeHex(hex,0.70,-0.35); const bg1=shadeHex(hex,0.85,-0.45); const bg2=shadeHex(hex,0.92,-0.50); const border=shadeHex(hex,0.60,-0.45); root.setProperty("--accent",hex); root.setProperty("--accent-strong",strong); root.setProperty("--accent-weak",weak); root.setProperty("--bg-start",bg1); root.setProperty("--bg-end",bg2); root.setProperty("--border",border); }
function applyThemeFromImage(img){ }
function applyAccentByLabel(en){ }
