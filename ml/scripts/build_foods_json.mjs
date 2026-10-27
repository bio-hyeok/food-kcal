// 영양DB(xlsx) → 앱용 foods.json + ml/label_map.csv 변환.
//
// 사용: node scripts/build_foods_json.mjs "<영양DB.xlsx 경로>"
// 출처: AI-Hub 74번 「음식 이미지 및 영양정보 텍스트」의 `음식분류 AI 데이터 영양DB.xlsx` (400종).
//
// 규칙
// - 열: 음식명, 중량(g), 에너지(kcal), 탄수화물(g), 당류(g), 지방(g), 단백질(g), ... 나트륨(mg) ...
// - 영양값은 "중량(g) 한 접시(양 Q4=100%)" 기준이다 → basisG = servingSizeG = 중량.
// - foodId는 엑셀 행 순서(1부터)다. 모델 클래스 번호와 이어지므로 한 번 정하면 바꾸지 않는다.
//   엑셀이 갱신되면 기존 행의 순서를 유지하고 뒤에 추가해야 한다.
// - 값이 '-'이거나 비어 있으면 null. 앱이 쓰는 5개 값(kcal, 탄수화물, 단백질, 지방, 나트륨)이
//   하나라도 비면 변환을 멈춘다 (앱 계산이 깨지므로).
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import readXlsxFile from 'read-excel-file/node';

const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(here, '..', '..');
const xlsxPath = process.argv[2];
if (!xlsxPath) {
  console.error('사용법: node scripts/build_foods_json.mjs "<영양DB.xlsx 경로>"');
  process.exit(1);
}

const num = (v) => {
  if (v === null || v === undefined) return null;
  if (typeof v === 'number') return v;
  const s = String(v).trim().replace(/,/g, '');
  if (s === '' || s === '-') return null;
  const n = Number(s);
  return Number.isFinite(n) ? n : null;
};

const rows = await readXlsxFile(xlsxPath);
const headerIdx = rows.findIndex((r) => r.some((c) => String(c ?? '').includes('중량(g)')));
if (headerIdx < 0) throw new Error("헤더 행('중량(g)')을 찾지 못했습니다.");
const header = rows[headerIdx].map((c) => String(c ?? '').replace(/\s+/g, ''));
const col = (name) => {
  const i = header.findIndex((h) => h.startsWith(name));
  if (i < 0) throw new Error(`열을 찾지 못했습니다: ${name}`);
  return i;
};
const C = {
  name: 0,
  weight: col('중량'),
  kcal: col('에너지'),
  carb: col('탄수화물'),
  protein: col('단백질'),
  fat: col('지방'),
  sodium: col('나트륨'),
};

const foods = [];
const problems = [];
const seen = new Map();
for (const r of rows.slice(headerIdx + 1)) {
  const nameKo = String(r[C.name] ?? '').trim();
  if (!nameKo) continue;
  const weight = num(r[C.weight]);
  const n = {
    kcal: num(r[C.kcal]),
    carbG: num(r[C.carb]),
    proteinG: num(r[C.protein]),
    fatG: num(r[C.fat]),
    sodiumMg: num(r[C.sodium]),
  };
  const missing = [['중량', weight], ...Object.entries(n)].filter(([, v]) => v === null).map(([k]) => k);
  if (missing.length || !(weight > 0) || !(n.kcal > 0)) {
    problems.push(`${nameKo}: 값 없음/이상 (${missing.join(', ') || '0 이하'})`);
    continue;
  }
  if (seen.has(nameKo)) problems.push(`${nameKo}: 이름 중복 (행 ${seen.get(nameKo)})`);
  seen.set(nameKo, foods.length + 1);
  foods.push({
    foodId: foods.length + 1,
    nameKo,
    servingSizeG: weight,
    nutrition: { basisG: weight, ...n },
  });
}

if (problems.length) {
  console.error('변환 중단. 아래 항목을 확인하세요:\n - ' + problems.join('\n - '));
  process.exit(1);
}

const round = (v) => Math.round(v * 100) / 100;
for (const f of foods) for (const k of Object.keys(f.nutrition)) f.nutrition[k] = round(f.nutrition[k]);
const jsonPath = path.join(root, 'app', 'src', 'foods.json');
fs.writeFileSync(jsonPath, JSON.stringify(foods, null, 2) + '\n');

// label_map.csv: dataset_label은 학습 데이터 쪽 이름(폴더명/코드)과 맞출 때 채운다. 지금은 비워 둔다.
const csv = ['dataset_label,food_id,name_ko', ...foods.map((f) => `,${f.foodId},${f.nameKo}`)].join('\n') + '\n';
fs.writeFileSync(path.join(root, 'ml', 'label_map.csv'), csv);

console.log(`${foods.length}종 변환 완료 → ${path.relative(root, jsonPath)}, ml/label_map.csv`);
