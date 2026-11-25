// 한 사진의 음식 목록(식사). 화면과 분리해 시험할 수 있게 둔다.
// 항목: { key, itemNo, box, status, candidates, food, gramText }
//   box       사진 대비 0~1 비율 (검색으로 직접 추가한 음식은 null)
//   food      고른 음식 (아직 안 골랐으면 null)
//   gramText  먹은 양(g) 입력값. 문자열로 두어 입력 중인 값도 그대로 보여준다
const { calculate } = require('./calc');

let seq = 0;
const nextKey = () => `m${++seq}`;

// 인식 결과 → 항목들. 인식에 성공한(SUCCESS) 음식은 1순위를 미리 골라 둔다.
function entriesFromItems(items) {
  return items.map((it) => {
    const food = it.status === 'SUCCESS' && it.candidates.length ? it.candidates[0].food : null;
    return {
      key: nextKey(),
      itemNo: it.itemNo,
      box: it.box || null,
      status: it.status,
      candidates: it.candidates,
      food,
      gramText: food ? String(food.servingSizeG) : '',
    };
  });
}

// 검색에서 고른 음식을 목록 끝에 더한다 (위치 없음).
function addManual(entries, food) {
  const itemNo = entries.reduce((m, e) => Math.max(m, e.itemNo), 0) + 1;
  return [
    ...entries,
    { key: nextKey(), itemNo, box: null, status: 'MANUAL', candidates: [], food, gramText: String(food.servingSizeG) },
  ];
}

// 항목의 음식을 바꾼다. 양은 새 음식의 1인분으로 다시 맞춘다.
function chooseFood(entries, key, food) {
  return entries.map((e) => (e.key === key ? { ...e, food, gramText: String(food.servingSizeG) } : e));
}

function setGrams(entries, key, gramText) {
  return entries.map((e) => (e.key === key ? { ...e, gramText } : e));
}

function removeEntry(entries, key) {
  return entries.filter((e) => e.key !== key);
}

// 항목 하나의 영양값 (음식을 안 골랐거나 양이 없으면 null)
function entryNutrition(e) {
  return e.food ? calculate(e.food, Number(e.gramText)) : null;
}

// 합계. 계산할 수 있는 항목만 더하고, 몇 개를 더했는지 함께 준다.
function totals(entries) {
  const sum = { kcal: 0, carbG: 0, proteinG: 0, fatG: 0, sodiumMg: 0, count: 0 };
  for (const e of entries) {
    const n = entryNutrition(e);
    if (!n) continue;
    for (const k of ['kcal', 'carbG', 'proteinG', 'fatG', 'sodiumMg']) sum[k] += n[k];
    sum.count += 1;
  }
  return sum;
}

module.exports = { entriesFromItems, addManual, chooseFood, setGrams, removeEntry, entryNutrition, totals };
