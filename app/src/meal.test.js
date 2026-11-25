const test = require('node:test');
const assert = require('node:assert');
const { entriesFromItems, addManual, chooseFood, setGrams, removeEntry, totals } = require('./meal');

const food = (foodId, nameKo, g, kcal) => ({
  foodId, nameKo, servingSizeG: g,
  nutrition: { basisG: g, kcal, carbG: 10, proteinG: 5, fatG: 2, sodiumMg: 100 },
});
const rice = food(1, '쌀밥', 210, 334.8);
const soup = food(2, '된장찌개', 300, 150);
const kimchi = food(3, '배추김치', 50, 15);

const items = [
  { itemNo: 1, box: { x0: 0, y0: 0, x1: 0.5, y1: 1 }, small: true, status: 'SUCCESS',
    candidates: [{ foodId: 1, confidence: 0.9, food: rice }, { foodId: 2, confidence: 0.05, food: soup }] },
  { itemNo: 2, box: { x0: 0.5, y0: 0, x1: 1, y1: 1 }, status: 'FAILED',
    candidates: [{ foodId: 2, confidence: 0.2, food: soup }] },
];

test('인식 성공한 음식은 1순위를 1인분으로 미리 고르고, 실패한 음식은 비워 둔다', () => {
  const es = entriesFromItems(items);
  assert.equal(es[0].food.nameKo, '쌀밥');
  assert.equal(es[0].gramText, '210');
  assert.equal(es[1].food, null);
  assert.equal(es[1].gramText, '');
  assert.notEqual(es[0].key, es[1].key);
  assert.deepEqual(es.map((e) => e.small), [true, false]);
});

test('합계는 계산할 수 있는 음식만 더한다', () => {
  const es = entriesFromItems(items);
  const t = totals(es);
  assert.equal(t.count, 1);
  assert.equal(Math.round(t.kcal), 335);
});

test('후보를 바꾸면 양도 새 음식의 1인분으로 맞춘다', () => {
  let es = entriesFromItems(items);
  es = chooseFood(es, es[1].key, soup);
  assert.equal(es[1].food.nameKo, '된장찌개');
  assert.equal(es[1].gramText, '300');
  assert.equal(Math.round(totals(es).kcal), 335 + 150);
});

test('양 바꾸기, 삭제, 검색으로 추가', () => {
  let es = entriesFromItems(items);
  es = setGrams(es, es[0].key, '105');                   // 반 공기
  assert.equal(Math.round(totals(es).kcal), 167);
  es = addManual(es, kimchi);
  assert.deepEqual(es.map((e) => e.itemNo), [1, 2, 3]);
  assert.equal(es[2].box, null);
  assert.equal(es[2].status, 'MANUAL');
  es = removeEntry(es, es[1].key);
  assert.deepEqual(es.map((e) => e.food.nameKo), ['쌀밥', '배추김치']);
  assert.equal(Math.round(totals(es).kcal), 167 + 15);
});

test('양이 비었거나 0 이면 그 음식은 합계에서 뺀다', () => {
  let es = entriesFromItems(items);
  es = setGrams(es, es[0].key, '');
  assert.equal(totals(es).count, 0);
  assert.equal(totals(es).kcal, 0);
});
