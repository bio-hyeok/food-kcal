const test = require('node:test');
const assert = require('node:assert/strict');
const { calculate, gramsForServings } = require('./calc');
const foods = require('./foods.sample.json');

const byName = (name) => foods.find((f) => f.nameKo === name);

test('1인분 기준 음식은 1인분이면 기준 칼로리 그대로', () => {
  assert.equal(calculate(byName('공깃밥'), 210).kcal, 300);
});

test('양에 비례한다 (0.5인분 = 105g)', () => {
  const rice = byName('공깃밥');
  assert.equal(calculate(rice, gramsForServings(rice, 0.5)).kcal, 150);
});

test('100g 기준 음식은 g에 비례한다', () => {
  assert.equal(calculate(byName('닭가슴살 (삶은 것)'), 250).kcal, 412.5);
});

test('영양소도 같은 비율로 계산한다', () => {
  assert.equal(calculate(byName('삶은 달걀 (1개)'), 100).proteinG, 12.6);
});

test('양이 0 이하이거나 숫자가 아니면 null', () => {
  const rice = byName('공깃밥');
  for (const bad of [0, -5, NaN, undefined]) assert.equal(calculate(rice, bad), null);
});

test('인분 배율을 g으로 바꾼다', () => {
  assert.equal(gramsForServings(byName('공깃밥'), 1.5), 315);
});

test('샘플 데이터: foodId 중복이 없고 계산에 필요한 값이 모두 양수', () => {
  assert.equal(new Set(foods.map((f) => f.foodId)).size, foods.length);
  for (const f of foods) {
    assert.ok(f.servingSizeG > 0 && f.nutrition.basisG > 0 && f.nutrition.kcal > 0, f.nameKo);
  }
});
