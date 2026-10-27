const test = require('node:test');
const assert = require('node:assert/strict');
const { calculate, gramsForServings } = require('./calc');
const foods = require('./foods.json');

const byName = (name) => foods.find((f) => f.nameKo === name);

test('영양DB: 400종, foodId는 1부터 연속이고 이름 중복이 없다', () => {
  assert.equal(foods.length, 400);
  assert.deepEqual(foods.map((f) => f.foodId), foods.map((_, i) => i + 1));
  assert.equal(new Set(foods.map((f) => f.nameKo)).size, foods.length);
});

test('영양DB: 계산에 필요한 값이 모두 있고 basisG = servingSizeG', () => {
  for (const f of foods) {
    const n = f.nutrition;
    assert.ok(f.servingSizeG > 0 && n.basisG === f.servingSizeG, f.nameKo);
    for (const k of ['kcal', 'carbG', 'proteinG', 'fatG', 'sodiumMg']) {
      assert.ok(Number.isFinite(n[k]) && n[k] >= 0, `${f.nameKo}.${k}`);
    }
    assert.ok(n.kcal > 0, f.nameKo);
  }
});

test('엑셀 원본 값과 일치한다 (오므라이스 450g, 684.99kcal)', () => {
  const f = byName('오므라이스');
  assert.equal(f.servingSizeG, 450);
  assert.equal(f.nutrition.kcal, 684.99);
});

test('양 단계 Q3(75%)는 한 접시 칼로리의 0.75배', () => {
  const f = byName('오므라이스');
  const g = gramsForServings(f, 0.75); // 337.5g
  assert.equal(g, 337.5);
  assert.ok(Math.abs(calculate(f, g).kcal - 684.99 * 0.75) < 1e-9);
});
