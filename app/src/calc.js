// 칼로리 계산. 규칙은 위키 ERD 문서 4절을 따른다.
// shortcut: basisG와 servingSizeG가 모두 있는 음식만 다룬다. basis_g가 NULL인 조합(ERD 4.1)은 서버 연동 때 추가한다.

// amountG(g)를 먹었을 때의 영양소. 양이 0 이하이거나 숫자가 아니면 null.
function calculate(food, amountG) {
  if (!(amountG > 0)) return null;
  const { basisG, kcal, carbG, proteinG, fatG, sodiumMg } = food.nutrition;
  const k = amountG / basisG;
  return { kcal: kcal * k, carbG: carbG * k, proteinG: proteinG * k, fatG: fatG * k, sodiumMg: sodiumMg * k };
}

// 인분 배율 → g
function gramsForServings(food, ratio) {
  return ratio * food.servingSizeG;
}

module.exports = { calculate, gramsForServings };
