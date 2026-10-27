const test = require('node:test');
const assert = require('node:assert');
const { resolveInferenceUrl, parseResponse, recognize } = require('./recognize');

const foodsById = new Map([
  [202, { foodId: 202, nameKo: '불고기' }],
  [256, { foodId: 256, nameKo: '돼지고기볶음' }],
]);

test('추론 서버 주소: 환경 변수가 가장 우선', () => {
  assert.equal(
    resolveInferenceUrl({ envUrl: 'http://10.0.0.2:9000/', hostUri: '192.168.0.5:8081', platform: 'ios' }),
    'http://10.0.0.2:9000'
  );
});

test('추론 서버 주소: 휴대폰은 Expo 개발 서버 PC 의 IP, 웹은 localhost', () => {
  assert.equal(resolveInferenceUrl({ hostUri: '192.168.0.5:8081', platform: 'android' }), 'http://192.168.0.5:8100');
  assert.equal(resolveInferenceUrl({ hostUri: '192.168.0.5:8081', platform: 'web' }), 'http://localhost:8100');
  assert.equal(resolveInferenceUrl({ platform: 'ios' }), 'http://localhost:8100');
});

test('성공 응답: 후보를 음식 데이터와 묶고, 모르는 foodId 는 뺀다', () => {
  const r = parseResponse(
    200,
    {
      success: true,
      data: {
        status: 'SUCCESS',
        candidates: [
          { rankNo: 1, foodId: 256, confidence: 0.42 },
          { rankNo: 2, foodId: 202, confidence: 0.4 },
          { rankNo: 3, foodId: 9999, confidence: 0.01 },
        ],
      },
    },
    foodsById
  );
  assert.equal(r.status, 'SUCCESS');
  assert.deepEqual(r.candidates.map((c) => c.food.nameKo), ['돼지고기볶음', '불고기']);
});

test('FAILED 응답은 status 로 알린다', () => {
  const r = parseResponse(200, { success: true, data: { status: 'FAILED', candidates: [] } }, foodsById);
  assert.equal(r.status, 'FAILED');
});

test('오류 응답은 한국어 안내로 바꾼다', () => {
  const r = parseResponse(415, { success: false, error: { code: 'UNSUPPORTED_IMAGE', message: 'x' } }, foodsById);
  assert.match(r.error, /JPEG 또는 PNG/);
  assert.match(parseResponse(502, null, foodsById).error, /502/);
});

test('recognize: 휴대폰은 uri 로 파일을 붙여 /predict 로 보낸다', async () => {
  let called;
  const fetchImpl = async (url, opts) => {
    called = { url, opts };
    return { status: 200, json: async () => ({ success: true, data: { status: 'SUCCESS', candidates: [{ rankNo: 1, foodId: 202, confidence: 0.9 }] } }) };
  };
  const r = await recognize({ uri: 'file:///a.jpg', url: 'http://pc:8100', platform: 'android', foodsById, fetchImpl });
  assert.equal(called.url, 'http://pc:8100/predict');
  assert.equal(called.opts.method, 'POST');
  assert.equal(r.candidates[0].food.nameKo, '불고기');
});

test('recognize: 연결 실패는 오류 메시지로 돌려준다', async () => {
  const fetchImpl = async () => {
    throw new TypeError('Network request failed');
  };
  const r = await recognize({ uri: 'file:///a.jpg', url: 'http://pc:8100', platform: 'ios', foodsById, fetchImpl });
  assert.match(r.error, /연결하지 못했어요/);
});
