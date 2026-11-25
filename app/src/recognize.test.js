const test = require('node:test');
const assert = require('node:assert');
const { resolveInferenceUrl, parseResponse, parseDetectResponse, recognize, recognizeMulti } = require('./recognize');

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

const detectBody = {
  success: true,
  data: {
    imageWidth: 1024,
    imageHeight: 768,
    items: [
      { itemNo: 1, box: { x0: 0, y0: 0, x1: 0.5, y1: 1 }, detConfidence: 0.9, status: 'SUCCESS',
        candidates: [{ rankNo: 1, foodId: 202, confidence: 0.8 }, { rankNo: 2, foodId: 9999, confidence: 0.1 }] },
      { itemNo: 2, box: { x0: 0.5, y0: 0, x1: 1, y1: 1 }, detConfidence: 0.7, status: 'FAILED',
        candidates: [{ rankNo: 1, foodId: 256, confidence: 0.2 }] },
    ],
  },
};

test('/detect 응답: 음식마다 위치와 후보를 묶고, 모르는 foodId 는 뺀다', () => {
  const r = parseDetectResponse(200, detectBody, foodsById);
  assert.equal(r.imageWidth, 1024);
  assert.deepEqual(r.items.map((i) => [i.itemNo, i.status, i.box.x1]), [[1, 'SUCCESS', 0.5], [2, 'FAILED', 1]]);
  assert.deepEqual(r.items[0].candidates.map((c) => c.food.nameKo), ['불고기']);
});

test('recognizeMulti: /detect 로 보내고, 음식을 못 찾으면 빈 목록', async () => {
  let calledUrl;
  const fetchImpl = async (url) => {
    calledUrl = url;
    return { status: 200, json: async () => ({ success: true, data: { imageWidth: 10, imageHeight: 10, items: [] } }) };
  };
  const r = await recognizeMulti({ uri: 'file:///a.jpg', url: 'http://pc:8100', platform: 'ios', foodsById, fetchImpl });
  assert.equal(calledUrl, 'http://pc:8100/detect');
  assert.deepEqual(r.items, []);
});

test('recognizeMulti: 서버에 검출 모델이 없으면 /predict 로 한 음식만 인식한다', async () => {
  const urls = [];
  const fetchImpl = async (url) => {
    urls.push(url);
    if (url.endsWith('/detect')) {
      return { status: 503, json: async () => ({ success: false, error: { code: 'DETECTOR_NOT_AVAILABLE', message: 'x' } }) };
    }
    return { status: 200, json: async () => ({ success: true, data: { status: 'SUCCESS', candidates: [{ rankNo: 1, foodId: 256, confidence: 0.9 }] } }) };
  };
  const r = await recognizeMulti({ uri: 'file:///a.jpg', url: 'http://pc:8100', platform: 'ios', foodsById, fetchImpl });
  assert.deepEqual(urls, ['http://pc:8100/detect', 'http://pc:8100/predict']);
  assert.equal(r.items.length, 1);
  assert.equal(r.items[0].box, null);
  assert.equal(r.items[0].candidates[0].food.nameKo, '돼지고기볶음');
});
