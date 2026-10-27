// 사진 인식. 과제 시연용으로 앱이 추론 서버(server/inference)를 직접 부른다.
// shortcut: 위키 설계는 앱 → API 서버 → 추론 서버다. API 서버가 생기면 POST /images, /recognitions 로 바꾼다.

const DEFAULT_PORT = 8100;
const TIMEOUT_MS = 20000;

// 추론 서버 주소.
// 1. app/.env 의 EXPO_PUBLIC_INFERENCE_URL 이 있으면 그 값
// 2. 휴대폰(Expo Go)이면 Expo 개발 서버가 도는 PC 의 IP (hostUri "192.168.0.5:8081" → http://192.168.0.5:8100)
// 3. 그 밖(웹)은 http://localhost:8100
function resolveInferenceUrl({ envUrl, hostUri, platform }) {
  if (envUrl) return envUrl.replace(/\/+$/, '');
  if (platform !== 'web' && hostUri) return `http://${hostUri.split(':')[0]}:${DEFAULT_PORT}`;
  return `http://localhost:${DEFAULT_PORT}`;
}

// 서버 응답의 후보를 앱 음식 데이터와 묶는다. 앱 데이터에 없는 foodId 는 뺀다.
function toCandidates(data, foodsById) {
  return (data.candidates || [])
    .map((c) => ({ ...c, food: foodsById.get(c.foodId) }))
    .filter((c) => c.food);
}

const ERROR_MESSAGES = {
  IMAGE_TOO_LARGE: '사진 용량이 너무 커요. 다른 사진을 골라 주세요.',
  UNSUPPORTED_IMAGE: 'JPEG 또는 PNG 사진만 인식할 수 있어요.',
  INVALID_REQUEST: '사진을 보내지 못했어요. 다시 시도해 주세요.',
};

// 서버 응답(JSON) → { status, candidates } 또는 { error }
function parseResponse(httpStatus, body, foodsById) {
  if (body && body.success) {
    return { status: body.data.status, candidates: toCandidates(body.data, foodsById) };
  }
  const code = body?.error?.code;
  return { error: ERROR_MESSAGES[code] || `인식 서버 오류가 났어요 (${httpStatus}).` };
}

// 사진 1장을 보내고 결과를 받는다. 네트워크 오류도 { error } 로 돌려준다.
async function recognize({ uri, mimeType, url, platform, foodsById, fetchImpl = fetch }) {
  const form = new FormData();
  const type = mimeType || 'image/jpeg';
  const name = type === 'image/png' ? 'photo.png' : 'photo.jpg';
  if (platform === 'web') {
    // 웹은 uri 가 blob:/data: 라서 파일 내용을 읽어 붙인다
    form.append('file', await (await fetchImpl(uri)).blob(), name);
  } else {
    form.append('file', { uri, name, type });
  }

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), TIMEOUT_MS);
  try {
    const res = await fetchImpl(`${url}/predict`, { method: 'POST', body: form, signal: controller.signal });
    const body = await res.json().catch(() => null);
    return parseResponse(res.status, body, foodsById);
  } catch (e) {
    return {
      error:
        e.name === 'AbortError'
          ? '인식 서버가 응답하지 않아요. 잠시 후 다시 시도해 주세요.'
          : `인식 서버에 연결하지 못했어요 (${url}). 서버가 켜져 있는지, 같은 Wi-Fi 인지 확인해 주세요.`,
    };
  } finally {
    clearTimeout(timer);
  }
}

module.exports = { resolveInferenceUrl, toCandidates, parseResponse, recognize };
