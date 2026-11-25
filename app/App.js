import { useMemo, useState } from 'react';
import {
  ActivityIndicator,
  Alert,
  Image,
  Platform,
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  View,
} from 'react-native';
import { SafeAreaProvider, SafeAreaView } from 'react-native-safe-area-context';
import { StatusBar } from 'expo-status-bar';
import * as ImagePicker from 'expo-image-picker';
import Constants from 'expo-constants';
import foods from './src/foods.json';
import { gramsForServings } from './src/calc';
import { recognizeMulti, resolveInferenceUrl } from './src/recognize';
import {
  addManual,
  chooseFood,
  entriesFromItems,
  entryNutrition,
  removeEntry,
  setGrams,
  totals,
} from './src/meal';

const SERVINGS = [0.5, 1, 1.5, 2];
const MAX_LIST = 30;
const round1 = (n) => Math.round(n * 10) / 10;
const foodsById = new Map(foods.map((f) => [f.foodId, f]));
const INFERENCE_URL = resolveInferenceUrl({
  envUrl: process.env.EXPO_PUBLIC_INFERENCE_URL,
  hostUri: Constants.expoConfig?.hostUri,
  platform: Platform.OS,
});

export default function App() {
  const [photo, setPhoto] = useState(null); // { uri, width, height }
  const [query, setQuery] = useState('');
  // 인식 상태: null(사진 없음) · { loading } · { done } · { error }
  const [recog, setRecog] = useState(null);
  const [entries, setEntries] = useState([]);

  const allMatches = useMemo(() => foods.filter((f) => f.nameKo.includes(query.trim())), [query]);
  const matches = allMatches.slice(0, MAX_LIST); // 400종을 한 번에 그리지 않는다
  const sum = totals(entries);

  const runRecognition = async (asset) => {
    setRecog({ loading: true });
    const r = await recognizeMulti({
      uri: asset.uri,
      mimeType: asset.mimeType,
      url: INFERENCE_URL,
      platform: Platform.OS,
      foodsById,
    });
    if (r.error) {
      setRecog({ error: r.error });
      return;
    }
    // 서버가 알려 준 크기는 EXIF 회전을 반영한 값이라 상자 위치와 맞는다
    if (r.imageWidth && r.imageHeight) {
      setPhoto((p) => (p && p.uri === asset.uri ? { ...p, width: r.imageWidth, height: r.imageHeight } : p));
    }
    setEntries(entriesFromItems(r.items));
    setRecog({ done: true, found: r.items.length });
  };
  const pick = async (launch) => {
    const r = await launch({ mediaTypes: ['images'], quality: 0.7 });
    if (r.canceled) return;
    const a = r.assets[0];
    setPhoto({ uri: a.uri, width: a.width, height: a.height });
    setEntries([]);
    await runRecognition(a);
  };
  const takePhoto = async () => {
    const p = await ImagePicker.requestCameraPermissionsAsync();
    if (!p.granted) {
      Alert.alert('카메라 권한이 필요해요', '설정에서 카메라 권한을 허용해 주세요.');
      return;
    }
    await pick(ImagePicker.launchCameraAsync);
  };

  return (
    <SafeAreaProvider>
      <SafeAreaView style={styles.safe}>
        <StatusBar style="dark" />
        <ScrollView contentContainerStyle={styles.body} keyboardShouldPersistTaps="handled">
          <Text style={styles.title}>음식 칼로리</Text>
          <Text style={styles.notice}>※ 한 접시 기준 영양 값이며, 실제 음식과 다를 수 있어요.</Text>

          <PhotoWithBoxes photo={photo} entries={entries} />
          <View style={styles.row}>
            <Button label="갤러리" onPress={() => pick(ImagePicker.launchImageLibraryAsync)} />
            <Button label="카메라" onPress={takePhoto} />
          </View>

          {!recog && <Text style={styles.hint}>사진을 고르면 음식을 찾아 하나씩 칼로리를 알려줘요. 여러 음식도 돼요.</Text>}
          {recog?.loading && (
            <View style={styles.gramRow}>
              <ActivityIndicator color={GREEN} />
              <Text style={styles.muted}>음식을 찾는 중이에요…</Text>
            </View>
          )}
          {recog?.error && <Text style={styles.warn}>{recog.error} 아래에서 직접 검색해 추가할 수 있어요.</Text>}
          {recog?.done && recog.found === 0 && (
            <Text style={styles.warn}>음식을 찾지 못했어요. 아래에서 직접 검색해 추가해 주세요.</Text>
          )}
          {recog?.done && recog.found > 0 && (
            <Text style={styles.hint}>음식 {recog.found}개를 찾았어요. 틀린 음식은 후보를 눌러 바꾸거나 삭제해 주세요.</Text>
          )}

          {entries.length > 0 && (
            <View style={styles.totalBox}>
              <Text style={styles.totalLabel}>
                합계 · 음식 {entries.length}개{sum.count < entries.length ? ` 중 ${sum.count}개 계산` : ''}
              </Text>
              <Text style={styles.kcal}>약 {Math.round(sum.kcal)} kcal</Text>
              <Text style={styles.muted}>
                탄수화물 {round1(sum.carbG)}g · 단백질 {round1(sum.proteinG)}g · 지방 {round1(sum.fatG)}g · 나트륨{' '}
                {Math.round(sum.sodiumMg)}mg
              </Text>
            </View>
          )}

          {entries.map((e) => (
            <EntryCard
              key={e.key}
              entry={e}
              onChoose={(f) => setEntries((xs) => chooseFood(xs, e.key, f))}
              onGrams={(t) => setEntries((xs) => setGrams(xs, e.key, t))}
              onRemove={() => setEntries((xs) => removeEntry(xs, e.key))}
            />
          ))}

          <Text style={styles.section}>음식 검색</Text>
          <Text style={styles.hint}>찾지 못한 음식이 있으면 검색해서 눌러 주세요. 목록에 추가돼요.</Text>
          <TextInput style={styles.input} placeholder="음식 이름 검색" value={query} onChangeText={setQuery} />
          {matches.length === 0 && <Text style={styles.muted}>검색 결과가 없어요</Text>}
          {allMatches.length > MAX_LIST && (
            <Text style={styles.muted}>
              {allMatches.length}개 중 {MAX_LIST}개만 보여요. 이름을 더 입력해 보세요.
            </Text>
          )}
          {matches.map((f) => (
            <Pressable key={f.foodId} onPress={() => setEntries((xs) => addManual(xs, f))} style={styles.item}>
              <Text style={styles.itemName}>{f.nameKo}</Text>
              <Text style={styles.muted}>
                {f.servingSizeG}g · {f.nutrition.kcal}kcal · 추가
              </Text>
            </Pressable>
          ))}
        </ScrollView>
      </SafeAreaView>
    </SafeAreaProvider>
  );
}

// 사진과 찾은 음식 상자. 상자 위치는 사진 대비 비율이라, 사진을 자르지 않고 원래 비율 그대로 보여준다.
function PhotoWithBoxes({ photo, entries }) {
  if (!photo) {
    return (
      <View style={[styles.photoBox, styles.photoEmpty]}>
        <Text style={styles.muted}>사진을 추가해 주세요</Text>
      </View>
    );
  }
  const ratio = photo.width && photo.height ? photo.width / photo.height : 4 / 3;
  return (
    <View style={[styles.photoBox, { aspectRatio: ratio }]}>
      <Image source={{ uri: photo.uri }} style={styles.photo} resizeMode="stretch" />
      {entries
        .filter((e) => e.box)
        .map((e) => (
          <View
            key={e.key}
            style={[
              styles.box,
              {
                left: `${e.box.x0 * 100}%`,
                top: `${e.box.y0 * 100}%`,
                width: `${(e.box.x1 - e.box.x0) * 100}%`,
                height: `${(e.box.y1 - e.box.y0) * 100}%`,
              },
            ]}
          >
            <Text style={styles.boxLabel}>{e.itemNo}</Text>
          </View>
        ))}
    </View>
  );
}

// 음식 하나: 후보 고르기, 양 입력, 칼로리
function EntryCard({ entry: e, onChoose, onGrams, onRemove }) {
  const n = entryNutrition(e);
  const amountG = Number(e.gramText);
  return (
    <View style={styles.card}>
      <View style={styles.cardHead}>
        <View style={styles.numBadge}>
          <Text style={styles.numText}>{e.itemNo}</Text>
        </View>
        <Text style={[styles.section, styles.cardTitle]}>{e.food ? e.food.nameKo : '어떤 음식인가요?'}</Text>
        <Pressable onPress={onRemove} hitSlop={8}>
          <Text style={styles.remove}>삭제</Text>
        </Pressable>
      </View>

      {e.status === 'MANUAL' && <Text style={styles.hint}>검색으로 추가한 음식</Text>}
      {e.status === 'FAILED' && !e.food && (
        <Text style={styles.warn}>확실하지 않아요. 아래 후보에 있으면 골라 주세요. 없으면 삭제하고 검색해 주세요.</Text>
      )}
      {e.candidates.length > 0 && (
        <View style={styles.row}>
          {e.candidates.map((c) => (
            <Pressable
              key={c.foodId}
              onPress={() => onChoose(c.food)}
              style={[styles.chip, e.food?.foodId === c.foodId && styles.chipOn]}
            >
              <Text style={[styles.chipText, e.food?.foodId === c.foodId && styles.chipTextOn]}>
                {c.food.nameKo} {Math.round(c.confidence * 100)}%
              </Text>
            </Pressable>
          ))}
        </View>
      )}

      {e.food && (
        <>
          <View style={styles.row}>
            {SERVINGS.map((r) => (
              <Button
                key={r}
                label={`${r}인분`}
                on={amountG === gramsForServings(e.food, r)}
                onPress={() => onGrams(String(gramsForServings(e.food, r)))}
              />
            ))}
          </View>
          <View style={styles.gramRow}>
            <TextInput
              style={[styles.input, styles.gramInput]}
              keyboardType="numeric"
              value={e.gramText}
              onChangeText={onGrams}
            />
            <Text style={styles.itemName}>g</Text>
            {n ? (
              <Text style={styles.itemKcal}>약 {Math.round(n.kcal)} kcal</Text>
            ) : (
              <Text style={styles.muted}>먹은 양을 입력해 주세요</Text>
            )}
          </View>
        </>
      )}
    </View>
  );
}

function Button({ label, onPress, on }) {
  return (
    <Pressable onPress={onPress} style={[styles.btn, on && styles.btnOn]}>
      <Text style={[styles.btnText, on && styles.btnTextOn]}>{label}</Text>
    </Pressable>
  );
}

const GREEN = '#2e7d32';
const styles = StyleSheet.create({
  safe: { flex: 1, backgroundColor: '#fff' },
  body: { padding: 16, gap: 12 },
  title: { fontSize: 24, fontWeight: '700' },
  notice: { fontSize: 12, color: '#b26a00' },
  photoBox: { width: '100%', borderRadius: 12, overflow: 'hidden', backgroundColor: '#f0f0f0' },
  photoEmpty: {
    height: 220,
    borderWidth: 1,
    borderStyle: 'dashed',
    borderColor: '#bbb',
    backgroundColor: '#fff',
    alignItems: 'center',
    justifyContent: 'center',
  },
  photo: { width: '100%', height: '100%' },
  // 상자는 터치를 받지 않는다 (아래 스크롤 방해 방지)
  box: { position: 'absolute', borderWidth: 3, borderColor: GREEN, borderRadius: 4, pointerEvents: 'none' },
  boxLabel: {
    position: 'absolute',
    left: -3,
    top: -3,
    backgroundColor: GREEN,
    color: '#fff',
    fontWeight: '700',
    paddingHorizontal: 7,
    paddingVertical: 1,
    borderBottomRightRadius: 6,
  },
  row: { flexDirection: 'row', gap: 8, flexWrap: 'wrap' },
  gramRow: { flexDirection: 'row', alignItems: 'center', gap: 8 },
  gramInput: { width: 100 },
  hint: { fontSize: 12, color: '#777' },
  warn: { fontSize: 14, color: '#b26a00' },
  section: { fontSize: 18, fontWeight: '600', marginTop: 8 },
  input: { borderWidth: 1, borderColor: '#ccc', borderRadius: 8, padding: 10, fontSize: 16 },
  item: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    padding: 12,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: '#e0e0e0',
  },
  itemName: { fontSize: 16 },
  itemKcal: { fontSize: 18, fontWeight: '700', color: GREEN, marginLeft: 'auto' },
  muted: { color: '#777' },
  totalBox: { gap: 4, padding: 14, borderRadius: 12, backgroundColor: '#e8f5e9' },
  totalLabel: { fontSize: 14, fontWeight: '600', color: GREEN },
  kcal: { fontSize: 32, fontWeight: '700', color: GREEN },
  card: { gap: 10, padding: 14, borderRadius: 12, backgroundColor: '#f6f6f6' },
  cardHead: { flexDirection: 'row', alignItems: 'center', gap: 10 },
  cardTitle: { flex: 1, marginTop: 0 },
  numBadge: {
    width: 28,
    height: 28,
    borderRadius: 14,
    backgroundColor: GREEN,
    alignItems: 'center',
    justifyContent: 'center',
  },
  numText: { color: '#fff', fontWeight: '700' },
  remove: { color: '#b00020', fontWeight: '600' },
  chip: {
    paddingVertical: 6,
    paddingHorizontal: 10,
    borderRadius: 16,
    borderWidth: 1,
    borderColor: '#ccc',
    backgroundColor: '#fff',
  },
  chipOn: { borderColor: GREEN, backgroundColor: '#e8f5e9' },
  chipText: { color: '#333' },
  chipTextOn: { color: GREEN, fontWeight: '700' },
  btn: { paddingVertical: 8, paddingHorizontal: 14, borderRadius: 8, borderWidth: 1, borderColor: GREEN },
  btnOn: { backgroundColor: GREEN },
  btnText: { color: GREEN, fontWeight: '600' },
  btnTextOn: { color: '#fff' },
});
