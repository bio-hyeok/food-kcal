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
import { calculate, gramsForServings } from './src/calc';
import { recognize, resolveInferenceUrl } from './src/recognize';

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
  const [photoUri, setPhotoUri] = useState(null);
  const [query, setQuery] = useState('');
  const [food, setFood] = useState(null);
  const [gramText, setGramText] = useState('');
  // 인식 상태: null(사진 없음) · { loading } · { status, candidates } · { error }
  const [recog, setRecog] = useState(null);

  const allMatches = useMemo(() => foods.filter((f) => f.nameKo.includes(query.trim())), [query]);
  const matches = allMatches.slice(0, MAX_LIST); // 400종을 한 번에 그리지 않는다

  const runRecognition = async (asset) => {
    setRecog({ loading: true });
    setRecog(
      await recognize({ uri: asset.uri, mimeType: asset.mimeType, url: INFERENCE_URL, platform: Platform.OS, foodsById })
    );
  };
  const pick = async (launch) => {
    const r = await launch({ mediaTypes: ['images'], quality: 0.7 });
    if (r.canceled) return;
    setPhotoUri(r.assets[0].uri);
    setFood(null);
    await runRecognition(r.assets[0]);
  };
  const takePhoto = async () => {
    const p = await ImagePicker.requestCameraPermissionsAsync();
    if (!p.granted) {
      Alert.alert('카메라 권한이 필요해요', '설정에서 카메라 권한을 허용해 주세요.');
      return;
    }
    await pick(ImagePicker.launchCameraAsync);
  };
  const choose = (f) => {
    setFood(f);
    setGramText(String(f.servingSizeG));
  };

  return (
    <SafeAreaProvider>
      <SafeAreaView style={styles.safe}>
        <StatusBar style="dark" />
        <ScrollView contentContainerStyle={styles.body} keyboardShouldPersistTaps="handled">
          <Text style={styles.title}>음식 칼로리</Text>
          <Text style={styles.notice}>※ 한 접시 기준 영양 값이며, 실제 음식과 다를 수 있어요.</Text>

          <View style={styles.photoBox}>
            {photoUri ? (
              <Image source={{ uri: photoUri }} style={styles.photo} resizeMode="cover" />
            ) : (
              <Text style={styles.muted}>사진을 추가해 주세요</Text>
            )}
          </View>
          <View style={styles.row}>
            <Button label="갤러리" onPress={() => pick(ImagePicker.launchImageLibraryAsync)} />
            <Button label="카메라" onPress={takePhoto} />
          </View>

          {!recog && <Text style={styles.hint}>사진을 고르면 음식을 자동으로 찾아 후보를 보여줘요.</Text>}
          {recog?.loading && (
            <View style={styles.gramRow}>
              <ActivityIndicator color={GREEN} />
              <Text style={styles.muted}>음식을 찾는 중이에요…</Text>
            </View>
          )}
          {recog?.error && <Text style={styles.warn}>{recog.error} 아래에서 직접 검색할 수 있어요.</Text>}
          {recog?.status === 'FAILED' && (
            <Text style={styles.warn}>음식을 찾지 못했어요. 아래에서 직접 검색해 주세요.</Text>
          )}
          {recog?.status === 'SUCCESS' && (
            <>
              <Text style={styles.section}>이 음식인가요?</Text>
              {recog.candidates.map((c) => (
                <Pressable
                  key={c.foodId}
                  onPress={() => choose(c.food)}
                  style={[styles.item, food?.foodId === c.foodId && styles.itemOn]}
                >
                  <Text style={styles.itemName}>
                    {c.rankNo}. {c.food.nameKo}
                  </Text>
                  <Text style={styles.muted}>
                    {Math.round(c.confidence * 100)}% · {Math.round(c.food.nutrition.kcal)}kcal
                  </Text>
                </Pressable>
              ))}
              <Text style={styles.hint}>목록에 없으면 아래에서 직접 검색해 주세요.</Text>
            </>
          )}

          {food && <ResultCard food={food} gramText={gramText} setGramText={setGramText} />}

          <Text style={styles.section}>음식 검색</Text>
          <TextInput style={styles.input} placeholder="음식 이름 검색" value={query} onChangeText={setQuery} />
          {matches.length === 0 && <Text style={styles.muted}>검색 결과가 없어요</Text>}
          {allMatches.length > MAX_LIST && (
            <Text style={styles.muted}>
              {allMatches.length}개 중 {MAX_LIST}개만 보여요. 이름을 더 입력해 보세요.
            </Text>
          )}
          {matches.map((f) => (
            <Pressable
              key={f.foodId}
              onPress={() => choose(f)}
              style={[styles.item, food?.foodId === f.foodId && styles.itemOn]}
            >
              <Text style={styles.itemName}>{f.nameKo}</Text>
              <Text style={styles.muted}>
                {f.servingSizeG}g · {f.nutrition.kcal}kcal
              </Text>
            </Pressable>
          ))}
        </ScrollView>
      </SafeAreaView>
    </SafeAreaProvider>
  );
}

// 고른 음식의 양 입력과 칼로리. 후보 목록 바로 아래, 음식 검색 위에 둔다.
function ResultCard({ food, gramText, setGramText }) {
  const amountG = Number(gramText);
  const result = calculate(food, amountG);
  return (
    <View style={styles.card}>
      <Text style={styles.section}>{food.nameKo}</Text>
      <View style={styles.row}>
        {SERVINGS.map((r) => (
          <Button
            key={r}
            label={`${r}인분`}
            on={amountG === gramsForServings(food, r)}
            onPress={() => setGramText(String(gramsForServings(food, r)))}
          />
        ))}
      </View>
      <View style={styles.gramRow}>
        <TextInput
          style={[styles.input, styles.gramInput]}
          keyboardType="numeric"
          value={gramText}
          onChangeText={setGramText}
        />
        <Text style={styles.itemName}>g</Text>
      </View>

      {result ? (
        <>
          <Text style={styles.kcal}>약 {Math.round(result.kcal)} kcal</Text>
          <Text style={styles.muted}>
            탄수화물 {round1(result.carbG)}g · 단백질 {round1(result.proteinG)}g · 지방 {round1(result.fatG)}g ·
            나트륨 {Math.round(result.sodiumMg)}mg
          </Text>
        </>
      ) : (
        <Text style={styles.muted}>먹은 양을 입력해 주세요</Text>
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
  photoBox: {
    height: 220,
    borderRadius: 12,
    borderWidth: 1,
    borderStyle: 'dashed',
    borderColor: '#bbb',
    alignItems: 'center',
    justifyContent: 'center',
    overflow: 'hidden',
  },
  photo: { width: '100%', height: '100%' },
  row: { flexDirection: 'row', gap: 8, flexWrap: 'wrap' },
  gramRow: { flexDirection: 'row', alignItems: 'center', gap: 8 },
  gramInput: { width: 120 },
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
  itemOn: { borderColor: GREEN, backgroundColor: '#e8f5e9' },
  itemName: { fontSize: 16 },
  muted: { color: '#777' },
  card: { gap: 10, padding: 14, borderRadius: 12, backgroundColor: '#f6f6f6', marginTop: 8 },
  kcal: { fontSize: 32, fontWeight: '700', color: GREEN },
  btn: { paddingVertical: 8, paddingHorizontal: 14, borderRadius: 8, borderWidth: 1, borderColor: GREEN },
  btnOn: { backgroundColor: GREEN },
  btnText: { color: GREEN, fontWeight: '600' },
  btnTextOn: { color: '#fff' },
});
