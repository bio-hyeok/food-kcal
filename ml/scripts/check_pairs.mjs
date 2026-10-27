// 분류용 이미지 ↔ 라벨(xml/txt) 짝 점검.
// 사용: node scripts/check_pairs.mjs "<음식데이터 폴더>" [출력 csv]
//
// 짝 = 파일명(확장자 제외)이 같은 이미지와 라벨. 코드(8자리) 폴더별로 센다.
// 결과 csv는 기본으로 data/pair_report.csv (Git 제외 폴더)에 쓴다.
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const repo = path.resolve(here, '..', '..');
const root = process.argv[2];
if (!root) {
  console.error('사용법: node scripts/check_pairs.mjs "<음식데이터 폴더>" [출력 csv]');
  process.exit(1);
}
const outCsv = process.argv[3] ?? path.join(repo, 'data', 'pair_report.csv');
const nfc = (s) => s.normalize('NFC');
const IMG = /\.(jpe?g|png)$/i;
const stem = (f) => f.replace(/\.[^.]+$/, '');

const dirs = (d) => fs.readdirSync(d, { withFileTypes: true }).filter((e) => e.isDirectory()).map((e) => e.name);
const files = (d) => fs.readdirSync(d, { withFileTypes: true }).filter((e) => e.isFile()).map((e) => e.name);

// 1) 이미지: [원천]음식분류_* 아래에서 8자리 코드 폴더를 찾는다 (깊이 1~2)
const imgs = new Map(); // code -> Set(stem)
const tops = fs.readdirSync(root).filter((n) => nfc(n).startsWith('[원천]음식분류'));
for (const t of tops) {
  const base = path.join(root, t);
  const walk = (d, depth) => {
    for (const n of dirs(d)) {
      const p = path.join(d, n);
      if (/^\d{8}$/.test(n)) {
        const s = imgs.get(n) ?? new Set();
        for (const f of files(p)) if (IMG.test(f)) s.add(stem(f));
        imgs.set(n, s);
      } else if (depth < 2) walk(p, depth + 1);
    }
  };
  walk(base, 0);
}

// 2) 라벨: [라벨]음식분류_VAL/{xml,txt}/<코드>/
const labDir = fs.readdirSync(root).find((n) => nfc(n).startsWith('[라벨]음식분류'));
const xmls = new Map();
const txts = new Map();
for (const [kind, store] of [['xml', xmls], ['txt', txts]]) {
  const kd = path.join(root, labDir, kind);
  for (const code of dirs(kd)) store.set(code, new Set(files(path.join(kd, code)).map(stem)));
}

// 3) 코드별 집계
const codes = [...new Set([...imgs.keys(), ...xmls.keys(), ...txts.keys()])].sort();
const rows = [['code', 'images', 'xml', 'txt', 'paired_xml', 'image_only', 'xml_only']];
const tot = { images: 0, xml: 0, txt: 0, paired: 0, imgOnly: 0, xmlOnly: 0 };
const noImg = [];
const noLabel = [];
for (const c of codes) {
  const i = imgs.get(c) ?? new Set();
  const x = xmls.get(c) ?? new Set();
  const t = txts.get(c) ?? new Set();
  let paired = 0;
  for (const s of i) if (x.has(s)) paired++;
  const imgOnly = i.size - paired;
  const xmlOnly = x.size - paired;
  rows.push([c, i.size, x.size, t.size, paired, imgOnly, xmlOnly]);
  tot.images += i.size; tot.xml += x.size; tot.txt += t.size; tot.paired += paired; tot.imgOnly += imgOnly; tot.xmlOnly += xmlOnly;
  if (i.size === 0) noImg.push(c);
  if (x.size === 0) noLabel.push(c);
}
fs.mkdirSync(path.dirname(outCsv), { recursive: true });
fs.writeFileSync(outCsv, rows.map((r) => r.join(',')).join('\n') + '\n');

console.log(`코드 ${codes.length}개 (이미지 폴더 ${imgs.size}, xml 폴더 ${xmls.size}, txt 폴더 ${txts.size})`);
console.log(`이미지 ${tot.images} / xml ${tot.xml} / txt ${tot.txt}`);
console.log(`짝(이미지+xml) ${tot.paired} / 이미지만 ${tot.imgOnly} / xml만 ${tot.xmlOnly}`);
console.log(`이미지 폴더가 없는 코드 ${noImg.length}개:`, noImg.slice(0, 10).join(' '));
console.log(`xml 라벨이 없는 코드 ${noLabel.length}개:`, noLabel.slice(0, 10).join(' '));
console.log(`→ ${path.relative(repo, outCsv)}`);
