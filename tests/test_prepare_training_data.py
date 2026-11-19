"""Synthetische Regressionen ohne Zugriff auf die Amazon-Datensätze."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
import io

spec = importlib.util.spec_from_file_location('prepare_training_data', Path(__file__).resolve().parent.parent / 'scripts' / 'D07_prepare_training_data.py')
m = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m
spec.loader.exec_module(m)


class SplitTests(unittest.TestCase):
    def fixtures(self, root):
        rows = []
        for c in m.KATEGORIEN:
            current = []
            for i in range(120):
                current.append(dict(category=c, rating=i % 5 + 1, title='Titel', text=f'{c} Rezension {i}',
                                    source_line=i + 1, user_id=f'u{i}', parent_asin=f'p{i}'))
            # Kategorieübergreifende, unterschiedlich bewertete große Textgruppe.
            for i in range(20):
                current[i]['text'] = 'Gemeinsamer Text'
            # Ein echtes Duplikat mit anderer Herkunft und normalisiert gleichem Text.
            duplicate = dict(current[50], source_line=501, text='  ' + current[50]['text'] + '\n')
            conflict = dict(current[50], source_line=502, rating=2)
            different_title = dict(current[50], source_line=503, title='Anderer Titel')
            current += [duplicate, conflict, different_title]
            f = root / f'{c}_enriched_50000_seed42.jsonl'
            f.write_text(''.join(json.dumps(r) + '\n' for r in current))
            rows += current
        return rows

    def test_full_pipeline_reproducibility_and_no_overwrite(self):
        with tempfile.TemporaryDirectory(prefix='nlp-split-test-') as d:
            root = Path(d)
            rows = self.fixtures(root)
            originals = {p.name: p.read_bytes() for p in root.glob('*.jsonl')}
            with redirect_stdout(io.StringIO()):
                a = m.vorbereiten(root, root / 'a')
                b = m.vorbereiten(root, root / 'b')
            self.assertEqual(a, b)
            self.assertEqual(a['eingabe'], 369)
            self.assertEqual(a['entfernt'], 3)
            self.assertEqual(a['behalten'], 366)
            self.assertEqual(a['bewertungskonfliktgruppen'], 3)
            for p in (root / 'a').iterdir():
                self.assertEqual(p.read_bytes(), (root / 'b' / p.name).read_bytes())
            with self.assertRaises(FileExistsError):
                m.vorbereiten(root, root / 'a')
            for name, content in originals.items():
                self.assertEqual((root / name).read_bytes(), content)
            text_sets, output_rows = [], []
            for s in m.SPLITS:
                records = [json.loads(x) for x in (root / 'a' / f'{s}.jsonl').read_text().splitlines()]
                output_rows += records
                text_sets.append({m.normalisiere_text(r['text']) for r in records})
            self.assertEqual(len(output_rows), 366)
            for c in m.KATEGORIEN:
                same_key = [r for r in output_rows if r['category'] == c and r['user_id'] == 'u50']
                self.assertEqual(len(same_key), 3)  # Original, Konflikt, anderer Titel
                self.assertEqual({r['rating'] for r in same_key}, {1, 2})
            self.assertFalse(text_sets[0] & text_sets[1] or text_sets[0] & text_sets[2] or text_sets[1] & text_sets[2])

    def test_validation_and_normalization(self):
        good = dict(category=m.KATEGORIEN[0], rating=5, title='', text='ok', source_line=1, user_id='u', parent_asin='p')
        m.pruefe_datensatz(good, m.KATEGORIEN[0], 'test')
        for field, value in [('rating', True), ('rating', 0), ('rating', 2.5), ('rating', float('nan')),
                             ('text', '  '), ('user_id', None), ('source_line', False), ('category', 'unknown')]:
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                m.pruefe_datensatz(dict(good, **{field: value}), m.KATEGORIEN[0], 'test')
        missing = dict(good)
        del missing['parent_asin']
        with self.assertRaises(ValueError):
            m.pruefe_datensatz(missing, m.KATEGORIEN[0], 'test')
        self.assertEqual(m.normalisiere_text(' Cafe\u0301\t gut\n'), 'Café gut')
        self.assertNotEqual(m.normalisiere_text('Gut!'), m.normalisiere_text('gut'))

    def test_invariant_checks_detect_bad_assignments(self):
        r = m.Rezension(dict(category=m.KATEGORIEN[0], rating=1, source_line=1), ('a', 1), 'same')
        s = m.Rezension(dict(category=m.KATEGORIEN[1], rating=2, source_line=2), ('b', 1), 'same')
        with self.assertRaises(ValueError):
            m.pruefe_zuordnung([r, s], [r, s], [], [[0], [1], []])
        with self.assertRaises(ValueError):
            m.pruefe_zuordnung([r, s], [r, s], [], [[0, 0], [], []])
        self.assertEqual(m.erstelle_split([r, s], 42)[0], m.erstelle_split([r, s], 42)[0])


if __name__ == '__main__':
    unittest.main(verbosity=2)
