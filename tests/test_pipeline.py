import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('pipeline',ROOT/'python/pipeline.py')
pipeline=importlib.util.module_from_spec(spec)
spec.loader.exec_module(pipeline)

class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
    def recording(self,name,content,label='0 0.5 0.5 0.4 0.4\n'):
        folder=self.root/name
        (folder/'images').mkdir(parents=True)
        (folder/'labels').mkdir()
        (folder/'images/f.png').write_bytes(content)
        (folder/'labels/f.txt').write_text(label)
        return folder
    def manifest(self,train,val):
        path=self.root/'manifest.json'
        path.write_text(json.dumps([{'directory':str(train),'split':'train'}, {'directory':str(val),'split':'val'}]))
        return path
    def test_split_and_refuse_overwrite(self):
        a=self.recording('a',b'frame a'); b=self.recording('b',b'frame b')
        manifest=self.manifest(a,b); out=self.root/'dataset'
        pipeline.split_dataset(manifest,out,'duck')
        config=json.loads((out/'dataset.yaml').read_text())
        self.assertEqual(config['names'],{'0':'duck'})
        self.assertEqual(len(list((out/'labels/train').glob('*.txt'))),1)
        with self.assertRaises(FileExistsError): pipeline.split_dataset(manifest,out,'duck')
    def test_reject_leakage_before_writing(self):
        a=self.recording('a',b'same frame'); b=self.recording('b',b'same frame')
        out=self.root/'dataset'
        with self.assertRaisesRegex(ValueError,'crosses splits'):
            pipeline.split_dataset(self.manifest(a,b),out,'bike')
        self.assertFalse(out.exists())
    def test_reject_bad_label(self):
        a=self.recording('a',b'frame a','0 2 0.5 0.4 0.4\n'); b=self.recording('b',b'frame b')
        with self.assertRaisesRegex(ValueError,'coordinates'):
            pipeline.split_dataset(self.manifest(a,b),self.root/'dataset','duck')
    def test_config_paths(self):
        if not (ROOT/'configs/project.json').is_file():
            self.skipTest('Local models and labels are optional and not distributed')
        cfg=pipeline.load_config()
        for obj in cfg['objects'].values():
            for key in ['ground_truth','matlab_model','python_model']:
                if obj[key]: self.assertTrue(pipeline.resolve(obj[key]).is_file(),obj[key])

    def test_portable_configuration(self):
        cfg=json.loads((ROOT/'configs/project.example.json').read_text())
        self.assertEqual(cfg['original_matlab_root'],'')
        self.assertEqual(cfg['rtsp_url'],'')
        self.assertEqual(set(cfg['objects']),{'bike','duck','laser'})

if __name__=='__main__': unittest.main()
