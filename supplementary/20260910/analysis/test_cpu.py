"""Small CPU checks on synthetic poses; no dataset or model weights needed."""
import json,subprocess,sys,tempfile,unittest
from pathlib import Path
import numpy as np
import skeleton_metrics_cpu as sm
import projection_core as core
from run_postprocess import topology,project_source

HERE=Path(__file__).resolve().parent

class Tests(unittest.TestCase):
    def test_direction_fallback_records_future_and_past_tie(self):
        vectors=np.array([[0.,0,0],[1.,0,0],[0.,0,0],[0.,1,0]])
        directions,events=core.resolve_directions(vectors,np.array([0.,0,1]),'segment')
        self.assertTrue(events[0]['uses_future'])
        self.assertEqual(events[1]['source_frame'],1)
        np.testing.assert_array_equal(directions[0],[1,0,0])
        _,fallback=core.resolve_directions(np.zeros((2,3)),np.array([0.,0,1]),'segment')
        self.assertTrue(all(e['fallback']=='training_reference_direction' for e in fallback))

    def test_csl_mapping_preserves_the_same_projection(self):
        configs={d:sm.validate_config(json.loads((HERE/'configurations'/f'{d}_segments.json').read_text())) for d in ('phix','csl')}
        edge=topology(configs['phix']);rng=np.random.default_rng(7)
        canonical=rng.normal(0,.1,(6,178,3));reference=core.fit_reference([('train',canonical,{})],configs['phix'],edge)
        perm=np.array(configs['csl']['canonical_to_source']);source=np.empty_like(canonical);source[:,perm]=canonical
        for treatment in core.TREATMENTS:
            expected,events,_=core.project_pose(canonical,reference,edge,treatment)
            actual,other=project_source(source,reference,edge,perm,treatment)
            np.testing.assert_array_equal(actual[:,perm],expected)
            np.testing.assert_array_equal(actual[:,8:136],source[:,8:136].astype(np.float32))
            self.assertEqual(events,other)

    def test_complete_cli_both_layouts_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);rng=np.random.default_rng(42)
            poses=rng.normal(0,.1,(2,6,178,3))
            for ds in ('phix','csl'):
                config=json.loads((HERE/'configurations'/f'{ds}_segments.json').read_text());perm=np.array(config['canonical_to_source'])
                source=np.empty_like(poses);source[:,:,perm]=poses
                train=root/f'{ds}_train.npz';pred=root/f'{ds}_pred.npz'
                np.savez(train,poses=source,sample_ids=np.array(['a','b']))
                np.savez(pred,poses=source*1.1,sample_ids=np.array(['a','b']))
                out=root/ds
                cmd=[sys.executable,str(HERE/'run_postprocess.py'),'--dataset',ds,'--train',str(train),'--pred',str(pred),'--out',str(out),'--diagnostics']
                result=subprocess.run(cmd,capture_output=True,text=True)
                self.assertEqual(result.returncode,0,result.stderr)
                manifest=json.loads((out/'manifest.json').read_text());self.assertEqual(manifest['sample_ids'],['a','b'])
                for treatment in core.TREATMENTS:
                    rows=list(sm.load_samples(out/(treatment+'.pickle')))
                    self.assertEqual([(sid,len(pose)) for sid,pose,_ in rows],[('a',6),('b',6)])
                    report=json.loads((out/('diagnostics_'+treatment)/'summary.json').read_text())
                    self.assertEqual(report['results']['valid_samples'],2)
                self.assertNotEqual(subprocess.run(cmd,capture_output=True).returncode,0)

if __name__=='__main__':unittest.main()
