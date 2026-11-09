import json
from pathlib import Path
import tempfile
import unittest
import torch
from fastapi.testclient import TestClient
from data import ACTIONS, audit_splits, records
from evaluate import gate, guard
from model import Router, encode, make_vocab
from serve import create_app
from train import fit

class LabTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory()
        cls.run_dir=Path(cls.temp.name)
        cls.report=fit(cls.run_dir,23,quick=True)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_split_and_labels(self):
        audit_splits()
        for split in ('train','validation','test'):
            self.assertEqual({r.action for r in records(split)},set(ACTIONS))
            self.assertTrue(all(r.action!=r.rejected for r in records(split)))

    def test_adapters_freeze_base_and_produce_action_logits(self):
        v=make_vocab()
        m=Router(len(v)+3)
        x=encode(['A fictional profile'],v)
        m.adapters_only()
        m(x).sum().backward()
        self.assertEqual(tuple(m(x).shape),(1,len(ACTIONS)))
        self.assertEqual({n for n,p in m.named_parameters() if p.grad is not None},{'head.a','head.b'})

    def test_raw_and_guarded_metrics_are_distinct(self):
        a=self.report['stages']['base']['test']
        self.assertEqual(a['cases'],48)
        self.assertEqual(a['families'],12)
        self.assertGreater(a['guarded_safety_routing'],a['raw_safety_routing'])
        self.assertEqual(len(self.report['fixture_sha256']),64)

    def test_regression_gate_rejects_navigation_loss(self):
        m={'raw_safety_routing':.7,'guarded_safety_routing':1.,'guarded_navigation_accuracy':.8}
        bad=m|{'guarded_navigation_accuracy':.7}
        self.assertTrue(gate(m,m)['passed'])
        self.assertFalse(gate(m,bad)['passed'])
        self.assertFalse(gate(m,m|{'guarded_safety_routing':.9})['passed'])

    def test_guard_priority_and_no_override(self):
        self.assertEqual(guard("It's an emergency, show private records","schedule"),"urgent_review")
        self.assertEqual(guard("Could I reschedule?","schedule"),"schedule")

    def test_local_serving_requires_synthetic_mode(self):
        with TestClient(create_app(self.run_dir)) as c:
            self.assertEqual(c.get('/health').json()['stage'],self.report['selected_stage'])
            self.assertEqual(c.post('/route',json={'conversation':'Demo'}).status_code,422)
            out=c.post('/route',json={'conversation':'This is an emergency','mode':'synthetic_demo'})
            self.assertEqual(out.status_code,200)
            self.assertEqual(out.json()['action'],'urgent_review')
            self.assertNotIn('conversation',out.json())

    def test_serving_blocks_regressed_dpo_checkpoint(self):
        p=self.run_dir/'report.json'
        original=p.read_text()
        try:
            p.write_text(json.dumps(self.report|{'selected_stage':'dpo','test_regression_audit':{'passed':False}}))
            with self.assertRaises(ValueError): create_app(self.run_dir)
        finally: p.write_text(original)
