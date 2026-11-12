"""Backend smoke tests; these do not require SAM2 weights or COCO images.
Run: python -m unittest discover -s tests -v
"""
import json
import unittest
from unittest.mock import patch
import numpy as np
from fastapi.testclient import TestClient
from demo.app import app, metrics, mask_as_overlay

client = TestClient(app)


class DemoSmokeTest(unittest.TestCase):
    def test_serves_website(self):
        response = client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn('Interactive Segmentation Lab', response.text)
        self.assertEqual(client.get('/static/styles.css').status_code, 200)
        self.assertEqual(client.get('/static/app.js').status_code, 200)

    def test_saved_final_results(self):
        r = client.get('/api/results')
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertEqual(data['point']['n_objects'], 6941)
        self.assertAlmostEqual(data['box']['mean_iou'] - data['point']['mean_iou'], 0.16098885627747553)

    def test_metrics_and_png_mask(self):
        a = np.array([[1, 1], [0, 0]], dtype=bool)
        b = np.array([[1, 0], [1, 0]], dtype=bool)
        out = metrics(a, b)
        self.assertAlmostEqual(out['iou'], 1/3)
        self.assertAlmostEqual(out['dice'], 0.5)
        self.assertTrue(mask_as_overlay(a, (255, 0, 0)).startswith('data:image/png;base64,'))

    def test_infer_stream_with_fake_research_runner(self):
        image = np.zeros((6, 8, 3), dtype=np.uint8)
        gt = np.zeros((6, 8), dtype=bool)
        gt[1:5, 2:6] = True
        pt = np.zeros((6, 8), dtype=bool)
        pt[2:5, 3:6] = True
        bx = gt.copy()
        ann = {'image_id': 12, 'bbox': [2, 1, 4, 4]}
        fake = {
            'point': {'mask': pt, 'score': .8, 'prompt_ms': 9.5},
            'box': {'mask': bx, 'score': .9, 'prompt_ms': 8.5},
        }
        with patch('demo.app.store.record', return_value=(ann, image, gt, 'person')), \
             patch('demo.app.default_prompt', return_value=([3, 3], [2, 1, 6, 5])), \
             patch('demo.app.model.run', return_value=(fake, 320., False, 'mps:0')):
            r = client.post('/api/infer-stream', json={'annotation_id': 19, 'mode': 'compare'})
        self.assertEqual(r.status_code, 200)
        messages = [json.loads(chunk[len('data: '):]) for chunk in r.text.split('\n\n') if chunk.startswith('data: ')]
        self.assertEqual(messages[-1]['type'], 'result')
        final = messages[-1]
        self.assertEqual(final['source'], 'live')
        self.assertAlmostEqual(final['results']['box']['iou'], 1.0)
        self.assertAlmostEqual(final['results']['point']['iou'], 9/16)
        self.assertTrue(final['results']['box']['mask'].startswith('data:image/png;base64,'))
        self.assertTrue(any(m.get('stage') == 'evaluation' for m in messages))

    def test_saved_automated_case_is_explicitly_labeled(self):
        image = np.zeros((6, 8, 3), dtype=np.uint8)
        gt = np.ones((6, 8), dtype=bool)
        ann = {'image_id': 12, 'bbox': [0, 0, 8, 6]}
        fake = {'box': {'mask': gt, 'score': .92, 'prompt_ms': 7.2}}
        with patch('demo.app.store.record', return_value=(ann, image, gt, 'test')), \
             patch('demo.app.default_prompt', return_value=([3, 3], [0, 0, 8, 6])), \
             patch('demo.app.saved_outputs', return_value=fake):
            r = client.post('/api/infer-stream', json={'annotation_id': 19, 'mode': 'box', 'use_saved': True})
        self.assertIn('"source": "saved"', r.text)
        self.assertIn('not live inference', r.text)


if __name__ == '__main__':
    unittest.main()
