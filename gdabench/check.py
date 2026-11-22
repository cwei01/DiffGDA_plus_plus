import unittest
from types import SimpleNamespace
import numpy as np
import torch
from scipy.sparse import coo_matrix
from model.joint_graph import GraphHead
from gdabench.head_mmd import head_logits_features, head_features, generated_target_mmd
from gdabench.sampling import head_logits
from model.pp_training import rbf_mmd

class GeneratedMMDTests(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(19)
        torch.set_num_threads(1)
        self.pairs = torch.tensor([[0, 1, 2, 2], [1, 2, 3, 4]])
        self.values = torch.tensor([0.2, 0.7, 1.0, 0.4])
        (a, b) = self.pairs.numpy()
        v = self.values.numpy()
        self.adj = coo_matrix((np.r_[v, v], (np.r_[a, b], np.r_[b, a])), shape=(6, 6)).tocsr()
        self.z = np.random.default_rng(2).normal(size=(6, 4)).astype(np.float32)

    def test_full_neighbor_hidden_logits_losses_and_gradients(self):
        h = GraphHead(4, 3, 0.2).eval()
        roots = np.array([4, 0, 5])
        (logits, features, _) = head_logits_features(h, self.z, self.adj, roots, np.random.default_rng(1), None)
        target = head_features(h, self.z + 0.4, self.adj, roots, np.random.default_rng(2), None)
        loss = torch.nn.functional.cross_entropy(logits, torch.tensor([1, 0, 2])) + 0.1 * rbf_mmd(features, target)
        loss.backward()
        grads = [p.grad.clone() for p in h.parameters()]
        h.zero_grad()
        (ref, rf) = h(torch.from_numpy(self.z), self.pairs, self.values, return_features=True)
        tf = h.features(torch.from_numpy(self.z + 0.4), self.pairs, self.values)
        reference = torch.nn.functional.cross_entropy(ref[roots], torch.tensor([1, 0, 2])) + 0.1 * rbf_mmd(rf[roots], tf[roots])
        reference.backward()
        torch.testing.assert_close(loss, reference)
        torch.testing.assert_close(features, rf[roots])
        torch.testing.assert_close(logits, ref[roots])
        for (a, p) in zip(grads, h.parameters()):
            torch.testing.assert_close(a, p.grad, atol=2e-06, rtol=2e-05)

    def test_sampled_ce_same_rng_and_mmd_has_head_gradient(self):
        h = GraphHead(4, 3, 0.2).eval()
        roots = np.array([0, 2, 5])
        rng1 = np.random.default_rng(3)
        rng2 = np.random.default_rng(3)
        (a, f, _) = head_logits_features(h, self.z, self.adj, roots, rng1, 1)
        (b, _) = head_logits(h, self.z, self.adj, roots, rng2, 1)
        torch.testing.assert_close(a, b)
        self.assertEqual(rng1.bit_generator.state, rng2.bit_generator.state)
        cfg = dict(seed=1, batch=3, fanout=None)
        args = SimpleNamespace(encoder_mmd_weight=0.1, encoder_mmd_max_samples=3)
        loss = generated_target_mmd(h, f, self.z + 0.6, self.adj, roots, args, cfg, 7)
        loss.backward()
        self.assertGreater(float(h.first.linear.weight.grad.norm()), 0)
        self.assertIsNone(h.second.linear.weight.grad)
        h.zero_grad()
        args.encoder_mmd_weight = 0
        self.assertEqual(float(generated_target_mmd(h, f, None, None, None, args, cfg, 7)), 0)

    def test_target_adaptation_roots_only_repeatable(self):
        h = GraphHead(4, 3, 0).eval()
        roots = np.array([0, 2])
        (_, f, _) = head_logits_features(h, self.z, self.adj, roots, np.random.default_rng(1), None)
        cfg = dict(seed=1, batch=2, fanout=None)
        args = SimpleNamespace(encoder_mmd_weight=0.1, encoder_mmd_max_samples=2)
        a = generated_target_mmd(h, f, self.z + 0.3, self.adj, roots, args, cfg, 4)
        b = generated_target_mmd(h, f, self.z + 0.3, self.adj, roots, args, cfg, 4)
        torch.testing.assert_close(a, b, atol=0, rtol=0)

def validate_math():
    result=unittest.TestResult()
    unittest.defaultTestLoader.loadTestsFromTestCase(GeneratedMMDTests).run(result)
    if not result.wasSuccessful():raise RuntimeError(str(result.errors+result.failures))
    return dict(tests=result.testsRun,status='passed')
