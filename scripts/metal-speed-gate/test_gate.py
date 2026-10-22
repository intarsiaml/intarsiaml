"""Unit tests for the clock normalisation in gate.py: python3 -m unittest scripts/metal-speed-gate/test_gate.py"""
import contextlib, io, json, os, subprocess, sys, tempfile, unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gate


class Math(unittest.TestCase):
    def test_factor_and_normalise(self):
        f = gate.state_factor(240.0, 250.0)
        self.assertAlmostEqual(f, 0.96)
        self.assertAlmostEqual(gate.normalise(33.6, f), 35.0)  # slow machine: figure is raised
        self.assertAlmostEqual(gate.normalise(36.0, gate.state_factor(270.0, 250.0)), 33.3333, places=3)  # fast: lowered
        self.assertAlmostEqual(gate.normalise(34.0, 1.0), 34.0)

    def test_factor_window(self):
        self.assertTrue(gate.factor_valid(1.0))
        self.assertTrue(gate.factor_valid(0.70) and gate.factor_valid(1.30))
        self.assertFalse(gate.factor_valid(0.69))
        self.assertFalse(gate.factor_valid(1.31))

    def test_ref_gbs_median_and_missing(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "x-ref.jsonl")
            self.assertIsNone(gate.ref_gbs(p))
            open(p, "w").write('{"gbs": 240}\nnot json\n{"gbs": 250}\n{"gbs": 300}\n')
            self.assertEqual(gate.ref_gbs(p), 250.0)


def write_arm(d, label, tok_s_4k, ref_gbs=None, mhz=1350, cpu=0.1):
    rows = [{"target_prompt_tokens": 4096, "prompt_tokens": 4096, "decode_tok_s": {"median": tok_s_4k}}]
    json.dump({"speed": {"gen_tokens": 256, "rows": rows}}, open(os.path.join(d, f"{label}-localeval.json"), "w"))
    with open(os.path.join(d, f"{label}-macmon.jsonl"), "w") as f:
        for _ in range(8):
            f.write(json.dumps({"gpu_usage": [mhz, 0.9], "cpu_usage_pct": cpu}) + "\n")
    if ref_gbs is not None:
        open(os.path.join(d, f"{label}-ref.jsonl"), "w").write(json.dumps({"gbs": ref_gbs}) + "\n")


def run_gate(d, base=250.0):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        old = sys.argv
        sys.argv = ["gate.py", d, "--ref-baseline-gbs", str(base)]
        try:
            rc = gate.main()
        finally:
            sys.argv = old
    return rc, out.getvalue()


class EndToEnd(unittest.TestCase):
    # context-aware floor at 4k/256 gen for Qwen3-8B: 614e9 / (15.1368e9 + 147456 * 4224) = 39.0 tok/s; 85 % = 33.15 tok/s
    def test_slow_state_rescued_and_visible(self):
        with tempfile.TemporaryDirectory() as d:
            write_arm(d, "intarsia-1", 32.5, ref_gbs=225.0, mhz=1180, cpu=0.6)  # 83.3 % raw, factor 0.90 -> 36.1 = 92.6 %
            rc, out = run_gate(d)
            self.assertEqual(rc, 0, out)
            self.assertIn("raw 32.50 tok/s", out)
            self.assertIn("factor 0.900", out)
            self.assertIn("normalised 36.11 tok/s", out)

    def test_real_regression_still_fails(self):
        with tempfile.TemporaryDirectory() as d:
            write_arm(d, "intarsia-1", 30.0, ref_gbs=250.0)  # healthy machine, slow code: 76.9 %
            rc, out = run_gate(d)
            self.assertEqual(rc, 1, out)

    def test_out_of_window_is_state_invalid(self):
        with tempfile.TemporaryDirectory() as d:
            write_arm(d, "intarsia-1", 20.0, ref_gbs=100.0)  # factor 0.4
            rc, out = run_gate(d)
            self.assertEqual(rc, 3, out)

    def test_no_reference_uses_legacy_admission(self):
        with tempfile.TemporaryDirectory() as d:
            write_arm(d, "intarsia-1", 34.5, mhz=1350, cpu=0.1)
            rc, out = run_gate(d)
            self.assertEqual(rc, 0, out)
            self.assertIn("not normalised", out)
            write_arm(d, "intarsia-1", 34.5, mhz=1200, cpu=0.1)
            self.assertEqual(run_gate(d)[0], 3)


class DefinitionHistory(unittest.TestCase):
    """Every gate run carries the verdict under the pre-2026-10-21 definition (raw decode against the flat 85 % of
    bandwidth / weights = 34.48 tok/s) next to the current one, in the output and in gate-verdict.json."""

    def record(self, d):
        j = json.load(open(os.path.join(d, "gate-verdict.json")))
        (c,) = j["definition_changes"]
        return j, c

    def test_rescued_run_shows_old_fail_and_current_pass(self):
        with tempfile.TemporaryDirectory() as d:
            write_arm(d, "intarsia-1", 32.5, ref_gbs=225.0, mhz=1180, cpu=0.6)  # raw 32.5 < 34.48 (old FAIL); normalised 36.11 (new PASS)
            rc, out = run_gate(d)
            self.assertEqual(rc, 0, out)  # the verdict is the current definition's, unchanged
            j, c = self.record(d)
            self.assertEqual(j["verdict"], "PASS")
            self.assertEqual(c["old_def"]["verdict"], "FAIL")
            self.assertEqual(c["current_def"]["verdict"], "PASS")
            self.assertAlmostEqual(c["old_def"]["metric"], 32.5)
            self.assertAlmostEqual(c["current_def"]["metric"], 36.11, places=2)
            self.assertAlmostEqual(c["old_def"]["limit"], 0.85 * 614.0 / 15.1368, places=3)
            self.assertEqual(c["changed_on"], "2026-10-21")
            self.assertTrue(any("12:25" in r for r in c["log_ref"]))
            self.assertIn("gate-definition: metal_decode_floor[4096]: old def: FAIL 32.50 tok/s", out)
            self.assertIn("| current def: PASS 36.11 tok/s", out)

    def test_state_invalid_run_shows_both(self):
        with tempfile.TemporaryDirectory() as d:
            write_arm(d, "intarsia-1", 20.0, ref_gbs=100.0)  # factor 0.4: no valid arm
            rc, out = run_gate(d)
            self.assertEqual(rc, 3, out)
            _, c = self.record(d)
            self.assertEqual(c["old_def"]["verdict"], "FAIL")
            self.assertEqual(c["current_def"]["verdict"], "STATE-INVALID")
            self.assertIsNone(c["current_def"]["metric"])

    def test_healthy_run_passes_under_both(self):
        with tempfile.TemporaryDirectory() as d:
            write_arm(d, "intarsia-1", 36.0, ref_gbs=250.0)  # 36.0 >= 34.48 and >= 33.15
            rc, out = run_gate(d)
            self.assertEqual(rc, 0, out)
            _, c = self.record(d)
            self.assertEqual((c["old_def"]["verdict"], c["current_def"]["verdict"]), ("PASS", "PASS"))

    def test_real_regression_fails_under_both(self):
        with tempfile.TemporaryDirectory() as d:
            write_arm(d, "intarsia-1", 30.0, ref_gbs=250.0)
            rc, _ = run_gate(d)
            self.assertEqual(rc, 1)
            _, c = self.record(d)
            self.assertEqual((c["old_def"]["verdict"], c["current_def"]["verdict"]), ("FAIL", "FAIL"))


if __name__ == "__main__":
    unittest.main()
