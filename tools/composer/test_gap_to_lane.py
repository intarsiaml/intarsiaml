"""Tests for gap_to_lane.py. Run: python3 -m unittest discover -s tools/composer -p 'test_*.py'

The sample files in testdata/ are cells copied from model-benchmarks composer/data/regimes.json and catalog.json
(2026-10-22). The live test runs only when that checkout is present."""
import json
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gap_to_lane as g  # noqa: E402

SAMPLE = json.loads((HERE / "testdata" / "regimes-sample.json").read_text())
CATALOG = json.loads((HERE / "testdata" / "catalog-sample.json").read_text())
TECH = "vllm-gb10-w8a8-fp8-projection-gemm"


def contender(engine, quant, value, spread=1.0, quality="ok"):
    return {"config": f"{engine} · {quant}", "engine": engine, "quant": quant, "tp": None, "value": value,
            "spread": spread, "noisy": False, "run_id": f"run-{engine}-{value}", "version": "", "n_runs": 1, "quality": quality}


def cell(phase, bucket, contenders, lane="plain"):
    w = max(contenders, key=lambda c: c["value"])
    return {"id": f"HW · Fam | {lane} | {phase} | {bucket}", "setup": "HW · Fam", "hardware": "HW", "family": "Fam",
            "lane": lane, "phase": phase, "bucket": bucket, "winner": w["config"], "winner_engine": w["engine"],
            "winner_value": w["value"], "winner_run": w["run_id"], "winner_quality": w["quality"], "contenders": contenders}


class Rules(unittest.TestCase):
    def test_known_gap_is_found_with_its_technique(self):
        res = g.analyse(SAMPLE, CATALOG)
        gaps = {(e["phase"], e["bucket"]): e for e in res["gaps"] if e["setup"] == "DGX Spark GB10 · Qwen3.6-35B-A3B"}
        self.assertEqual(gaps[("prefill", "4k")]["margin_pct"], 16.2)
        self.assertEqual(gaps[("prefill", "16k")]["margin_pct"], 15.9)
        for k in (("prefill", "4k"), ("prefill", "16k")):
            self.assertIn(TECH, [t["id"] for t in gaps[k]["techniques"]])
            self.assertEqual(gaps[k]["rival"]["engine"], "vLLM")
            self.assertTrue(gaps[k]["intarsia"]["quality"].startswith("drop:"))

    def test_margin_is_against_intarsia_own_quant_not_the_winners(self):
        # tatooine: the winner is MXFP4, IntarsiaML is NVFP4 and the only other contender is MXFP4: no gap, quant-only
        res = g.analyse(SAMPLE, CATALOG)
        self.assertFalse([e for e in res["gaps"] if e["hardware"].startswith("Radeon")])
        self.assertTrue([e for e in res["unmatched_quant"] if e["hardware"].startswith("Radeon")])

    def test_solo_cells_are_measure_first(self):
        res = g.analyse(SAMPLE, CATALOG)
        solo = [e for e in res["measure_first"] if e["kind"] == "solo" and e["hardware"] == "2x DGX Spark GB10 (TP2)"]
        self.assertTrue(solo)

    def test_tie_and_ahead_and_weights(self):
        me = contender("IntarsiaML", "NVFP4", 100.0)
        cs = [cell("decode", "4k", [me, contender("vLLM", "NVFP4", 103.0)]),     # +3 % < 5 % floor: tie
              cell("decode", "16k", [me, contender("vLLM", "NVFP4", 90.0)]),     # IntarsiaML ahead: not listed
              cell("concurrent", "4x", [me, contender("vLLM", "NVFP4", 150.0)]),
              cell("decode", "1k", [me, contender("vLLM", "NVFP4", 150.0)])]
        res = g.analyse(cs, [])
        self.assertEqual([(e["phase"], e["bucket"]) for e in res["ties"]], [("decode", "4k")])
        self.assertEqual([(e["phase"], e["bucket"]) for e in res["gaps"]], [("decode", "1k"), ("concurrent", "4x")])
        self.assertEqual(res["gaps"][0]["score"], 50.0)
        self.assertEqual(res["gaps"][1]["score"], 25.0)  # concurrent weight 0.5

    def test_noise_uses_spreads(self):
        me = contender("IntarsiaML", "NVFP4", 100.0, spread=12.0)
        res = g.analyse([cell("decode", "4k", [me, contender("vLLM", "NVFP4", 110.0)])], [])
        self.assertEqual(res["gaps"], [])
        self.assertEqual(res["ties"][0]["noise_pct"], 12.0)

    def test_empty_quant_is_never_a_match(self):
        me = contender("IntarsiaML", "", 100.0)
        res = g.analyse([cell("decode", "4k", [me, contender("vLLM", "", 200.0)])], [])
        self.assertEqual((res["gaps"], res["ties"]), ([], []))
        self.assertEqual(len(res["unknown_quant"]), 1)

    def test_spec_lane_is_skipped_unless_asked(self):
        me = contender("IntarsiaML", "NVFP4", 100.0)
        cs = [cell("decode", "4k", [me, contender("vLLM", "NVFP4", 200.0)], lane="spec")]
        self.assertEqual(g.analyse(cs, [])["gaps"], [])
        self.assertEqual(len(g.analyse(cs, [], lane="spec")["gaps"]), 1)

    def test_unmeasured_cell_in_an_intarsia_setup(self):
        me = contender("IntarsiaML", "NVFP4", 100.0)
        cs = [cell("decode", "4k", [me, contender("vLLM", "NVFP4", 90.0)]),
              cell("prefill", "4k", [contender("vLLM", "NVFP4", 90.0)])]
        res = g.analyse(cs, [])
        self.assertEqual([(e["kind"], e["phase"]) for e in res["measure_first"]], [("unmeasured", "prefill")])

    def test_render_and_json_roundtrip(self):
        res = g.analyse(SAMPLE, CATALOG)
        src = {"regimes": "r", "regimes_sha": "0", "catalog": "c", "catalog_sha": "0"}
        md = g.render(res, g.REPO, src)
        self.assertIn(TECH, md)
        self.assertIn("Measure first", md)
        json.dumps(res)


class Live(unittest.TestCase):
    @unittest.skipUnless(g.DEFAULT_REGIMES.exists() and g.DEFAULT_CATALOG.exists(), "model-benchmarks checkout not present")
    def test_live_regimes_contain_the_gb10_prefill_gap(self):
        res = g.analyse(json.loads(g.DEFAULT_REGIMES.read_text()), json.loads(g.DEFAULT_CATALOG.read_text()))
        hit = {(e["phase"], e["bucket"]): e for e in res["gaps"] + res["ties"]
               if e["setup"] == "DGX Spark GB10 · Qwen3.6-35B-A3B"}
        for k in (("prefill", "4k"), ("prefill", "16k")):
            self.assertIn(k, hit, "known gap missing from live data")
            self.assertEqual(hit[k]["verdict"], "behind")
            self.assertIn(TECH, [t["id"] for t in hit[k]["techniques"]])
        self.assertAlmostEqual(hit[("prefill", "4k")]["margin_pct"], 16.2, delta=1.0)
        self.assertAlmostEqual(hit[("prefill", "16k")]["margin_pct"], 15.9, delta=1.0)


if __name__ == "__main__":
    unittest.main()
