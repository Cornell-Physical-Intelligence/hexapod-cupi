"""Fresh exploration settings and strict saved-policy restoration on CPU."""
from contextlib import redirect_stderr
from dataclasses import asdict, replace
import argparse
import io
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from experiments.paper_walk.learner import ActorCritic, PPOLearner, exact_equal
from experiments.paper_walk.train import LEARNING_PROBE_CASE_IDS, fresh_learner_overrides, learner_configuration, main


class TrainingConfigurationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def arguments(self, mode="train"):
        return ["--mode", mode, "--preflight-only", *[item for name in
            ("asset", "model", "geometry", "geometry-extrema", "prior", "prior-metadata", "output")
            for item in ("--"+name, "/nonexistent/"+name)]]

    def test_default_and_explicit_initial_exploration_reach_existing_actor(self):
        for requested, expected in ((None, .4), (.1, .1)):
            with self.subTest(initial_std=requested):
                cfg = learner_configuration("train", 32, requested)
                self.assertEqual(cfg.initial_std, expected)
                actor = ActorCritic(cfg)
                distribution, _ = actor.distribution(torch.zeros(1, 231))
                torch.testing.assert_close(distribution.stddev, torch.full((1, 18), expected))

    def test_invalid_cli_std_fails_before_assets_or_runtime_start(self):
        for value in ("0", "-1", "nan", "inf", "-inf", "abc"):
            with self.subTest(value=value), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as result:
                main(self.arguments()+["--initial-std="+value])
            self.assertEqual(result.exception.code, 2)

    def test_explicit_std_is_rejected_for_checkpoints_and_nontraining_modes(self):
        for mode in ("diagnostic", "replay", "video", "evaluate"):
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, "only for fresh training"):
                main(self.arguments(mode)+["--initial-std", ".1"])
        with self.assertRaisesRegex(ValueError, "only for fresh training"):
            main(self.arguments()+["--checkpoint", "/nonexistent/checkpoint.pt", "--initial-std", ".1"])

    def test_fresh_optimizer_cli_reaches_config_and_preserves_omitted_defaults(self):
        default = learner_configuration("train", 32)
        self.assertEqual((default.learning_rate,default.target_kl,default.bc_optimizer,default.bc_learning_rate,
                          default.bc_velocity_coefficient,default.bc_detach_velocity),
                         (3e-4,None,"shared",None,0.,False))
        flags=["--initial-std",".1","--learning-rate",".0001","--target-kl",".02",
               "--bc-optimizer","separate","--bc-learning-rate",".0003",
               "--bc-velocity-coefficient","1","--bc-detach-velocity"]
        with patch("experiments.paper_walk.train.fresh_learner_overrides",wraps=fresh_learner_overrides) as validate, \
             patch("experiments.paper_walk.train.verify_assets",side_effect=RuntimeError("Reached asset validation")):
            with self.assertRaisesRegex(RuntimeError,"Reached asset validation"):
                main(self.arguments()+flags)
        self.assertEqual(validate.call_args.args,("train",None))
        expected={"initial_std":.1,"learning_rate":1e-4,"target_kl":.02,
                  "bc_optimizer":"separate","bc_learning_rate":3e-4,
                  "bc_velocity_coefficient":1.,"bc_detach_velocity":True}
        self.assertEqual(validate.call_args.kwargs,expected)
        cfg=learner_configuration("train",32,**validate.call_args.kwargs)
        for key,value in expected.items():self.assertEqual(getattr(cfg,key),value)

    def test_explicit_optimizer_flags_cannot_reconfigure_checkpoint_or_nontraining_mode(self):
        settings=(("--learning-rate",".0001"),("--target-kl",".02"),
                  ("--bc-optimizer","shared"),("--bc-learning-rate",".0003"),
                  ("--bc-velocity-coefficient","0"))
        for flag,value in settings:
            for mode in ("diagnostic","replay","video","evaluate"):
                with self.subTest(flag=flag,mode=mode),self.assertRaisesRegex(ValueError,"only for fresh training"):
                    main(self.arguments(mode)+[flag,value])
            with self.subTest(flag=flag,mode="resume"),self.assertRaisesRegex(ValueError,"only for fresh training"):
                main(self.arguments()+["--checkpoint","/nonexistent/checkpoint.pt",flag,value])
        for flag in ("--bc-detach-velocity","--no-bc-detach-velocity"):
            for mode in ("diagnostic","replay","video","evaluate"):
                with self.subTest(flag=flag,mode=mode),self.assertRaisesRegex(ValueError,"only for fresh training"):
                    main(self.arguments(mode)+[flag])
            with self.subTest(flag=flag,mode="resume"),self.assertRaisesRegex(ValueError,"only for fresh training"):
                main(self.arguments()+["--checkpoint","/nonexistent/checkpoint.pt",flag])

    def test_optimizer_cli_invalid_numbers_and_inconsistent_bc_settings_fail_before_assets(self):
        for flag in ("--learning-rate","--target-kl","--bc-learning-rate"):
            for value in ("0","-1","nan","inf","-inf","abc"):
                with self.subTest(flag=flag,value=value),redirect_stderr(io.StringIO()),self.assertRaises(SystemExit) as result:
                    main(self.arguments()+[flag+"="+value])
                self.assertEqual(result.exception.code,2)
        for key in ("learning_rate","target_kl","bc_learning_rate"):
            with self.subTest(key=key),self.assertRaises(argparse.ArgumentTypeError):
                fresh_learner_overrides("train",**{key:True})
        for flags in (["--bc-optimizer","separate"],
                      ["--bc-learning-rate",".0001"],
                      ["--learning-rate",".0001","--bc-optimizer","shared","--bc-learning-rate",".0003"]):
            with self.subTest(flags=flags),self.assertRaisesRegex(ValueError,"BC optimizer"):
                main(self.arguments()+flags)
        shared=learner_configuration("train",32,learning_rate=1e-4,bc_optimizer="shared",bc_learning_rate=1e-4)
        self.assertEqual(shared.bc_learning_rate,shared.learning_rate)

    def test_bc_zero_coefficient_and_explicit_false_remain_distinct_from_omitted_settings(self):
        with patch("experiments.paper_walk.train.fresh_learner_overrides",wraps=fresh_learner_overrides) as validate, \
             patch("experiments.paper_walk.train.verify_assets",side_effect=RuntimeError("Reached asset validation")):
            with self.assertRaisesRegex(RuntimeError,"Reached asset validation"):
                main(self.arguments()+["--bc-velocity-coefficient","0","--no-bc-detach-velocity"])
        provided=fresh_learner_overrides(*validate.call_args.args,**validate.call_args.kwargs)
        self.assertEqual(provided,{"bc_velocity_coefficient":0.,"bc_detach_velocity":False})
        for value in ("-1","nan","inf","-inf","abc"):
            with self.subTest(value=value),redirect_stderr(io.StringIO()),self.assertRaises(SystemExit) as result:
                main(self.arguments()+["--bc-velocity-coefficient="+value])
            self.assertEqual(result.exception.code,2)
        with self.assertRaises(argparse.ArgumentTypeError):
            fresh_learner_overrides("train",bc_velocity_coefficient=True)
        for value in (0,1,"false"):
            with self.subTest(value=value),self.assertRaisesRegex(ValueError,"must be a boolean"):
                fresh_learner_overrides("train",bc_detach_velocity=value)

    def test_probe_cli_names_match_canonical_helper_and_valid_order_reaches_asset_validation(self):
        from experiments.paper_walk.evaluate import learning_probe_cases
        self.assertEqual(LEARNING_PROBE_CASE_IDS,tuple(c["case_id"] for c in learning_probe_cases()))
        for selected in ([],["--probe-cases",LEARNING_PROBE_CASE_IDS[-1],LEARNING_PROBE_CASE_IDS[0]]):
            with self.subTest(selected=selected),patch("experiments.paper_walk.train.verify_assets",
                    side_effect=RuntimeError("Reached asset validation")) as verify:
                with self.assertRaisesRegex(RuntimeError,"Reached asset validation"):
                    main(self.arguments("evaluate")+["--eval-scope","learning"]+selected)
                verify.assert_called_once()

    def test_probe_cli_rejects_unknown_empty_duplicate_and_wrong_scope_before_assets(self):
        for values in ([],["unknown"]):
            with self.subTest(values=values),redirect_stderr(io.StringIO()),self.assertRaises(SystemExit) as result:
                main(self.arguments("evaluate")+["--eval-scope","learning","--probe-cases"]+values)
            self.assertEqual(result.exception.code,2)
        first=LEARNING_PROBE_CASE_IDS[0]
        with self.assertRaisesRegex(ValueError,"Duplicate"):
            main(self.arguments("evaluate")+["--eval-scope","learning","--probe-cases",first,first])
        for mode,scope in (("train","learning"),("diagnostic","learning"),("replay","learning"),
                           ("video","learning"),("evaluate","diagnostic"),("evaluate","full")):
            with self.subTest(mode=mode,scope=scope),self.assertRaisesRegex(ValueError,"requires --mode evaluate"):
                main(self.arguments(mode)+["--eval-scope",scope,"--probe-cases",first])
        with self.assertRaisesRegex(ValueError,"outside the selected"):
            main(self.arguments("evaluate")+["--eval-scope","learning","--probe-cases",first,
                "--probe-video-case",LEARNING_PROBE_CASE_IDS[-1]])

    def test_checkpoint_config_and_learned_std_survive_resume_and_single_env_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            states = np.random.default_rng(2).normal(size=(8, 61)).astype(np.float32)
            prior = root/"prior.npz"
            np.savez(prior, states=states, next_states=states+.01)
            cfg = replace(learner_configuration("train", 2, .1, learning_rate=1e-4,target_kl=.02,
                    bc_optimizer="separate",bc_learning_rate=3e-4,bc_velocity_coefficient=1.,bc_detach_velocity=True),
                actor_hidden=(8,), memory_hidden=(8,), estimator_hidden=(8,),
                critic_hidden=(8,), discriminator_hidden=(8,))
            original = PPOLearner(object(), prior, root/"original", cfg, device="cpu")
            # A trained log_std is distinct from its initialization parameter.
            with torch.no_grad():
                original.model.log_std.fill_(math.log(.17))
            path = root/"checkpoint.pt"
            original.save(path)
            payload = torch.load(path, map_location="cpu", weights_only=False)
            for mode, count in (("train", 2), ("video", 1), ("evaluate", 1)):
                with self.subTest(mode=mode):
                    restored_cfg = learner_configuration(mode, count, checkpoint=payload)
                    self.assertEqual(asdict(restored_cfg), payload["config"])
                    restored = PPOLearner(object(), prior, root/mode, restored_cfg, device="cpu")
                    restored.load(path)
                    self.assertTrue(exact_equal(original.model.state_dict(), restored.model.state_dict()))
                    torch.testing.assert_close(restored.model.log_std.exp(), torch.full((18,), .17))
                    obs = torch.zeros(count, 231)
                    torch.testing.assert_close(original.act(obs, deterministic=True),
                                               restored.act(obs, deterministic=True), rtol=0, atol=0)
            with self.assertRaisesRegex(ValueError, "replica count"):
                learner_configuration("train", 32, checkpoint=payload)
            with self.assertRaisesRegex(ValueError, "only for fresh training"):
                learner_configuration("train", 2, .4, payload)
            for key,value in (("learning_rate",1e-4),("target_kl",.02),("bc_optimizer","separate"),("bc_learning_rate",3e-4),
                              ("bc_velocity_coefficient",0.),("bc_detach_velocity",False)):
                with self.subTest(override=key),self.assertRaisesRegex(ValueError,"only for fresh training"):
                    learner_configuration("train",2,checkpoint=payload,**{key:value})
            incomplete={**payload,"config":{key:value for key,value in payload["config"].items() if key!="target_kl"}}
            with self.assertRaisesRegex(ValueError,"implicit migration is unsupported"):
                learner_configuration("evaluate",1,checkpoint=incomplete)
            wrong = PPOLearner(object(), prior, root/"wrong", replace(cfg, initial_std=.4), device="cpu")
            with self.assertRaisesRegex(ValueError, "model/config/prior"):
                wrong.load(path)


if __name__ == "__main__":
    unittest.main()
