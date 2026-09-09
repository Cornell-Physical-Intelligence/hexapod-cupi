"""Verify retrieved original bytes and the full nominal outcome, without Isaac."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parent
RUN = "nominal/hexapod-fourbar-validate-20260906T114048Z-9e3f8262/"
EXPECTED = {
    "campaign.json": (35188, "d9b799612614db763c3fd8757966b5aeb56aae37828be1bd7f2b1d457cf73bde"),
    RUN + "report.json": (1026669, "9390448658b1d03f0f270522a98a54435b7ee27e6f58a586314f81ff7f6db7dc"),
    RUN + "supervisor.json": (35039, "54578c13d7ec5446df8cd05cf164c89ce9bd05c5f0e1a4e24c8993d6cde97282"),
    RUN + "container.log": (38681, "3ac6aac39676040244bd2ae9fdecc41412800cfb87e26c00755ee1e9ce8f8c29"),
    RUN + "source.SHA256SUMS": (34927, "49067b3b727f8083e0ab58d4e66ca35bf463c917bbf088d7a7b0410968fc7d4c"),
    RUN + "cpu_asset_audit.json": (39872, "5a72c9e3971541e918a21e4068c0222987d797d0ce9cdeec253c2baea0914042"),
}

def verify():
    for relative, (size, digest) in EXPECTED.items():
        data = (ROOT / relative).read_bytes()
        if len(data) != size or hashlib.sha256(data).hexdigest() != digest:
            raise ValueError(f"Retrieved evidence changed: {relative}")
    report = json.loads((ROOT / RUN / "report.json").read_text())
    if report["pass"] is not False or report["errors"] != [
        "driven: max_closure_point_m=0.000111171 exceeds 0.0001"
    ]:
        raise ValueError("Unexpected outcome")
    if (report["num_envs"], report["steps_completed"], report["driven_steps"],
            report["physics_substeps"]) != (32, 1000, 2400, 54400):
        raise ValueError("Incomplete physical coverage")
    for key in ("individual_motor_positive_minus_negative_rad",
                "driven_positive_minus_negative_response_rad"):
        if len(report[key]) != 18 or min(report[key].values()) <= 0.005:
            raise ValueError("Motor-response evidence differs")
    print(json.dumps({"verified_original_files": len(EXPECTED), "physical_pass": False,
                      "ppo_started": False, "errors": report["errors"]}))

if __name__ == "__main__":
    verify()
