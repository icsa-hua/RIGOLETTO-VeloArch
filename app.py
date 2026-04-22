"""
VeloArch Web App — run with:
    conda activate veloarch && python app.py
then open http://localhost:5000
"""

import json
import os
import tempfile

from flask import Flask, jsonify, render_template, request
from werkzeug.utils import secure_filename

from src.engine import (
    Architecture,
    Evaluator,
    ModelParser,
    PRESET_ARCHITECTURES,
    Workload,
    load_model_from_path,
)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024 * 1024  # 2 GB

ALLOWED_WEIGHTS = {"pt", "pth"}
ALLOWED_CODE    = {"py"}


def _ext(filename: str) -> str:
    return filename.rsplit(".", 1)[-1].lower() if "." in filename else ""


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/textbook")
def textbook():
    return render_template("textbook.html")


@app.route("/api/presets")
def get_presets():
    return jsonify({
        name: Architecture.from_preset(name).to_dict()
        for name in PRESET_ARCHITECTURES
    })


@app.route("/api/evaluate", methods=["POST"])
def evaluate():
    tmp_files = []

    try:
        config = json.loads(request.form.get("config", "{}"))
        workload_cfgs = config.get("workloads", [])
        arch_cfgs     = config.get("architectures", [])

        if not workload_cfgs:
            return jsonify({"error": "No workloads configured."}), 400
        if not arch_cfgs:
            return jsonify({"error": "No architectures selected."}), 400

        # --- Architectures ---
        architectures = []
        for ac in arch_cfgs:
            if ac.get("preset"):
                architectures.append(Architecture.from_preset(ac["preset"]))
            else:
                architectures.append(Architecture.from_dict(ac))

        # --- Workloads ---
        workloads = []

        for i, wc in enumerate(workload_cfgs):
            label = f"Workload #{i+1} ({wc.get('name') or 'unnamed'})"

            # Manual workload — no file needed
            if wc.get("manual"):
                try:
                    ops = float(wc["operations"])
                    mem = float(wc["memory_bytes"])
                except (KeyError, ValueError, TypeError):
                    return jsonify({"error": f"{label}: invalid FLOPs or Memory value."}), 400
                workloads.append(Workload(
                    name=wc.get("name") or f"Workload_{i+1}",
                    operations=ops,
                    memory_bytes=mem,
                    task_type=wc.get("task_type", "custom"),
                ))
                continue

            # File-based workload
            weights_key = f"model_file_{i}"
            code_key    = f"code_file_{i}"

            if weights_key not in request.files or not request.files[weights_key].filename:
                return jsonify({"error": f"{label}: no weights file selected."}), 400

            weights_file = request.files[weights_key]
            if _ext(weights_file.filename) not in ALLOWED_WEIGHTS:
                return jsonify({
                    "error": f"{label}: weights file must be .pt or .pth "
                             f"(got '{weights_file.filename}')."
                }), 400

            # Save weights to temp file
            w_tmp = tempfile.NamedTemporaryFile(
                suffix="." + _ext(weights_file.filename), delete=False)
            weights_file.save(w_tmp.name)
            w_tmp.close()
            tmp_files.append(w_tmp.name)

            # Optional code file
            code_path = None
            code_file = request.files.get(code_key)
            if code_file and code_file.filename:
                if _ext(code_file.filename) not in ALLOWED_CODE:
                    return jsonify({
                        "error": f"{label}: model code file must be a .py script."
                    }), 400
                c_tmp = tempfile.NamedTemporaryFile(suffix=".py", delete=False)
                code_file.save(c_tmp.name)
                c_tmp.close()
                tmp_files.append(c_tmp.name)
                code_path = c_tmp.name

            try:
                model, source = load_model_from_path(w_tmp.name, code_path=code_path)
            except ValueError as exc:
                return jsonify({"error": f"{label}: {exc}"}), 400

            input_shape = tuple(int(v) for v in wc.get("input_shape", [1, 3, 224, 224]))

            w = ModelParser(model, input_shape=input_shape).analyze()
            w.name = wc.get("name") or secure_filename(weights_file.filename).rsplit(".", 1)[0]
            w.task_type = wc.get("task_type", "custom")
            workloads.append(w)

        results = Evaluator(workloads, architectures).run()

        return jsonify({
            "results": results,
            "workloads": [w.to_dict() for w in workloads],
            "architectures": [a.to_dict() for a in architectures],
        })

    except Exception as exc:
        return jsonify({"error": f"Internal error: {exc}"}), 500

    finally:
        for path in tmp_files:
            try:
                os.unlink(path)
            except OSError:
                pass


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5001)
