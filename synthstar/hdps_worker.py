"""Raw camera frames through hdps's own worker (hdps.orchestrate.worker.GPUWorker), as `hdps batch process` and the
orchestrator run it, without Ray (the class itself, which Ray only wraps):

- hdps.orchestrate.orchestrator.group_images_by_ratchet_camera -> data_sources.build_manifests_from_groups ->
  one ManifestSource per camera, attached to the worker (the orchestrator's dispatch);
- GPUWorker.initialize(WorkerConfig): the observation's [[hdps.tasks]] chain, its coadd accumulator ([hdps.coadd]),
  its WCS cache (CacheManager), its product writer into out_dir;
- GPUWorker.run, then drain (every open ratchet finalized, coadds and products written) and shutdown.

Toys, each standing in for a piece hdps runs against that does not exist here:
- valkey: an in-process fakeredis server (hdps's own tests use fakeredis) behind the worker's cache, DLQ and metrics;
- real/bogus: with no model_bundle, AllReal (every candidate scored 1); with one (tools/train_rb.py), hdps's own
  RFBundleClassifier loads it;
- Kafka: synthstar.alert_files.AvroFileProducer, hdps's packets written as Avro files;
- the astrometric and photometric solve, when the chain has TrueCalibrationTask in its place: the simulation's WCS
  and zero point (synthstar.true_calibration).
"""
from pathlib import Path

import numpy as np

from synthstar import schema as S


class AllReal:
    """Toy stand-in for hdps.alerts.classifier.RFBundleClassifier: every candidate scores 1 (no model exists)."""

    model_version = "none"

    def __init__(self):
        from hdps.alerts.features import FEATURE_NAMES

        self.feature_names = list(FEATURE_NAMES)

    def cut_for(self, label_or_value):
        return float(label_or_value) if not isinstance(label_or_value, str) else 0.5

    def score_features(self, feats):
        return np.ones(len(feats["x"]), dtype=float)

    def score_features_with_confidence(self, feats):
        n = len(feats["x"])
        return np.ones(n, dtype=float), np.ones(n, dtype=float)


def task_configs(obs, stop_after=None, reference_dir=None, out_dir=None):
    """[[hdps.tasks]] as build_task_list takes it ("none" -> None), up to and including stop_after; SFFT given
    reference_dir (its CoaddReferenceProvider's directory); a Task's relative output_dir under out_dir."""
    configs = [dict(type=t["type"], config={k: (None if v == S.NONE else v) for k, v in t["config"].items()})
               for t in obs["hdps"]["tasks"]]
    if stop_after is not None:
        configs = configs[: [c["type"] for c in configs].index(stop_after) + 1]
    for c in configs:
        if c["type"] == "SFFTSubtractionTask" and reference_dir is not None:
            c["config"]["reference_dir"] = str(reference_dir)
        d = c["config"].get("output_dir")
        if out_dir is not None and d and not Path(d).is_absolute():
            c["config"]["output_dir"] = str(Path(out_dir) / d)
    return configs


def _fake_valkey():
    """Point hdps's valkey factories at one in-process fakeredis server (the valkey toy)."""
    import fakeredis

    import hdps.orchestrate.cache as cache
    import hdps.orchestrate.worker as worker

    server = fakeredis.FakeServer()
    cache.create_valkey_client = lambda *a, **k: fakeredis.FakeRedis(server=server)
    worker.create_valkey_pool = lambda *a, **k: fakeredis.FakeRedis(server=server).connection_pool


def _physical_gpus():
    """The machine's physical GPU count (nvidia-smi); 0 on a CPU-only host."""
    import subprocess

    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=10)
        return sum(1 for line in out.stdout.splitlines() if line.strip().startswith("GPU "))
    except Exception:
        return 0


def _actor_setup():
    """worker_process_setup_hook: runs once in each Ray worker actor process at startup. Point hdps's valkey at an
    in-process fakeredis (no external valkey server for an offline run) and register the simulation's
    TrueCalibrationTask so the actor can build it from the task chain. Referenced by import path, so it runs in the
    actor, not the driver."""
    try:
        _fake_valkey()
    except Exception:
        pass
    try:
        from synthstar import true_calibration

        true_calibration.register()
    except Exception:
        pass


def run(obs, raw_dir, out_dir, reference_dir=None, stop_after=None, worker_id="synthstar", timeout_s=None):
    """Every raw frame in raw_dir through hdps's real Orchestrator (the path `hdps batch process` runs: one Ray
    worker actor, the [[hdps.tasks]] chain through hdps's PipelinedExecutor, coadds and products into out_dir).
    Alert packets are written as per-exposure Avro files by AlertVettingTask itself (emit_avro), as production would
    stream them to Kafka. Returns dict(processed, failed, coadds, alerts). No hdps internals are touched: the alert
    sink and the no-bundle classifier are selected through task config."""
    import ray
    from hdps.orchestrate.config import CoaddConfig, OrchestratorConfig, OutputConfig
    from hdps.orchestrate.orchestrator import Orchestrator

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    co = obs["hdps"]["coadd"]
    tcfgs = task_configs(obs, stop_after, reference_dir, out_dir)
    for c in tcfgs:  # the alert sink, wired through config: per-exposure Avro files in place of Kafka
        if c["type"] == "AlertVettingTask":
            c["config"]["emit_avro"] = True
            c["config"]["avro_dir"] = str(out_dir / "alerts")
    cfg = OrchestratorConfig(
        cameras=[],  # discovered from the frames (plan_batch)
        task_configs=tcfgs,
        coadd=CoaddConfig(strategy=co["strategy"], clip_kappa=co["clip_kappa"],
                          clip_min_frames=co["clip_min_frames"], accumulation_dtype=co["accumulation_dtype"]),
        output=OutputConfig(output_dir=str(out_dir)))
    if not ray.is_initialized():  # one Ray runtime; a logical GPU so the GPU-requesting worker actor schedules on CPU
        import os

        n_gpu = _physical_gpus()
        repo = str(Path(__file__).resolve().parents[1])  # the repo dir, so the actor can import synthstar
        env_vars = {"PYTHONPATH": os.pathsep.join(p for p in (repo, os.environ.get("PYTHONPATH", "")) if p)}
        for k in ("HDPS_FORCE_CPU", "OMP_NUM_THREADS", "NUMBA_NUM_THREADS", "MESA_DIR"):
            if k in os.environ:  # the CPU-route shim and thread caps must reach the worker process too
                env_vars[k] = os.environ[k]
        ray.init(num_gpus=(n_gpu if n_gpu > 0 else 1), include_dashboard=False, log_to_driver=False,
                 configure_logging=False, ignore_reinit_error=True,
                 runtime_env={"env_vars": env_vars,
                              "worker_process_setup_hook": "synthstar.hdps_worker._actor_setup"})
    orch = Orchestrator(config=cfg, input_dir=Path(raw_dir), output_dir=out_dir, max_workers=1)
    result = orch.run()  # start() + run_job() + stop(): hdps batch process's own path
    alert_dir = out_dir / "alerts"
    alerts = sum(1 for _ in alert_dir.glob("*.avro")) if alert_dir.exists() else 0
    return dict(processed=result.successful, failed=result.failed, coadds=result.coadds_produced, alerts=alerts)


def run_direct(obs, raw_dir, out_dir, reference_dir=None, stop_after=None, worker_id="synthstar", timeout_s=None):
    """The earlier in-process path: hdps's GPUWorker driven directly (no Ray/Orchestrator), the alert sink and
    classifier injected in-process. Kept for regression comparison against run(); produces identical data products
    for a single camera. Returns dict(processed, failed, coadds, alerts)."""
    from hdps.orchestrate.data_sources import ManifestSource, build_manifests_from_groups
    from hdps.orchestrate.orchestrator import group_images_by_ratchet_camera
    from hdps.orchestrate.worker import GPUWorker, WorkerConfig

    from synthstar.alert_files import AvroFileProducer

    _fake_valkey()
    if any(t["type"] == "TrueCalibrationTask" for t in obs["hdps"]["tasks"]):
        from synthstar import true_calibration

        true_calibration.register()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    co = obs["hdps"]["coadd"]
    worker = GPUWorker(worker_id=worker_id)
    worker.initialize(WorkerConfig(
        camera_id="auto", task_configs=task_configs(obs, stop_after, reference_dir, out_dir), coadd_strategy=co["strategy"],
        coadd_clip_kappa=co["clip_kappa"], coadd_clip_min_frames=co["clip_min_frames"],
        coadd_accumulation_dtype=co["accumulation_dtype"], output_dir=str(out_dir), checkpoint_interval=0,
        gpu_warmup_enabled=False, executor_type="pipelined"))
    stream = AvroFileProducer()
    for task in worker._pipeline._tasks:
        if type(task).__name__ == "AlertVettingTask":
            if not task.model_bundle:  # no trained bundle: every candidate scores 1; else hdps's RFBundleClassifier
                task._classifier = AllReal()
            task._producer = stream  # hdps builds the packets; the file sink replaces Kafka
    groups, skipped = group_images_by_ratchet_camera(sorted(Path(raw_dir).glob("*.fits")))
    if skipped:
        raise RuntimeError(f"hdps could not group {len(skipped)} frames: {skipped[:3]}")
    by_camera = {}
    for manifest in build_manifests_from_groups(groups):
        by_camera.setdefault(manifest.camera_id, []).append(manifest)
    worker.attach_sources([ManifestSource(m, source_camera_id=c) for c, m in by_camera.items()])
    worker.run()
    worker._batch_thread.join(timeout=timeout_s)
    worker.drain()  # every still-open ratchet finalized; coadds and products written
    n_alerts = stream.write_by_epoch(out_dir / "alerts")
    result = dict(processed=worker._images_processed, failed=worker._images_failed_count,
                  coadds=worker._coadds_written, alerts=n_alerts)
    worker.shutdown()
    return result
