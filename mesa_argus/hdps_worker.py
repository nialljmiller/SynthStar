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
- Kafka: mesa_argus.alert_files.AvroFileProducer, hdps's packets written as Avro files;
- the astrometric and photometric solve, when the chain has TrueCalibrationTask in its place: the simulation's WCS
  and zero point (mesa_argus.true_calibration).
"""
from pathlib import Path

import numpy as np

from mesa_argus import schema as S


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


def run(obs, raw_dir, out_dir, reference_dir=None, stop_after=None, worker_id="mesa_argus", timeout_s=None):
    """Every raw frame in raw_dir through hdps's worker; products and coadds into out_dir (hdps's layout). Returns
    dict(processed, failed, coadds, alerts)."""
    from hdps.orchestrate.data_sources import ManifestSource, build_manifests_from_groups
    from hdps.orchestrate.orchestrator import group_images_by_ratchet_camera
    from hdps.orchestrate.worker import GPUWorker, WorkerConfig

    from mesa_argus.alert_files import AvroFileProducer

    _fake_valkey()
    if any(t["type"] == "TrueCalibrationTask" for t in obs["hdps"]["tasks"]):
        from mesa_argus import true_calibration

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
