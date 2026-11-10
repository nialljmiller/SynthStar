"""Toy stand-in for hdps.alerts.producer.AlertProducer: the alert stream written to files instead of Kafka.

hdps's AlertVettingTask builds every packet (hdps.alerts.models.Alert, in its _maybe_kafka) and hands it to its
producer's send_async. This class is that producer with Kafka replaced by a file: each packet is Avro-encoded with
the public alert schema (hdps_alerts/schemas/*.avsc, read from the hdps checkout at run time, so a new schema merged
into hdps is used as it arrives) and written as an Avro object container file per exposure (the schema embedded).
ADR-0022: the external contract is Avro binary under argus.alert; on Kafka each message is one schemaless Avro
record, here the file's records are the same encoding. Nothing in a packet is changed or added.
"""
from pathlib import Path



def schema_dir():
    """hdps_alerts/schemas in the installed hdps checkout."""
    import hdps.alerts

    pkg = Path(hdps.alerts.__file__).resolve()
    for parent in pkg.parents:
        if (parent / "schemas" / "argus.alert.avsc").exists():
            return parent / "schemas"
    raise FileNotFoundError(f"hdps_alerts/schemas not found above {pkg}")


def schema_files(d=None):
    """hdps's argus.alert*.avsc files in named-type dependency order (each after the records it names; the envelope,
    argus.alert.avsc, last), as hdps_alerts/tests/test_models.py lists them. Read from the folder, so any schema
    version works: v3.0 has candidate, eventAssociation, ssoAssociation; v4.0 (hdps PR #474) drops eventAssociation."""
    import json

    def refs(t):  # the type names a type refers to (strings in type position)
        if isinstance(t, str):
            return {t}
        if isinstance(t, list):
            return set().union(*map(refs, t)) if t else set()
        if isinstance(t, dict):
            out = set()
            for key in ("type", "items", "values"):
                if key in t:
                    out |= refs(t[key])
            for field in t.get("fields", []):
                out |= refs(field["type"])
            return out
        return set()

    d = Path(d) if d is not None else schema_dir()
    named = {}
    for path in sorted(d.glob("argus.alert*.avsc")):
        s = json.loads(path.read_text())
        named[f"{s.get('namespace', '')}.{s['name']}".lstrip(".")] = (path, refs(s) - {s["name"], "record"})
    order, done = [], set()
    while len(order) < len(named):
        ready = [n for n, (_, uses) in named.items() if n not in done
                 and all(other in done or other == n or other not in uses for other in named)]
        if not ready:
            raise ValueError(f"{d}: circular named types among {sorted(set(named) - done)}")
        for n in sorted(ready):
            order.append(named[n][0])
            done.add(n)
    return order


def alert_schema():
    """The argus.alert envelope from hdps's .avsc files, its named records inlined (so the container file's
    embedded schema is complete and any Avro reader can decode it)."""
    from fastavro.schema import load_schema_ordered

    return load_schema_ordered([str(path) for path in schema_files()])


class AvroFileProducer:
    """AlertProducer's send_async, collecting packets; write(path) writes them as one Avro container file."""

    def __init__(self):
        self.schema = alert_schema()
        self.pending = []

    def send_async(self, alert, topic=None):
        self.pending.append(alert.model_dump())  # the record hdps's schema round-trip test encodes

    def write(self, path):
        """Write the collected packets to path (an Avro object container file); returns the count."""
        from fastavro import writer

        n = len(self.pending)
        if n:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            with open(path, "wb") as f:
                writer(f, self.schema, self.pending, codec="deflate")
        self.pending = []
        return n

    def write_by_epoch(self, folder):
        """Write the collected packets one Avro container file per exposure, <folder>/<epochId>.avro (the epoch
        hdps stamps on each candidate); returns the count."""
        from fastavro import writer

        by_epoch = {}
        for packet in self.pending:
            by_epoch.setdefault(str(packet["candidate"]["epochId"]), []).append(packet)
        for epoch, packets in by_epoch.items():
            Path(folder).mkdir(parents=True, exist_ok=True)
            with open(Path(folder) / f"{epoch}.avro", "wb") as f:
                writer(f, self.schema, packets, codec="deflate")
        n = len(self.pending)
        self.pending = []
        return n


def read(path):
    """The packets of an alert file, as dictionaries (fastavro reader against the embedded schema)."""
    from fastavro import reader

    with open(path, "rb") as f:
        return list(reader(f))
