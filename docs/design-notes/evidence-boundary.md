# Design note: composing monitoring evidence

## Decision

The synthetic walkthrough composes the existing Python components directly:
`validate_schema`, `detect_numerical_drift`, `build_incident`, and `investigate`. It translates
the detector outputs into the same typed evidence models accepted by `POST /investigate`,
assigns stable IDs within the run, and prints both detector findings and the investigator's
evidence references. The demo uses seed `42` by default and fixes the sample size at 200 so a
reviewer can reproduce the same result with one command.

## Alternatives

- Calling the FastAPI routes through `TestClient` would exercise HTTP serialization as well,
  but would mix an integration walkthrough with endpoint transport tests already covered in
  `tests/test_app.py`.
- Starting a local server and sending HTTP requests would demonstrate deployment behavior,
  but would add process lifecycle and port management to a local demo without improving the
  evidence-provenance lesson.
- Duplicating the full pipeline inside a notebook would make the walkthrough less reusable
  and could let example logic drift away from production components.

Direct composition keeps the example fast, deterministic, and easy to run offline. Separate
API tests remain responsible for status codes, request validation, response models, and body
limits.

## Evidence and limitations

Only aggregate schema counts, drift statistics, incident metadata, and bounded investigator
evidence are reported; synthetic row values are not printed. Evidence IDs connect each
diagnosis to the supplied schema or feature result. The demo checks that all three injected
categories appear before returning success, and its integration test checks those references
against the supplied IDs.

The example's deliberately strong location shift demonstrates the data flow, not statistical
power. Its independent synthetic samples, two-feature correction family, and single seeded
run are not a benchmark. The investigator consumes caller-provided claims and does not
authenticate their origin. A significant distribution test does not show model degradation
or establish a causal explanation. The seed is configurable, so a different random sample
may cause the success assertion to fail if the injected signals are not all detected; that
failure is explicit instead of silently presenting an incomplete walkthrough.
