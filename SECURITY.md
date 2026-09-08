# Security Policy

## Scope

This repository contains a benchmark: a simulation harness, synthetic task data, a
synthetic customer database, and knowledge-base documents. It ships no production
service and holds no real customer data. All names, accounts, addresses and
identifiers are fictional; any resemblance to real entities is coincidental.

## Reporting a vulnerability

If you find a security issue, for example in the bundled harness, its HTTP serving
utilities or a dependency, please report it privately by email to **npciai@npci.org.in**, or via GitHub's
"Report a vulnerability" (Security Advisories) feature on this repository, rather than
opening a public issue. Include the affected file, a reproduction, and the impact you
see. We aim to acknowledge reports within a week.

## Data concerns

If you believe any file in this repository contains non-synthetic personal data,
report it to the same address; it will be removed and the affected artefacts regenerated.

## Dependencies

The Python dependencies are pinned in `uv.lock`. Please report vulnerabilities in a
dependency to that project first; a version bump here is welcome once a fixed release
exists.
