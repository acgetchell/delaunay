# Security Policy

## Supported Versions

Security fixes are provided for the **latest published version** of `delaunay`.

| Version        | Supported |
|----------------|-----------|
| Latest release | ✅        |
| Older releases | ❌        |

Users are expected to upgrade to the most recent version to receive security updates. Backports to older versions are not guaranteed.

---

## Reporting a Vulnerability

If you discover a security vulnerability, please report it **privately**.

- Preferred: GitHub Security Advisory (Security tab → “Report a vulnerability”)
- Alternative: Contact the maintainer via GitHub

Do **not** open a public issue for security vulnerabilities.

### Please include

- Description of the issue
- Steps to reproduce (code, inputs, or configuration)
- Affected versions
- Impact (e.g., panic, denial of service, incorrect results)
- Suggested fix or mitigation (optional)

---

## Disclosure Process

- Reports will be acknowledged within a reasonable timeframe (typically a few days)
- The issue will be triaged and severity assessed
- If accepted:
  - A fix will be prepared and released
  - A GitHub Security Advisory will be published
  - If applicable, a RustSec advisory will be requested
- If declined:
  - The reporter will be notified with an explanation

Please follow **coordinated disclosure** and avoid public disclosure until a fix is available.

---

## Scope

This crate uses `#![forbid(unsafe_code)]`, reducing memory safety risks. However, the following are considered in scope:

- Panics or crashes triggered by malformed or adversarial inputs
- Denial-of-service (CPU or memory exhaustion)
- Incorrect geometric or topological results under adversarial conditions
- Serialization/deserialization issues (e.g., malformed JSON inputs)

Out of scope:

- Expected floating-point instability when using non-exact kernels (e.g., `FastKernel`)
- Performance limitations that are not exploitable as denial-of-service
- Issues arising from use outside documented APIs

---

## Patch Policy

- Fixes will be released in the **latest version**
- Releases will be published on crates.io with corresponding GitHub releases
- Security advisories will be published via GitHub and, where appropriate, RustSec

---

## RustSec

Security vulnerabilities may be disclosed via the RustSec Advisory Database:
https://github.com/RustSec/advisory-db

This enables detection via `cargo audit`.

---

## Acknowledgements

Responsible disclosure is appreciated. Reporters may be credited in advisories or release notes unless anonymity is requested.
