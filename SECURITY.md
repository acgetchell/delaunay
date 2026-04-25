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

- Preferred: [GitHub private vulnerability report](https://github.com/acgetchell/delaunay/security/advisories/new)
- Alternative: Email the maintainer at [adam@adamgetchell.org](mailto:adam@adamgetchell.org)

Do **not** open a public issue for security vulnerabilities.

### Please include

- Description of the issue
- Steps to reproduce (code, inputs, or configuration)
- Affected versions
- Impact (e.g., panic, denial of service, incorrect results)
- Suggested fix or mitigation (optional)

---

## Disclosure Process

- Reports will be acknowledged as soon as maintainer availability allows
- The issue will be triaged and severity assessed on a best-effort basis
- For accepted reports, status updates will be provided when there is meaningful progress or a material change in assessment
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
- Incorrect geometric or topological results that affect security, integrity, or availability when processing untrusted input
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

Security vulnerabilities may be disclosed via the
[RustSec Advisory Database](https://github.com/RustSec/advisory-db).

This enables detection via `cargo audit`.

---

## Safe Harbor

Good-faith security research is welcome. Please avoid privacy violations,
data destruction, persistence, service disruption, and public disclosure before
a fix or mitigation is available. Reports that follow coordinated disclosure
and make a reasonable effort to avoid harm will be treated as helpful
contributions.

---

## Acknowledgements

Responsible disclosure is appreciated. Reporters may be credited in advisories or release notes unless anonymity is requested.
