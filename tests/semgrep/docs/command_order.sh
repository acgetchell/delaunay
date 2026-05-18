#!/usr/bin/env bash

# ok: delaunay.docs.check-before-fix-command-order
just check
just fix

# ruleid: delaunay.docs.check-before-fix-command-order
just fix
just check

# ok: delaunay.docs.check-before-fix-command-order
just python-check
just python-fix

# ruleid: delaunay.docs.check-before-fix-command-order
just python-fix
just python-check
