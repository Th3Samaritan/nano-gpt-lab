"""tests package marker.

WHY this file exists: a third-party PyPI package named `tests` sometimes
lands in site-packages and shadows this directory during plain imports
(namespace packages lose to regular packages). Making tests/ a regular
package means the repo root - which every runner puts first on sys.path -
always wins.
"""
