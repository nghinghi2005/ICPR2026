"""ICPR_2026 toolkit package.

Keep top-level imports light to avoid breaking CLI entrypoints when optional
visualization dependencies aren't installed.
"""

try:
	from .utility.visualizer import visualize  # optional convenience import
except Exception:  # pragma: no cover
	visualize = None
