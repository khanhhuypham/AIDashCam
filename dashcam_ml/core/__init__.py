"""Core warning logic (tracking, FCW, LDW).

Only depends on numpy + the config/domain layers, never on torch / ultralytics /
OpenCV, so each module can be ported line by line to Swift [App iOS (sau pipeline)].
"""
