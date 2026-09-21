"""Codegen layer for live-inference node generation.

Owns the ``gen-node`` CLI and the Jinja template that turns a contract
YAML into a runnable ``rclpy.Node`` source file. Stays separate from
``runtime/`` so the generated node depends on the runtime kernel but
never imports anything from this package — codegen is a developer
tool, not a runtime concern.
"""
