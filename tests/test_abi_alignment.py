#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ABI Alignment and Binary Compatibility Suite for Micro-ActInf.
Validates zero-copy ctypes memory layout against C11 struct definitions.
"""

import sys
import os
import ctypes
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'mcp_server')))

from libactinf import (
    MicroActInfStruct,
    ActInfStepRecord,
    ActInfLearningConfig,
    ActInfCostConfig,
    MicroActInfEngine
)


class TestABIAlignment(unittest.TestCase):
    def test_struct_sizes(self):
        self.assertEqual(ctypes.sizeof(ActInfLearningConfig), 16)
        self.assertEqual(ctypes.sizeof(ActInfCostConfig), 24)
        self.assertEqual(ctypes.sizeof(ActInfStepRecord), 16)
        self.assertEqual(ctypes.sizeof(MicroActInfStruct), 21188)

    def test_struct_field_offsets(self):
        # Verify critical boundary offsets match C11 compiler layout exactly
        self.assertEqual(MicroActInfStruct.s.offset, 16)
        self.assertEqual(MicroActInfStruct.s_prev.offset, 80)
        self.assertEqual(MicroActInfStruct.A.offset, 144)
        self.assertEqual(MicroActInfStruct.B.offset, 2192)
        self.assertEqual(MicroActInfStruct.C.offset, 10384)
        self.assertEqual(MicroActInfStruct.history.offset, 21024)
        self.assertEqual(MicroActInfStruct.history_idx.offset, 21152)
        self.assertEqual(MicroActInfStruct.target_state.offset, 21156)
        self.assertEqual(MicroActInfStruct.goal_progress.offset, 21160)
        self.assertEqual(MicroActInfStruct.goal_drift.offset, 21164)
        self.assertEqual(MicroActInfStruct.progress_index.offset, 21168)
        self.assertEqual(MicroActInfStruct.step_count.offset, 21184)

    def test_engine_initialization_no_crash(self):
        engine = MicroActInfEngine()
        state = engine.get_state()
        self.assertIn('dominant_regime', state)
        self.assertIn('shannon_entropy_nats', state)
        self.assertIn('goal_drift', state)


if __name__ == '__main__':
    unittest.main()
