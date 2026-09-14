#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
۶ سناریوی آزمایشی به زبان فارسی برای سنجش سوییچ پویای رژیم‌های شناختی و اکشن‌های اکتیو اینفرنس
"""

import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'mcp_server'))
from server import state_filter, REGIMES, POLICIES

def run_persian_6_tests():
    scenarios = [
        {
            "num": 1,
            "title": "اکتشاف و درک مسئله (EXPLORATION)",
            "user_prompt": "سلام، نظرت درباره معماری سیستم‌های توزیع‌شده چیه و چطور می‌تونیم تحلیلش کنیم؟",
            "obs_type": "general_chat",
            "expected_regime": 0,
            "expected_policy": 0
        },
        {
            "num": 2,
            "title": "تولید مستقیم کد پروداکشن (CODE_GENERATION)",
            "user_prompt": "یک تابع صف بدون قفل (Lock-free Ring Buffer) در C11 با حافظه استاتیک بنویس.",
            "obs_type": "code_request",
            "expected_regime": 1,
            "expected_policy": 1
        },
        {
            "num": 3,
            "title": "کالبدشکافی و دیباگ باگ کرش (DEBUGGING)",
            "user_prompt": "ارور اومده: Segmentation fault (core dumped) در خط ۴۵ فایل vsa.c به خاطر اشاره‌گر NULL.",
            "obs_type": "error_log",
            "expected_regime": 3,
            "expected_policy": 2
        },
        {
            "num": 4,
            "title": "آزمون و اعتبارسنجی بنچمارک (VERIFICATION)",
            "user_prompt": "خروجی تست‌های gcc اومد: تمام ۴۵ تست پاس شدند و زمان تاخیر ۲.۴ میکروثانیه ثبت شد.",
            "obs_type": "test_output",
            "expected_regime": 4,
            "expected_policy": 3
        },
        {
            "num": 5,
            "title": "بهینه‌سازی معماری و جبر خطی (REFACTORING)",
            "user_prompt": "معادله کاهش بعد ماتریس کوواریانس و تقریب رتبه پایین SVD رو بهینه کن.",
            "obs_type": "math_query",
            "expected_regime": 2,
            "expected_policy": 1
        },
        {
            "num": 6,
            "title": "تصمیم‌گیری نهایی و قفل معماری (DECISION)",
            "user_prompt": "طرح تایید شد، تغییرات رو نهایی کن و روی برنچ اصلی گیت‌هاب مرج کن.",
            "obs_type": "confirmation",
            "expected_regime": 5,
            "expected_policy": 3
        }
    ]

    print("=" * 95)
    print("نتایج اجرای آزمون ۶ مرحله‌ای سوییچ شناختی اکتیو اینفرنس (Active Inference POMDP):")
    print("=" * 95)

    all_passed = True
    for sc in scenarios:
        # Step: Observe incoming prompt category
        state = state_filter.observe(sc["obs_type"])
        # Step: Prescribe active policy
        policy = state_filter.prescribe_policy()

        dom = state["regime_index"]
        conf = state["confidence"]
        entropy = state["shannon_entropy_nats"]
        act = policy["action_index"]
        regime_name = REGIMES[dom].split()[0]
        policy_name = POLICIES[act].split()[0]

        passed_regime = (dom == sc["expected_regime"])
        passed_policy = (act == sc["expected_policy"])
        status = "✅ PASS" if (passed_regime and passed_policy) else "❌ FAIL"
        if not (passed_regime and passed_policy):
            all_passed = False

        print(f"\n[تست {sc['num']}]: {sc['title']}")
        print(f"  ورودی فارسی: \"{sc['user_prompt']}\"")
        print(f"  دسته مشاهده: {sc['obs_type']}")
        print(f"  حالت فعال (Regime): {dom} -> {regime_name} (اطمینان: {conf}% | آنتروپی: {entropy} nats)")
        print(f"  خط‌مشی تجویزی (Policy): {act} -> {policy_name}")
        print(f"  دستور مستقیم هوش مصنوعی: {policy['directive']}")
        print(f"  وضعیت آزمون: {status}")

    print("\n" + "=" * 95)
    if all_passed:
        print(">>> نتیجه نهایی: تمام ۶ حالت و ۴ خط‌مشی با موفقیت ۱۰۰٪ سوییچ شدند و اینرسی برطرف شد! <<<")
    else:
        print(">>> هشدار: برخی انتقالات انجام نشد. <<<")
    print("=" * 95)

if __name__ == '__main__':
    run_persian_6_tests()
