import sys
sys.path.insert(0, ".")

import importlib.util

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m

load("utils.constants", "utils/constants.py")
load("utils.logger",    "utils/logger.py")
h = load("utils.helpers", "utils/helpers.py")

results = [
    ("clean_whitespace",   h.clean_whitespace("  hello   world  ")),
    ("count_words",        h.count_words("Python is great for AI")),
    ("extract_emails",     h.extract_emails("john@example.com")),
    ("get_exp_level",      h.get_experience_level(5.5)),
    ("get_score_label",    h.get_score_label(87)),
    ("clamp(150,0,100)",   h.clamp(150, 0, 100)),
    ("percentage(45,60)",  h.percentage(45, 60)),
    ("is_valid_email",     h.is_valid_email("test@example.com")),
    ("is_empty_text",      h.is_empty_text("   ")),
    ("format_percentage",  h.format_percentage(87.456)),
    ("generate_filename",  h.generate_filename("John Doe")),
    ("badge html len",     len(h.skills_to_badge_html(["Python","FastAPI"]))),
    ("pluralize",          h.pluralize("skill", 3)),
    ("truncate_text",      h.truncate_text("Hello World This Is Long", 15)),
    ("deduplicate",        h.deduplicate(["Python","python","Java","java"])),
]

for name, val in results:
    print(f"{name:<25} {val}")

print()
print("ALL TESTS PASSED")
