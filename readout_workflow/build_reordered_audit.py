"""Keep the original study's auditor intact while auditing the new ladder."""
from pathlib import Path
BASE=Path(__file__).resolve().parent
source=(BASE/'audit_selected_robustness.py').read_text()
source=source.replace('from robustness_rescue import','from reordered_robustness import')
source=source.replace('from run_selected_robustness import','from run_reordered_robustness import')
source=source.replace('min(ec)<150-1e-7','min(ec)<120-1e-7')
source=source.replace("('carrier',", "('operation_frequency',")
source=source.replace("('hybridization',point", "('EC_floor',min(point['loaded_ec1_mhz'],point['loaded_ec2_mhz'])<150-1e-7),\n            ('hybridization',point")
(BASE/'audit_reordered_robustness.py').write_text(source,encoding='utf-8')
