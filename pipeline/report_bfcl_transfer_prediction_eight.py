"""Resource-compatibility entry point for the unchanged stage0 full104 report."""
from pathlib import Path
import report_bfcl_transfer_prediction as original
import supervise_bfcl_source_evaluation_fresh_eight as eight

_original_validate = original.validate_manifest
_original_build = original.build_report
CAP_V3 = 'experiments/SSH_FUTURE_GPU_CAP_V3_20261014.json'


def validate_manifest(root, plan):
    resource = plan.get('resource_authorization')
    if resource is not None and resource['path'] == CAP_V3:
        return eight.validate_manifest(root, plan)
    return _original_validate(root, plan)


def build_report(root, fit_ref, heldout_ref, seal_ref):
    previous = original.validate_manifest
    original.validate_manifest = validate_manifest
    try:
        result = _original_build(root, fit_ref, heldout_ref, seal_ref)
    finally:
        original.validate_manifest = previous
    for path in [Path(__file__).resolve(), root/'pipeline/supervise_bfcl_source_evaluation_fresh_eight.py']:
        result['implementation_sha256'][str(path.relative_to(root))] = original.digest(path)
    result['evidence']['resource_validation_adapter'] = {
        'path': str(Path(__file__).resolve().relative_to(root)),
        'sha256': original.digest(Path(__file__).resolve()),
        'scope': 'Only manifest resource validation dispatch changes. Original label aggregation, sealed-parameter replay, all104/51 raw audits, G/V, fixed prediction errors, costs and markdown are retained.'}
    return result


def main():
    original.build_report = build_report
    original.main()


if __name__ == '__main__':
    main()
