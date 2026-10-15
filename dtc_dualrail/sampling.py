"""Parameter sampling and correlation utilities."""

import numpy as np

from dtc_dualrail.constants import DEFAULT_PARAM_REL_SIGMA, DEFAULT_PARAM_SIGMA


def _parameter_sigma(param_name, center, param_sigma, param_rel_sigma):
    if param_sigma is not None and param_name in param_sigma:
        return float(param_sigma[param_name])
    return abs(center) * param_rel_sigma.get(param_name, 0.01)


def _parameter_floor(param_name, center, param_rel_min):
    if isinstance(param_rel_min, dict):
        rel_min = float(param_rel_min.get(param_name, 0.01))
    else:
        rel_min = float(param_rel_min)
    if rel_min < 0:
        raise ValueError(f"param_rel_min for {param_name} must be non-negative")
    return max(abs(center) * rel_min, np.finfo(float).tiny)


def _normalize_correlation_groups(param_correlation_groups, param_names):
    if param_correlation_groups is None:
        return []

    param_names = set(param_names)
    seen_params = set()
    groups = []
    for group_idx, group in enumerate(param_correlation_groups):
        if isinstance(group, dict):
            if "params" not in group:
                raise ValueError("correlation group dictionaries must include 'params'")
            group_params = tuple(group["params"])
            correlation = float(group.get("correlation", 1.0))
        else:
            group_params = tuple(group)
            correlation = 1.0

        if len(group_params) < 2:
            raise ValueError("correlation groups must contain two or more parameters")
        if len(set(group_params)) != len(group_params):
            raise ValueError("correlation groups must not repeat a parameter")
        if correlation < 0.0 or correlation > 1.0:
            raise ValueError("correlation must be between 0 and 1")

        unknown_params = set(group_params) - param_names
        if unknown_params:
            raise ValueError(
                f"correlation group contains unknown parameters: {sorted(unknown_params)}"
            )

        repeated_params = seen_params & set(group_params)
        if repeated_params:
            raise ValueError(
                "parameters cannot appear in multiple correlation groups: "
                f"{sorted(repeated_params)}"
            )

        seen_params.update(group_params)
        groups.append(
            {
                "name": str(group.get("name", f"group_{group_idx}"))
                if isinstance(group, dict)
                else f"group_{group_idx}",
                "params": group_params,
                "correlation": correlation,
            }
        )

    return groups


def _sample_parameters(
    params,
    rng,
    *,
    param_sigma,
    param_rel_sigma,
    param_correlation_groups=None,
    param_rel_min=0.01,
    frozen_params=(),
):
    frozen_params = set(frozen_params)
    unknown_frozen = frozen_params - set(params)
    if unknown_frozen:
        raise ValueError(f"frozen parameters are unknown: {sorted(unknown_frozen)}")

    sampled_params = {}
    resolved_param_sigma = {}
    normalized_groups = _normalize_correlation_groups(
        param_correlation_groups,
        params.keys(),
    )
    grouped_params = set()

    for group in normalized_groups:
        correlation = group["correlation"]
        shared_noise = rng.normal()
        independent_weight = np.sqrt(1.0 - correlation)
        shared_weight = np.sqrt(correlation)
        for param_name in group["params"]:
            if param_name in frozen_params:
                sampled_params[param_name] = params[param_name]
                resolved_param_sigma[param_name] = 0.0
                grouped_params.add(param_name)
                continue
            center = params[param_name]
            sigma = _parameter_sigma(param_name, center, param_sigma, param_rel_sigma)
            if sigma < 0:
                raise ValueError(f"sigma for {param_name} must be non-negative")
            noise = shared_weight * shared_noise + independent_weight * rng.normal()
            floor = _parameter_floor(param_name, center, param_rel_min)
            resolved_param_sigma[param_name] = sigma
            sampled_params[param_name] = max(center + sigma * noise, floor)
            grouped_params.add(param_name)

    for param_name, center in params.items():
        if param_name in grouped_params:
            continue
        if param_name in frozen_params:
            sampled_params[param_name] = center
            resolved_param_sigma[param_name] = 0.0
            continue
        sigma = _parameter_sigma(param_name, center, param_sigma, param_rel_sigma)
        if sigma < 0:
            raise ValueError(f"sigma for {param_name} must be non-negative")
        floor = _parameter_floor(param_name, center, param_rel_min)
        resolved_param_sigma[param_name] = sigma
        sampled_params[param_name] = max(rng.normal(center, sigma), floor)

    return sampled_params, resolved_param_sigma


def _correlation_groups_attr(param_correlation_groups, param_names):
    groups = _normalize_correlation_groups(param_correlation_groups, param_names)
    if not groups:
        return ""
    return "; ".join(
        f"{group['name']}[{','.join(group['params'])}]:{group['correlation']}"
        for group in groups
    )


__all__ = [
    "DEFAULT_PARAM_REL_SIGMA",
    "DEFAULT_PARAM_SIGMA",
    "_correlation_groups_attr",
    "_normalize_correlation_groups",
    "_parameter_floor",
    "_parameter_sigma",
    "_sample_parameters",
]
