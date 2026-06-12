"""D-AAR Version A — decomposed adaptive agent routing.

Production (one query)::

    from daar.infer import route_query
    result = route_query(training_id="...", eval_tag="gaia")

CLI: ``scripts/infer_daar.py``

Batch eval / research: ``daar.routing.route_daar``, ``scripts/eval_*.py``
"""

from daar.infer import PRODUCTION_FLOW, RouteQueryResult, route_query

__all__ = [
    "PRODUCTION_FLOW",
    "RouteQueryResult",
    "route_query",
]
