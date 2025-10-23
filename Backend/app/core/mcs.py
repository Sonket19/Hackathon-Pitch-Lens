from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict

import numpy as np
from google.api_core.client_options import ClientOptions
from google.cloud.aiplatform.gapic import PredictionServiceClient

from app.models.risk import FinancialSignals, MCSConfig
from config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class MCSSimulationResult:
    metric: str
    iterations: int
    p10: float
    p50: float
    p90: float
    mean: float
    success_prob_vs_claim: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "metric": self.metric,
            "iterations": self.iterations,
            "p10": self.p10,
            "p50": self.p50,
            "p90": self.p90,
            "mean": self.mean,
            "success_prob_vs_claim": self.success_prob_vs_claim,
        }


def simulate_financials(financials: FinancialSignals, config: MCSConfig, seed: int = 12345) -> MCSSimulationResult:
    if settings.VERTEX_MCS_ENDPOINT_ID:
        try:
            return _invoke_vertex_simulation(financials, config)
        except Exception as exc:  # pragma: no cover - network/service failures
            logger.warning("Remote MCS prediction failed, falling back to local simulation: %s", exc)

    iterations = config.iterations
    horizon = config.horizon_months
    rng = np.random.default_rng(seed)

    base_revenue = max(financials.base_monthly_revenue or 0.0, 0.0)
    growth_mean = financials.growth_mean if financials.growth_mean is not None else 0.03
    growth_sd = financials.growth_sd if financials.growth_sd is not None else 0.02
    churn_mean = financials.churn_mean if financials.churn_mean is not None else 0.02
    churn_sd = financials.churn_sd if financials.churn_sd is not None else 0.01
    claimed = financials.claimed_month12_revenue if financials.claimed_month12_revenue is not None else base_revenue

    revenue = np.full(iterations, base_revenue, dtype=float)

    lognormal_mu = np.log1p(growth_mean) - 0.5 * (growth_sd ** 2)

    burn = abs(financials.burn or 0.0)

    for _ in range(horizon):
        growth_factors = rng.lognormal(lognormal_mu, growth_sd, iterations)
        churn = np.clip(rng.normal(churn_mean, churn_sd, iterations), 0.0, 0.6)

        if burn > 0:
            efficiency = np.clip(revenue / burn, 0.0, 2.5)
        else:
            efficiency = np.full(iterations, 1.5)
        efficiency_boost = efficiency * 0.0192

        revenue *= growth_factors + efficiency_boost
        revenue *= (1.0 - churn)
        noise = rng.normal(1.0, 0.02, iterations)
        revenue *= np.clip(noise, 0.9, 1.1)
        revenue = np.clip(revenue, 0.0, None)

    p10, p50, p90 = np.percentile(revenue, [10, 50, 90])
    mean = float(np.mean(revenue))
    success_prob = float(np.mean(revenue >= claimed))

    return MCSSimulationResult(
        metric=config.target,
        iterations=iterations,
        p10=float(p10),
        p50=float(p50),
        p90=float(p90),
        mean=mean,
        success_prob_vs_claim=success_prob,
    )


def _invoke_vertex_simulation(financials: FinancialSignals, config: MCSConfig) -> MCSSimulationResult:
    client = PredictionServiceClient(
        client_options=ClientOptions(api_endpoint=f"{settings.GCP_LOCATION}-aiplatform.googleapis.com")
    )
    endpoint_path = client.endpoint_path(settings.GCP_PROJECT_ID, settings.GCP_LOCATION, settings.VERTEX_MCS_ENDPOINT_ID)

    instance = {
        "financials": financials.model_dump(),
        "config": config.model_dump(),
    }

    prediction = client.predict(endpoint=endpoint_path, instances=[instance])
    if not prediction.predictions:
        raise RuntimeError("Vertex AI MCS endpoint returned no predictions")

    payload = prediction.predictions[0]
    return MCSSimulationResult(
        metric=str(payload.get("metric", config.target)),
        iterations=int(payload.get("iterations", config.iterations)),
        p10=float(payload.get("p10", 0.0)),
        p50=float(payload.get("p50", 0.0)),
        p90=float(payload.get("p90", 0.0)),
        mean=float(payload.get("mean", 0.0)),
        success_prob_vs_claim=float(payload.get("success_prob_vs_claim", 0.0)),
    )
