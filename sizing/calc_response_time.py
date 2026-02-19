"""
Cálculos de TTFT (Time to First Token) e TPOT/ITL (Time Per Output Token)
com validação de SLO (Service Level Objectives) para inferência LLM.

Fontes de dados:
  - models.json   → performance.prefill_tokens_per_sec_<gpu> / decode_tokens_per_sec_<gpu>
  - servers.json  → gpu.model (para selecionar throughput correto)
  - parameters.json → network_latency_*, avg_output_tokens, queuing_factor_*, latency_benchmarks
"""

import json
import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from .models import ModelSpec
from .servers import ServerSpec


# ---------------------------------------------------------------------------
# Helpers de carregamento de parâmetros
# ---------------------------------------------------------------------------

def load_parameter(param_name: str, default: Any) -> Any:
    """Carrega parâmetro de parameters.json com fallback para default."""
    try:
        with open('parameters.json', 'r', encoding='utf-8') as f:
            params = json.load(f)
            return params.get(param_name, default)
    except Exception:
        return default


def load_latency_benchmarks() -> Dict[str, Any]:
    """Carrega benchmarks de latência de parameters.json (retorna defaults se ausente)."""
    defaults = {
        'ttft_excellent_ms': 500,
        'ttft_good_ms': 1000,
        'ttft_acceptable_ms': 2000,
        'tpot_excellent_tokens_per_sec': 10,
        'tpot_good_tokens_per_sec': 8,
        'tpot_acceptable_tokens_per_sec': 6
    }
    try:
        with open('parameters.json', 'r', encoding='utf-8') as f:
            params = json.load(f)
            return params.get('latency_benchmarks', defaults)
    except Exception:
        return defaults


# ---------------------------------------------------------------------------
# Dataclass de resultado
# ---------------------------------------------------------------------------

@dataclass
class LatencyAnalysis:
    """Análise de latência TTFT/TPOT para um cenário."""

    # SLO
    target_ttft_p50_ms: Optional[int]
    target_ttft_p99_ms: Optional[int]
    target_tpot_tokens_per_sec: Optional[float]

    # Latências esperadas
    network_latency_p50_ms: float
    network_latency_p99_ms: float
    prefill_time_ms: float
    decode_time_ms: float
    queuing_delay_p50_ms: float
    queuing_delay_p99_ms: float
    ttft_p50_ms: float
    ttft_p99_ms: float
    tpot_tokens_per_sec: float
    itl_ms_per_token: float
    utilization: float

    # Validação SLO
    status: str            # 'OK' | 'SLO_MARGINAL' | 'SLO_VIOLATION' | 'NO_SLO'
    ttft_p50_ok: bool
    ttft_p99_ok: bool
    tpot_ok: bool
    ttft_p50_margin_percent: float
    ttft_p99_margin_percent: float
    tpot_margin_percent: float
    ttft_quality: str      # 'excellent' | 'good' | 'acceptable' | 'slow'
    tpot_quality: str

    # Diagnóstico
    bottleneck: str
    recommendation: str

    # Racional
    prefill_throughput: float
    decode_throughput: float
    avg_input_tokens: int
    avg_output_tokens: int
    source_prefill: str
    source_decode: str


# ---------------------------------------------------------------------------
# GPU model → chave de throughput no models.json
# ---------------------------------------------------------------------------

def _gpu_key(server: ServerSpec) -> str:
    """
    Deriva a chave de throughput (ex.: 'b300', 'b200', 'h100') a partir
    do modelo de GPU no servidor.
    """
    gpu_model = (server.gpu.model or "").lower()
    if "b300" in gpu_model:
        return "b300"
    if "b200" in gpu_model:
        return "b200"
    if "h200" in gpu_model:
        return "h200"
    if "h100" in gpu_model:
        return "h100"
    if "a100" in gpu_model:
        return "a100"
    return "h100"  # fallback conservador


# ---------------------------------------------------------------------------
# Throughput de tokens
# ---------------------------------------------------------------------------

def _gpu_fp4_tflops(server: ServerSpec) -> float:
    """Retorna TFLOPs FP4 estimados por nó para estimativa genérica."""
    gpu_key = _gpu_key(server)
    tflops_map = {
        "b300": 144.0,
        "b200": 90.0,
        "h200": 60.0,
        "h100": 40.0,
        "a100": 20.0,
    }
    return tflops_map.get(gpu_key, 40.0)


def estimate_throughput(model: ModelSpec, server: ServerSpec) -> Tuple[float, float]:
    """
    Estima throughput de prefill e decode se não especificado em models.json.

    Heurística simplificada baseada em FLOPs e tamanho do modelo.
    Retorna (prefill_tok/s, decode_tok/s) por nó.
    """
    model_size_b = (model.total_params_b or 70.0)
    gpu_tflops = _gpu_fp4_tflops(server) * server.gpu.count  # nó inteiro
    prefill_throughput = max(100.0, (gpu_tflops * 2.0) / model_size_b)
    decode_throughput = max(10.0, (gpu_tflops * 0.1) / model_size_b)
    return prefill_throughput, decode_throughput


def get_token_throughput(
    model: ModelSpec, server: ServerSpec
) -> Tuple[float, float]:
    """
    Retorna (prefill_tok/s, decode_tok/s) por nó.

    Prioridade:
      1. models.json → performance.prefill/decode_tokens_per_sec_<gpu>
      2. Estimativa genérica via FLOPs
    """
    perf = getattr(model, 'performance', None) or {}
    gpu_key = _gpu_key(server)

    prefill_key = f"prefill_tokens_per_sec_{gpu_key}"
    decode_key = f"decode_tokens_per_sec_{gpu_key}"

    prefill_thr = perf.get(prefill_key)
    decode_thr = perf.get(decode_key)

    source_prefill = f"models.json ({model.name}.performance.{prefill_key})"
    source_decode = f"models.json ({model.name}.performance.{decode_key})"

    if prefill_thr is None or decode_thr is None:
        est_prefill, est_decode = estimate_throughput(model, server)
        if prefill_thr is None:
            prefill_thr = est_prefill
            source_prefill = f"estimativa genérica (FLOPs/{model.total_params_b}B)"
        if decode_thr is None:
            decode_thr = est_decode
            source_decode = f"estimativa genérica (FLOPs/{model.total_params_b}B)"

    return float(prefill_thr), float(decode_thr), source_prefill, source_decode


def has_performance_data(model: ModelSpec, server: ServerSpec) -> bool:
    """Verifica se dados de performance existem para o par modelo/GPU."""
    perf = getattr(model, 'performance', None) or {}
    gpu_key = _gpu_key(server)
    return f"prefill_tokens_per_sec_{gpu_key}" in perf


# ---------------------------------------------------------------------------
# Classificação de qualidade
# ---------------------------------------------------------------------------

def classify_ttft(ttft_ms: float, benchmarks: Optional[Dict] = None) -> str:
    """Classifica TTFT segundo benchmarks da indústria (lidos de parameters.json)."""
    if benchmarks is None:
        benchmarks = load_latency_benchmarks()
    ttft_excellent = benchmarks.get('ttft_excellent_ms', 500)
    ttft_good = benchmarks.get('ttft_good_ms', 1000)
    ttft_acceptable = benchmarks.get('ttft_acceptable_ms', 2000)

    if ttft_ms < ttft_excellent:
        return 'excellent'
    elif ttft_ms < ttft_good:
        return 'good'
    elif ttft_ms <= ttft_acceptable:
        return 'acceptable'
    else:
        return 'slow'


def classify_tpot(tpot_tokens_per_sec: float, benchmarks: Optional[Dict] = None) -> str:
    """Classifica TPOT segundo benchmarks da indústria (lidos de parameters.json)."""
    if benchmarks is None:
        benchmarks = load_latency_benchmarks()
    tpot_excellent = benchmarks.get('tpot_excellent_tokens_per_sec', 10)
    tpot_good = benchmarks.get('tpot_good_tokens_per_sec', 8)
    tpot_acceptable = benchmarks.get('tpot_acceptable_tokens_per_sec', 6)

    if tpot_tokens_per_sec > tpot_excellent:
        return 'excellent'
    elif tpot_tokens_per_sec >= tpot_good:
        return 'good'
    elif tpot_tokens_per_sec >= tpot_acceptable:
        return 'acceptable'
    else:
        return 'slow'


# ---------------------------------------------------------------------------
# Identificação de gargalo
# ---------------------------------------------------------------------------

def identify_bottleneck(
    queuing_ms: float,
    prefill_ms: float,
    tpot_tokens_per_sec: float
) -> str:
    """Identifica o principal gargalo de latência."""
    benchmarks = load_latency_benchmarks()
    tpot_acceptable = benchmarks.get('tpot_acceptable_tokens_per_sec', 6)

    if queuing_ms == float('inf') or queuing_ms >= 99990:
        return 'QUEUING_DELAY - Sistema saturado (utilização >= threshold). Adicionar nós imediatamente.'

    if queuing_ms > (prefill_ms * 2):
        return 'QUEUING_DELAY - Alta utilização causando fila. Adicionar nós ou reduzir concorrência.'

    if prefill_ms > 1000:
        return 'PREFILL_COMPUTE - Processamento de prompt muito lento. GPU/modelo não otimizado para prefill.'

    if tpot_tokens_per_sec < tpot_acceptable:
        return 'DECODE_THROUGHPUT - Geração de tokens lenta. Considerar modelo menor, GPU mais rápida ou FP8/Flash Attention.'

    if prefill_ms > 500:
        return 'PREFILL_MODERATE - Prefill poderia ser mais rápido. Considerar GPU com maior throughput ou KV cache eficiente.'

    return 'BALANCED - Nenhum gargalo evidente. Sistema bem dimensionado.'


# ---------------------------------------------------------------------------
# Recomendação
# ---------------------------------------------------------------------------

def generate_recommendation(
    status: str,
    bottleneck: str,
    utilization: float,
    num_nodes: int,
    sessions_per_node: int,
    tpot_tokens_per_sec: float,
    target_tpot: Optional[float],
    ttft_p50_ms: Optional[float],
    target_ttft_p50: Optional[int]
) -> str:
    """Gera diagnóstico de gargalo baseado no status e bottleneck."""
    if status in ('OK', 'SLO_MARGINAL', 'NO_SLO'):
        return ''

    lines = []

    if 'QUEUING_DELAY' in bottleneck:
        lines.append(
            f"SLO inviável com a configuração atual: sistema saturado ({utilization*100:.1f}% utilização). "
            f"Impossível atender TTFT/TPOT com o modelo e servidor selecionados sob as premissas atuais. "
            f"Reduza contexto, reduza concorrência, use mais servidores de inferência ou ajuste quantização."
        )
    elif 'PREFILL_COMPUTE' in bottleneck or 'PREFILL_MODERATE' in bottleneck:
        min_feasible = (ttft_p50_ms or 0) * 1.05
        lines.append(
            f"Tempo de prefill ({ttft_p50_ms:.0f}ms) domina a latência. "
            f"TTFT mínimo viável estimado: {min_feasible:.0f}ms. "
            f"Impossível atender TTFT/TPOT com o modelo e servidor selecionados sob as premissas atuais. "
            f"Reduza contexto, use servidor com maior throughput de prefill ou ajuste quantização."
        )
    elif 'DECODE_THROUGHPUT' in bottleneck:
        lines.append(
            f"Throughput de decode insuficiente ({tpot_tokens_per_sec:.2f} tok/s). "
            f"Impossível atender TTFT/TPOT com o modelo e servidor selecionados sob as premissas atuais. "
            f"Reduza concorrência por nó, use servidor com maior throughput de decode ou ajuste quantização."
        )
    else:
        lines.append(
            "Impossível atender TTFT/TPOT com o modelo e servidor selecionados sob as premissas atuais. "
            "Verifique configuração de paralelismo e otimizações do serving framework."
        )

    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Cálculo principal
# ---------------------------------------------------------------------------

def calc_latency_analysis(
    model: ModelSpec,
    server: ServerSpec,
    num_nodes: int,
    sessions_per_node: int,
    concurrency: int,
    target_ttft_p50_ms: Optional[int],
    target_ttft_p99_ms: Optional[int],
    target_tpot_min_tokens_per_sec: Optional[float],
    effective_context: int
) -> 'LatencyAnalysis':
    """
    Calcula TTFT e TPOT esperados e valida contra SLOs.

    Quando targets são None (Modo A - Concorrência-Driven), retorna estimativas
    sem validação de SLO (status = 'NO_SLO').

    Parâmetros de cálculo lidos de parameters.json:
      - network_latency_p50_ms, network_latency_p99_ms
      - avg_output_tokens
      - max_utilization_threshold, ttft_p99_multiplier
      - queuing_factor_p50, queuing_factor_p99
      - latency_benchmarks (para classificação)
    """

    # -- Carregar parâmetros de parameters.json ----------------------------
    network_p50 = float(load_parameter('network_latency_p50_ms', 10))
    network_p99 = float(load_parameter('network_latency_p99_ms', 50))
    avg_output_tokens = int(load_parameter('avg_output_tokens', 100))
    max_util_threshold = float(load_parameter('max_utilization_threshold', 0.95))
    ttft_p99_multiplier = float(load_parameter('ttft_p99_multiplier', 2.0))
    qf_p50 = float(load_parameter('queuing_factor_p50', 0.3))
    qf_p99 = float(load_parameter('queuing_factor_p99', 0.8))

    # Default P99 se não especificado pelo usuário
    if target_ttft_p50_ms is not None and target_ttft_p99_ms is None:
        target_ttft_p99_ms = int(target_ttft_p50_ms * ttft_p99_multiplier)

    # -- Throughput de tokens ----------------------------------------------
    prefill_thr, decode_thr, src_prefill, src_decode = get_token_throughput(model, server)

    # -- Tokens de entrada: effective_context / 2 --------------------------
    avg_input_tokens = max(1, effective_context // 2)

    # -- Tempos de compute -------------------------------------------------
    prefill_time_ms = (avg_input_tokens / prefill_thr) * 1000.0
    decode_time_ms = (avg_output_tokens / decode_thr) * 1000.0

    # -- TPOT por sessão ---------------------------------------------------
    total_sessions = num_nodes * sessions_per_node
    tpot_tokens_per_sec = decode_thr / sessions_per_node if sessions_per_node > 0 else 0.0
    itl_ms = 1000.0 / tpot_tokens_per_sec if tpot_tokens_per_sec > 0 else 99999.0

    # -- Utilização e queuing delay ----------------------------------------
    utilization = min(concurrency / total_sessions, 1.0) if total_sessions > 0 else 1.0

    if utilization >= max_util_threshold:
        queuing_p50 = 99999.0
        queuing_p99 = 99999.0
    else:
        service_time = prefill_time_ms + decode_time_ms
        queuing_factor = utilization / (1.0 - utilization)
        queuing_p50 = queuing_factor * service_time * qf_p50
        queuing_p99 = queuing_factor * service_time * qf_p99

    # -- TTFT --------------------------------------------------------------
    ttft_p50 = network_p50 + queuing_p50 + prefill_time_ms
    ttft_p99 = network_p99 + queuing_p99 + (prefill_time_ms * 1.2)

    # -- Validação SLO (apenas Modo B) -------------------------------------
    ttft_p50_ok = True
    ttft_p99_ok = True
    tpot_ok = True
    ttft_p50_margin = 0.0
    ttft_p99_margin = 0.0
    tpot_margin = 0.0

    no_slo = target_ttft_p50_ms is None and target_tpot_min_tokens_per_sec is None

    if target_ttft_p50_ms is not None:
        ttft_p50_ok = ttft_p50 <= target_ttft_p50_ms
        ttft_p50_margin = ((target_ttft_p50_ms - ttft_p50) / target_ttft_p50_ms) * 100

    if target_ttft_p99_ms is not None:
        ttft_p99_ok = ttft_p99 <= target_ttft_p99_ms
        ttft_p99_margin = ((target_ttft_p99_ms - ttft_p99) / target_ttft_p99_ms) * 100

    if target_tpot_min_tokens_per_sec is not None:
        tpot_ok = tpot_tokens_per_sec >= target_tpot_min_tokens_per_sec
        tpot_margin = ((tpot_tokens_per_sec - target_tpot_min_tokens_per_sec) /
                       target_tpot_min_tokens_per_sec) * 100

    if no_slo:
        status = 'NO_SLO'
    else:
        all_ok = ttft_p50_ok and ttft_p99_ok and tpot_ok
        min_margin = min(ttft_p50_margin, ttft_p99_margin, tpot_margin)
        if all_ok and min_margin > 10:
            status = 'OK'
        elif all_ok:
            status = 'SLO_MARGINAL'
        else:
            status = 'SLO_VIOLATION'

    # -- Qualidade ---------------------------------------------------------
    benchmarks = load_latency_benchmarks()
    ttft_quality = classify_ttft(ttft_p50, benchmarks)
    tpot_quality = classify_tpot(tpot_tokens_per_sec, benchmarks)

    # -- Gargalo e recomendação -------------------------------------------
    bottleneck = identify_bottleneck(queuing_p50, prefill_time_ms, tpot_tokens_per_sec)
    recommendation = generate_recommendation(
        status, bottleneck, utilization,
        num_nodes, sessions_per_node,
        tpot_tokens_per_sec, target_tpot_min_tokens_per_sec,
        ttft_p50, target_ttft_p50_ms
    )

    return LatencyAnalysis(
        target_ttft_p50_ms=target_ttft_p50_ms,
        target_ttft_p99_ms=target_ttft_p99_ms,
        target_tpot_tokens_per_sec=target_tpot_min_tokens_per_sec,
        network_latency_p50_ms=network_p50,
        network_latency_p99_ms=network_p99,
        prefill_time_ms=prefill_time_ms,
        decode_time_ms=decode_time_ms,
        queuing_delay_p50_ms=queuing_p50,
        queuing_delay_p99_ms=queuing_p99,
        ttft_p50_ms=ttft_p50,
        ttft_p99_ms=ttft_p99,
        tpot_tokens_per_sec=tpot_tokens_per_sec,
        itl_ms_per_token=itl_ms,
        utilization=utilization,
        status=status,
        ttft_p50_ok=ttft_p50_ok,
        ttft_p99_ok=ttft_p99_ok,
        tpot_ok=tpot_ok,
        ttft_p50_margin_percent=ttft_p50_margin,
        ttft_p99_margin_percent=ttft_p99_margin,
        tpot_margin_percent=tpot_margin,
        ttft_quality=ttft_quality,
        tpot_quality=tpot_quality,
        bottleneck=bottleneck,
        recommendation=recommendation,
        prefill_throughput=prefill_thr,
        decode_throughput=decode_thr,
        avg_input_tokens=avg_input_tokens,
        avg_output_tokens=avg_output_tokens,
        source_prefill=src_prefill,
        source_decode=src_decode
    )


def calc_max_concurrency_from_slo(
    model: ModelSpec,
    server: ServerSpec,
    num_nodes: int,
    sessions_per_node: int,
    target_ttft_p50_ms: Optional[int],
    target_tpot_min_tokens_per_sec: Optional[float],
    effective_context: int
) -> 'SLOCapacityResult':
    """
    Sizing reverso: calcula a concorrência máxima atendível dado os SLOs.

    Inversão da fórmula M/M/c para TTFT:
      queuing_budget = TTFT_SLO - rede_p50 - prefill_time
      util_max = queuing_budget / (service_time * qf_p50 + queuing_budget)
      max_conc_ttft = floor(util_max * num_nodes * sessions_per_node)

    Inversão de TPOT:
      sess_max = floor(decode_thr / tpot_min)
      max_conc_tpot = sess_max * num_nodes
    """
    from .calc_scenarios import SLOCapacityResult

    network_p50 = float(load_parameter('network_latency_p50_ms', 10))
    qf_p50 = float(load_parameter('queuing_factor_p50', 0.3))
    max_util = float(load_parameter('max_utilization_threshold', 0.95))

    prefill_thr, decode_thr, _, _ = get_token_throughput(model, server)

    avg_input_tokens = max(1, effective_context // 2)
    avg_output_tokens = int(load_parameter('avg_output_tokens', 100))

    prefill_time_ms = (avg_input_tokens / prefill_thr) * 1000.0
    decode_time_ms = (avg_output_tokens / decode_thr) * 1000.0
    service_time = prefill_time_ms + decode_time_ms

    # === LIMITE POR TTFT ===
    queuing_budget_ms = 0.0
    util_max_from_ttft = max_util
    max_concurrency_from_ttft = int(max_util * num_nodes * sessions_per_node)
    is_feasible = True
    infeasibility_reason = ""

    if target_ttft_p50_ms is not None:
        queuing_budget_ms = float(target_ttft_p50_ms) - network_p50 - prefill_time_ms
        if queuing_budget_ms <= 0:
            is_feasible = False
            min_feasible = network_p50 + prefill_time_ms * 1.05
            infeasibility_reason = (
                f"Prefill ({prefill_time_ms:.0f}ms) + rede ({network_p50:.0f}ms) = {prefill_time_ms+network_p50:.0f}ms "
                f"ja excede o SLO de TTFT ({target_ttft_p50_ms}ms). "
                f"TTFT minimo viavel: {min_feasible:.0f}ms. "
                f"Reducao de contexto ou GPU com maior throughput de prefill sao necessarios."
            )
            util_max_from_ttft = 0.0
            max_concurrency_from_ttft = 0
        else:
            denom = service_time * qf_p50 + queuing_budget_ms
            util_max_from_ttft = min(queuing_budget_ms / denom, max_util) if denom > 0 else 0.0
            max_concurrency_from_ttft = int(util_max_from_ttft * num_nodes * sessions_per_node)

    # === LIMITE POR TPOT ===
    sessions_per_node_max_from_tpot = sessions_per_node
    max_concurrency_from_tpot = num_nodes * sessions_per_node

    if target_tpot_min_tokens_per_sec is not None and target_tpot_min_tokens_per_sec > 0:
        sessions_per_node_max_from_tpot = max(1, int(decode_thr / target_tpot_min_tokens_per_sec))
        max_concurrency_from_tpot = sessions_per_node_max_from_tpot * num_nodes

    # === GARGALO ===
    max_concurrency_combined = min(max_concurrency_from_ttft, max_concurrency_from_tpot)

    if target_ttft_p50_ms is None and target_tpot_min_tokens_per_sec is None:
        limiting_factor = "NO_SLO"
    elif target_ttft_p50_ms is None:
        limiting_factor = "TPOT"
    elif target_tpot_min_tokens_per_sec is None:
        limiting_factor = "TTFT"
    elif max_concurrency_from_ttft < max_concurrency_from_tpot:
        limiting_factor = "TTFT"
    elif max_concurrency_from_tpot < max_concurrency_from_ttft:
        limiting_factor = "TPOT"
    else:
        limiting_factor = "BALANCED"

    return SLOCapacityResult(
        max_concurrency_from_ttft=max_concurrency_from_ttft,
        max_concurrency_from_tpot=max_concurrency_from_tpot,
        max_concurrency_combined=max_concurrency_combined,
        limiting_factor=limiting_factor,
        util_max_from_ttft=util_max_from_ttft,
        sessions_per_node_max_from_tpot=sessions_per_node_max_from_tpot,
        prefill_time_ms=prefill_time_ms,
        queuing_budget_ms=queuing_budget_ms,
        is_feasible=is_feasible,
        infeasibility_reason=infeasibility_reason
    )


def latency_analysis_to_dict(la: Optional['LatencyAnalysis']) -> Optional[Dict[str, Any]]:
    """Converte LatencyAnalysis para dict serializável em JSON."""
    if la is None:
        return None
    return {
        "slo_defined": {
            "target_ttft_p50_ms": la.target_ttft_p50_ms,
            "target_ttft_p99_ms": la.target_ttft_p99_ms,
            "target_tpot_min_tokens_per_sec": la.target_tpot_tokens_per_sec,
            "avg_input_tokens": la.avg_input_tokens,
            "avg_output_tokens": la.avg_output_tokens
        },
        "expected": {
            "network_latency_p50_ms": round(la.network_latency_p50_ms, 1),
            "network_latency_p99_ms": round(la.network_latency_p99_ms, 1),
            "prefill_time_ms": round(la.prefill_time_ms, 1),
            "decode_time_ms": round(la.decode_time_ms, 1),
            "queuing_delay_p50_ms": round(la.queuing_delay_p50_ms, 1) if la.queuing_delay_p50_ms < 99000 else "saturated",
            "queuing_delay_p99_ms": round(la.queuing_delay_p99_ms, 1) if la.queuing_delay_p99_ms < 99000 else "saturated",
            "ttft_p50_ms": round(la.ttft_p50_ms, 1),
            "ttft_p99_ms": round(la.ttft_p99_ms, 1),
            "tpot_tokens_per_sec": round(la.tpot_tokens_per_sec, 2),
            "itl_ms_per_token": round(la.itl_ms_per_token, 1) if la.itl_ms_per_token < 99000 else "n/a",
            "utilization_percent": round(la.utilization * 100, 1)
        },
        "validation": {
            "status": la.status,
            "ttft_p50_ok": la.ttft_p50_ok,
            "ttft_p99_ok": la.ttft_p99_ok,
            "tpot_ok": la.tpot_ok,
            "ttft_p50_margin_percent": round(la.ttft_p50_margin_percent, 1),
            "ttft_p99_margin_percent": round(la.ttft_p99_margin_percent, 1),
            "tpot_margin_percent": round(la.tpot_margin_percent, 1),
            "ttft_quality": la.ttft_quality,
            "tpot_quality": la.tpot_quality,
            "bottleneck": la.bottleneck,
            "recommendation": la.recommendation
        },
        "rationale": {
            "prefill_throughput_tokens_per_sec": round(la.prefill_throughput, 1),
            "decode_throughput_tokens_per_sec": round(la.decode_throughput, 1),
            "source_prefill": la.source_prefill,
            "source_decode": la.source_decode,
            "queuing_model": "M/M/c Little's Law (queuing_factor_* from parameters.json)"
        }
    }
