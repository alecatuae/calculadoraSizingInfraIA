"""
Cálculos de VRAM (pesos fixos + KV variável + budget operacional).
"""

import math
from dataclasses import dataclass
from typing import List, Optional

from .models import ModelSpec
from .servers import ServerSpec


GIB_FACTOR = 2**30


@dataclass
class VRAMResult:
    """Resultado do cálculo de VRAM."""
    # Pesos do modelo
    weights_gib: float
    weights_estimated: bool
    fixed_model_gib: float  # Por nó
    
    # Budget e capacidade
    hbm_total_gib: float
    budget_for_sessions_gib: float
    sessions_budget_gib: float
    
    # Sessões
    vram_per_session_gib: float
    sessions_per_node: int
    
    # VRAM efetiva
    vram_total_node_at_limit_gib: float
    
    # Warnings
    warnings: List[str]


def calc_weights_memory(
    model: ModelSpec,
    weights_precision: str,
    weights_memory_override: Optional[float] = None
) -> tuple[float, bool, List[str]]:
    """
    Calcula memória dos pesos do modelo.
    
    Prioridade: Override CLI > models.json > Estimativa
    
    Returns:
        (weights_gib, was_estimated, warnings)
    """
    warnings = []
    
    if weights_memory_override is not None:
        return weights_memory_override, False, warnings
    
    # Tentar obter do models.json
    weights_gib = model.get_weights_memory(weights_precision)
    
    if weights_gib is not None:
        return weights_gib, False, warnings
    
    # Estimar se total_params_b disponível
    if model.total_params_b is not None:
        bytes_per_param = model.weights_bytes_per_param(weights_precision)
        weights_gib = model.total_params_b * 1e9 * bytes_per_param / GIB_FACTOR
        
        warnings.append(
            f"⚠️  AVISO: Memória de pesos para {weights_precision.upper()} foi ESTIMADA "
            f"({weights_gib:.2f} GiB) a partir de total_params_b={model.total_params_b}B. "
            f"Para sizing preciso, forneça valor exato em models.json ou via CLI."
        )
        return weights_gib, True, warnings
    
    # Fallback: não foi possível determinar
    warnings.append(
        f"🚨 ERRO CRÍTICO: Memória de pesos para {weights_precision.upper()} NÃO PÔDE SER "
        "DETERMINADA. Assumindo 0 GiB, o que provavelmente levará a sizing incorreto. "
        "Forneça weights_memory_gib em models.json, total_params_b, ou via CLI."
    )
    return 0.0, True, warnings


def calc_vram(
    model: ModelSpec,
    server: ServerSpec,
    kv_gib_per_session: float,
    concurrency: int,
    runtime_overhead_gib: float,
    kv_budget_ratio: float,
    weights_precision: str,
    weights_memory_override: Optional[float] = None,
    replicas_per_node: int = 1,
    tensor_parallel: Optional[int] = None,
    pipeline_parallel: int = 1
) -> VRAMResult:
    """
    Calcula consumo real de VRAM por nó e capacidade.
    
    Args:
        model: Especificação do modelo
        server: Especificação do servidor
        kv_gib_per_session: KV cache por sessão (GiB)
        concurrency: Sessões simultâneas alvo
        runtime_overhead_gib: Overhead do runtime (buffers, etc)
        kv_budget_ratio: Ratio de HBM disponível para KV (0.7 = 70%)
        weights_precision: Precisão dos pesos
        weights_memory_override: Override manual da memória de pesos
        replicas_per_node: Número de réplicas por nó
        tensor_parallel: Grau de paralelismo de tensor
        pipeline_parallel: Grau de paralelismo de pipeline
    
    Returns:
        VRAMResult com métricas de VRAM
    """
    warnings = []
    
    # 1. Calcular memória dos pesos
    weights_gib, weights_estimated, w_warnings = calc_weights_memory(
        model, weights_precision, weights_memory_override
    )
    warnings.extend(w_warnings)
    
    # 2. Calcular VRAM fixa por nó (pesos distribuídos)
    actual_tensor_parallel = tensor_parallel if tensor_parallel is not None else server.gpu.count
    if tensor_parallel is None:
        warnings.append(
            f"[INFO] --tensor-parallel nao especificado. Assumindo TP = GPUs do servidor ({server.gpu.count})."
        )
    
    gpus_per_replica = actual_tensor_parallel * pipeline_parallel
    if gpus_per_replica == 0:
        warnings.append(
            "🚨 ERRO: gpus_per_replica é zero (TP ou PP inválido). Assumindo 1 para cálculo."
        )
        gpus_per_replica = 1
    
    # VRAM dos pesos por réplica (sharded)
    vram_weights_per_replica = weights_gib / gpus_per_replica
    
    # VRAM fixa total por nó
    fixed_model_gib = vram_weights_per_replica * replicas_per_node * gpus_per_replica
    
    # 3. Calcular budget real para sessões
    hbm_total_gib = server.total_hbm_gib
    budget_for_sessions_gib = hbm_total_gib - fixed_model_gib - runtime_overhead_gib
    budget_for_sessions_gib = max(0.0, budget_for_sessions_gib)
    
    if budget_for_sessions_gib <= 0:
        warnings.append(
            f"🚨 ERRO CRÍTICO: Budget para sessões ≤ 0 GiB! "
            f"(HBM={hbm_total_gib:.1f} - Pesos={fixed_model_gib:.1f} - "
            f"Overhead={runtime_overhead_gib:.1f}). "
            "Pesos + overhead consomem toda a HBM. Use servidor maior ou reduza pesos/overhead."
        )
    
    # Aplicar ratio operacional
    sessions_budget_gib = budget_for_sessions_gib * kv_budget_ratio
    
    # 4. Calcular sessões por nó
    if kv_gib_per_session > 0:
        sessions_per_node = math.floor(sessions_budget_gib / kv_gib_per_session)
    else:
        sessions_per_node = 0
        warnings.append("🚨 ERRO: kv_gib_per_session é zero. Não é possível calcular sessões/nó.")
    
    if sessions_per_node == 0:
        warnings.append(
            f"🚨 ERRO CRÍTICO: Não cabe NEM 1 SESSÃO por nó! "
            f"(Budget={sessions_budget_gib:.1f} GiB, KV/sessão={kv_gib_per_session:.2f} GiB). "
            "Ações: (1) Reduza effective_context, (2) Use KV precision FP8, "
            "(3) Use servidor com mais HBM, (4) Reduza runtime_overhead."
        )
    
    # 5. VRAM total por nó no limite
    vram_total_node_at_limit_gib = (
        fixed_model_gib + runtime_overhead_gib + (sessions_per_node * kv_gib_per_session)
    )
    
    # 6. Alertas adicionais
    if kv_budget_ratio > 0.75:
        warnings.append(
            f"⚠️  AVISO: kv_budget_ratio={kv_budget_ratio*100:.0f}% é alto (>75%). "
            "Risco de fragmentação e instabilidade. Recomendado: 60-70%."
        )
    
    if runtime_overhead_gib < 50:
        warnings.append(
            f"⚠️  AVISO: runtime_overhead_gib={runtime_overhead_gib} GiB parece baixo (<50 GiB). "
            "Pode estar subestimado para modelos grandes. Considere 100-150 GiB."
        )
    
    utilization_at_limit = vram_total_node_at_limit_gib / hbm_total_gib if hbm_total_gib > 0 else 0
    if utilization_at_limit > 0.90:
        warnings.append(
            f"⚠️  ALERTA: Utilização de HBM no limite seria {utilization_at_limit*100:.1f}% (>90%). "
            "Sistema operaria muito próximo do limite, arriscado para produção."
        )
    
    return VRAMResult(
        weights_gib=weights_gib,
        weights_estimated=weights_estimated,
        fixed_model_gib=fixed_model_gib,
        hbm_total_gib=hbm_total_gib,
        budget_for_sessions_gib=budget_for_sessions_gib,
        sessions_budget_gib=sessions_budget_gib,
        vram_per_session_gib=kv_gib_per_session,
        sessions_per_node=sessions_per_node,
        vram_total_node_at_limit_gib=vram_total_node_at_limit_gib,
        warnings=warnings
    )
