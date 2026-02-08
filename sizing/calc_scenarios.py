"""
Cálculos de cenários (Mínimo, Recomendado, Ideal).
"""

import math
from dataclasses import dataclass
from typing import List

from .calc_vram import VRAMResult


@dataclass
class ScenarioConfig:
    """Configuração de um cenário."""
    name: str
    peak_headroom_ratio: float
    ha_mode: str  # "none", "n+1", "n+2"
    ha_extra_nodes: int
    kv_budget_ratio: float


@dataclass
class ScenarioResult:
    """Resultado completo de um cenário."""
    config: ScenarioConfig
    vram: VRAMResult
    
    # Nós
    nodes_capacity: int
    nodes_with_headroom: int
    nodes_final: int
    
    # Sessões efetivas
    sessions_per_node_effective: int
    vram_total_node_effective_gib: float
    hbm_utilization_ratio_effective: float
    
    # Físico (será preenchido por calc_physical)
    total_power_kw: float = 0.0
    total_rack_u: int = 0
    total_heat_btu_hr: float = 0.0


def create_scenario_configs(
    peak_headroom_ratio: float,
    kv_budget_ratio: float
) -> dict[str, ScenarioConfig]:
    """
    Cria configurações dos 3 cenários padrão.
    
    Args:
        peak_headroom_ratio: Ratio de headroom fornecido pelo usuário
        kv_budget_ratio: Ratio de budget fornecido pelo usuário
    
    Returns:
        Dict com configurações "minimum", "recommended", "ideal"
    """
    return {
        "minimum": ScenarioConfig(
            name="MÍNIMO",
            peak_headroom_ratio=0.0,
            ha_mode="none",
            ha_extra_nodes=0,
            kv_budget_ratio=kv_budget_ratio
        ),
        "recommended": ScenarioConfig(
            name="RECOMENDADO",
            peak_headroom_ratio=peak_headroom_ratio,
            ha_mode="n+1",
            ha_extra_nodes=1,
            kv_budget_ratio=kv_budget_ratio
        ),
        "ideal": ScenarioConfig(
            name="IDEAL",
            peak_headroom_ratio=max(peak_headroom_ratio, 0.30),
            ha_mode="n+2",
            ha_extra_nodes=2,
            kv_budget_ratio=min(kv_budget_ratio, 0.65)
        )
    }


def calc_scenario(
    config: ScenarioConfig,
    vram: VRAMResult,
    concurrency: int,
    runtime_overhead_gib: float
) -> ScenarioResult:
    """
    Calcula resultado de um cenário específico.
    
    Args:
        config: Configuração do cenário
        vram: Resultado do cálculo de VRAM
        concurrency: Sessões simultâneas alvo
        runtime_overhead_gib: Overhead do runtime
    
    Returns:
        ScenarioResult com métricas do cenário
    """
    # Calcular número de nós
    if vram.sessions_per_node > 0:
        nodes_capacity = math.ceil(concurrency / vram.sessions_per_node)
    else:
        nodes_capacity = 999999  # Indicador de erro
    
    nodes_with_headroom = math.ceil(nodes_capacity * (1 + config.peak_headroom_ratio))
    nodes_final = nodes_with_headroom + config.ha_extra_nodes
    
    # Calcular sessões efetivas por nó (operando)
    sessions_per_node_effective = math.ceil(concurrency / nodes_final) if nodes_final > 0 else 0
    
    # VRAM total efetiva por nó
    vram_total_node_effective_gib = (
        vram.fixed_model_gib +
        runtime_overhead_gib +
        (sessions_per_node_effective * vram.vram_per_session_gib)
    )
    
    # Utilização de HBM efetiva
    hbm_utilization_ratio_effective = (
        vram_total_node_effective_gib / vram.hbm_total_gib 
        if vram.hbm_total_gib > 0 else 0.0
    )
    
    return ScenarioResult(
        config=config,
        vram=vram,
        nodes_capacity=nodes_capacity,
        nodes_with_headroom=nodes_with_headroom,
        nodes_final=nodes_final,
        sessions_per_node_effective=sessions_per_node_effective,
        vram_total_node_effective_gib=vram_total_node_effective_gib,
        hbm_utilization_ratio_effective=hbm_utilization_ratio_effective
    )
