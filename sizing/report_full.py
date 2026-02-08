"""
Geração de relatório completo (técnico detalhado).
"""

from typing import Dict, Any, List
from .calc_scenarios import ScenarioResult
from .models import ModelSpec
from .servers import ServerSpec
from .storage import StorageProfile


def format_full_report(
    model: ModelSpec,
    server: ServerSpec,
    storage: StorageProfile,
    scenarios: Dict[str, ScenarioResult],
    concurrency: int,
    effective_context: int,
    kv_precision: str,
    warnings: List[str]
) -> str:
    """
    Gera relatório completo em texto.
    
    Returns:
        String com relatório formatado
    """
    lines = []
    
    # Cabeçalho
    lines.append("=" * 100)
    lines.append("RELATÓRIO COMPLETO DE SIZING - INFERÊNCIA DE LLM")
    lines.append("=" * 100)
    lines.append("")
    
    # Seção 1: Entradas
    lines.append("┌" + "─" * 98 + "┐")
    lines.append("│" + " SEÇÃO 1: ENTRADAS".ljust(98) + "│")
    lines.append("└" + "─" * 98 + "┘")
    lines.append("")
    
    lines.append(f"Modelo: {model.name}")
    lines.append(f"  • Camadas: {model.num_layers}")
    lines.append(f"  • KV Heads: {model.num_key_value_heads}")
    lines.append(f"  • Head Dim: {model.head_dim}")
    lines.append(f"  • Max Context: {model.max_position_embeddings:,}")
    lines.append(f"  • Attention Pattern: {model.attention_pattern}")
    lines.append("")
    
    lines.append(f"Servidor: {server.name}")
    lines.append(f"  • GPUs: {server.gpus}")
    lines.append(f"  • HBM per GPU: {server.hbm_per_gpu_gb} GB")
    lines.append(f"  • HBM Total: {server.total_hbm_gib:.1f} GiB")
    if server.power_kw_max:
        lines.append(f"  • Potência máxima: {server.power_kw_max} kW")
    if server.rack_units_u:
        lines.append(f"  • Rack: {server.rack_units_u}U")
    lines.append("")
    
    lines.append(f"Concorrência Alvo: {concurrency:,} sessões")
    lines.append(f"Contexto Efetivo: {effective_context:,} tokens")
    lines.append(f"Precisão KV: {kv_precision}")
    lines.append("")
    
    # Seção 2: Consumo Real de VRAM
    rec = scenarios["recommended"]
    lines.append("┌" + "─" * 98 + "┐")
    lines.append("│" + " SEÇÃO 2: CONSUMO REAL DE VRAM".ljust(98) + "│")
    lines.append("└" + "─" * 98 + "┘")
    lines.append("")
    
    lines.append("CONSUMO UNITÁRIO:")
    lines.append(f"  • Pesos do modelo: {rec.vram.fixed_model_gib:.2f} GiB")
    lines.append(f"  • KV cache por sessão: {rec.vram.vram_per_session_gib:.2f} GiB")
    lines.append(f"  • Overhead runtime: {rec.vram.hbm_total_gib - rec.vram.fixed_model_gib - rec.vram.budget_for_sessions_gib:.1f} GiB")
    lines.append("")
    
    lines.append("BUDGET E CAPACIDADE POR NÓ:")
    lines.append(f"  • HBM total: {rec.vram.hbm_total_gib:.1f} GiB")
    lines.append(f"  • Budget para sessões: {rec.vram.sessions_budget_gib:.1f} GiB")
    lines.append(f"  • Sessões suportadas: {rec.vram.sessions_per_node}")
    lines.append("")
    
    # Seção 3: Resultados por Cenário
    lines.append("┌" + "─" * 98 + "┐")
    lines.append("│" + " SEÇÃO 3: RESULTADOS POR CENÁRIO".ljust(98) + "│")
    lines.append("└" + "─" * 98 + "┘")
    lines.append("")
    
    for key in ["minimum", "recommended", "ideal"]:
        s = scenarios[key]
        lines.append("=" * 100)
        lines.append(f"CENÁRIO: {s.config.name}")
        lines.append("=" * 100)
        lines.append(f"  • Nós DGX: {s.nodes_final}")
        lines.append(f"  • Sessões por nó (capacidade): {s.vram.sessions_per_node}")
        lines.append(f"  • Sessões por nó (operando): {s.sessions_per_node_effective}")
        lines.append(f"  • VRAM por nó (efetiva): {s.vram_total_node_effective_gib:.1f} GiB ({s.hbm_utilization_ratio_effective*100:.1f}% HBM)")
        lines.append(f"  • Energia total: {s.total_power_kw:.1f} kW")
        lines.append(f"  • Rack total: {s.total_rack_u}U")
        lines.append(f"  • HA: {s.config.ha_mode}")
        lines.append("")
    
    # Seção 4: Alertas
    if warnings:
        lines.append("┌" + "─" * 98 + "┐")
        lines.append("│" + " SEÇÃO 4: ALERTAS E AVISOS".ljust(98) + "│")
        lines.append("└" + "─" * 98 + "┘")
        lines.append("")
        for warning in warnings:
            lines.append(f"  {warning}")
        lines.append("")
    
    lines.append("=" * 100)
    lines.append("FIM DO RELATÓRIO")
    lines.append("=" * 100)
    
    return "\n".join(lines)


def format_json_report(
    model: ModelSpec,
    server: ServerSpec,
    storage: StorageProfile,
    scenarios: Dict[str, ScenarioResult],
    concurrency: int,
    effective_context: int,
    kv_precision: str,
    warnings: List[str]
) -> Dict[str, Any]:
    """
    Gera relatório completo em formato JSON.
    
    Returns:
        Dict serializável para JSON
    """
    def scenario_to_dict(s: ScenarioResult) -> Dict[str, Any]:
        return {
            "config": {
                "name": s.config.name,
                "peak_headroom_ratio": s.config.peak_headroom_ratio,
                "ha_mode": s.config.ha_mode,
                "kv_budget_ratio": s.config.kv_budget_ratio
            },
            "results": {
                "fixed_model_gib": round(s.vram.fixed_model_gib, 2),
                "vram_per_session_gib": round(s.vram.vram_per_session_gib, 4),
                "sessions_budget_gib": round(s.vram.sessions_budget_gib, 2),
                "sessions_per_node": s.vram.sessions_per_node,
                "sessions_per_node_effective": s.sessions_per_node_effective,
                "vram_total_node_effective_gib": round(s.vram_total_node_effective_gib, 2),
                "hbm_utilization_ratio_effective": round(s.hbm_utilization_ratio_effective, 4),
                "nodes_capacity": s.nodes_capacity,
                "nodes_with_headroom": s.nodes_with_headroom,
                "nodes_final": s.nodes_final,
                "total_power_kw": round(s.total_power_kw, 2),
                "total_rack_u": s.total_rack_u,
                "total_heat_btu_hr": round(s.total_heat_btu_hr, 0)
            }
        }
    
    return {
        "inputs": {
            "model": model.name,
            "server": server.name,
            "storage": storage.name,
            "concurrency": concurrency,
            "effective_context": effective_context,
            "kv_precision": kv_precision
        },
        "scenarios": {
            "minimum": scenario_to_dict(scenarios["minimum"]),
            "recommended": scenario_to_dict(scenarios["recommended"]),
            "ideal": scenario_to_dict(scenarios["ideal"])
        },
        "warnings": warnings
    }
