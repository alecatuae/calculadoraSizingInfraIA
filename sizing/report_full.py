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
    lines.append(f"  • GPUs: {server.gpu.count}")
    lines.append(f"  • HBM per GPU: {server.gpu.hbm_per_gpu_gb} GB")
    lines.append(f"  • HBM Total: {server.total_hbm_gib:.1f} GiB")
    if server.power and server.power.power_kw_max:
        lines.append(f"  • Potência máxima: {server.power.power_kw_max} kW")
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
    
    # Seção 2.5: Perfil de Storage
    lines.append("┌" + "─" * 98 + "┐")
    lines.append("│" + " SEÇÃO 2.5: PERFIL DE STORAGE".ljust(98) + "│")
    lines.append("└" + "─" * 98 + "┘")
    lines.append("")
    
    lines.append(f"Storage: {storage.name}")
    lines.append(f"  • Tipo: {storage.type}")
    lines.append(f"  • Capacidade total: {storage.capacity_total_tb:.2f} TB")
    lines.append(f"  • Capacidade utilizável: {storage.usable_capacity_tb:.2f} TB")
    lines.append(f"  • IOPS leitura (max): {storage.iops_read_max:,}")
    lines.append(f"  • IOPS escrita (max): {storage.iops_write_max:,}")
    lines.append(f"  • Throughput leitura: {storage.throughput_read_mbps:.1f} MB/s ({storage.throughput_read_mbps/125:.2f} GB/s)")
    lines.append(f"  • Throughput escrita: {storage.throughput_write_mbps:.1f} MB/s ({storage.throughput_write_mbps/125:.2f} GB/s)")
    lines.append(f"  • Latência leitura (p50/p99): {storage.latency_read_ms_p50:.2f} / {storage.latency_read_ms_p99:.2f} ms")
    lines.append(f"  • Latência escrita (p50/p99): {storage.latency_write_ms_p50:.2f} / {storage.latency_write_ms_p99:.2f} ms")
    if storage.rack_units_u > 0 or storage.power_kw > 0:
        lines.append(f"  • Consumo físico: {storage.rack_units_u}U rack, {storage.power_kw:.1f} kW")
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
        lines.append("")
        lines.append("COMPUTAÇÃO:")
        lines.append(f"  • Nós DGX: {s.nodes_final}")
        lines.append(f"  • Sessões por nó (capacidade): {s.vram.sessions_per_node}")
        lines.append(f"  • Sessões por nó (operando): {s.sessions_per_node_effective}")
        lines.append(f"  • VRAM por nó (efetiva): {s.vram_total_node_effective_gib:.1f} GiB ({s.hbm_utilization_ratio_effective*100:.1f}% HBM)")
        lines.append("")
        
        if s.storage:
            lines.append("STORAGE:")
            lines.append(f"  • Volumetria total: {s.storage.storage_total_tb:.2f} TB")
            lines.append(f"    - Modelo: {s.storage.storage_model_tb:.2f} TB")
            lines.append(f"    - Cache: {s.storage.storage_cache_tb:.2f} TB")
            lines.append(f"    - Logs: {s.storage.storage_logs_tb:.2f} TB")
            lines.append(f"    - Operacional: {s.storage.storage_operational_tb:.2f} TB")
            lines.append(f"  • IOPS (pico): {s.storage.iops_read_peak:,} R / {s.storage.iops_write_peak:,} W")
            lines.append(f"  • IOPS (steady): {s.storage.iops_read_steady:,} R / {s.storage.iops_write_steady:,} W")
            lines.append(f"  • Throughput (pico): {s.storage.throughput_read_peak_gbps:.2f} R / {s.storage.throughput_write_peak_gbps:.2f} W GB/s")
            lines.append(f"  • Throughput (steady): {s.storage.throughput_read_steady_gbps:.2f} R / {s.storage.throughput_write_steady_gbps:.2f} W GB/s")
            lines.append("")
        
        lines.append("INFRAESTRUTURA FÍSICA:")
        lines.append(f"  • Energia (Compute): {s.total_power_kw:.1f} kW")
        lines.append(f"  • Energia (Storage): {s.storage_power_kw:.1f} kW")
        lines.append(f"  • Energia (Total): {s.total_power_kw_with_storage:.1f} kW")
        lines.append(f"  • Rack (Compute): {s.total_rack_u}U")
        lines.append(f"  • Rack (Storage): {s.storage_rack_u}U")
        lines.append(f"  • Rack (Total): {s.total_rack_u_with_storage}U")
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
        result = {
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
                "total_heat_btu_hr": round(s.total_heat_btu_hr, 0),
                "storage_power_kw": round(s.storage_power_kw, 2),
                "storage_rack_u": s.storage_rack_u,
                "total_power_kw_with_storage": round(s.total_power_kw_with_storage, 2),
                "total_rack_u_with_storage": s.total_rack_u_with_storage
            }
        }
        
        # Adicionar storage se disponível
        if s.storage:
            result["results"]["storage"] = {
                "storage_model_tb": round(s.storage.storage_model_tb, 3),
                "storage_cache_tb": round(s.storage.storage_cache_tb, 3),
                "storage_logs_tb": round(s.storage.storage_logs_tb, 3),
                "storage_operational_tb": round(s.storage.storage_operational_tb, 3),
                "storage_total_tb": round(s.storage.storage_total_tb, 3),
                "iops_read_peak": s.storage.iops_read_peak,
                "iops_write_peak": s.storage.iops_write_peak,
                "iops_read_steady": s.storage.iops_read_steady,
                "iops_write_steady": s.storage.iops_write_steady,
                "throughput_read_peak_gbps": round(s.storage.throughput_read_peak_gbps, 2),
                "throughput_write_peak_gbps": round(s.storage.throughput_write_peak_gbps, 2),
                "throughput_read_steady_gbps": round(s.storage.throughput_read_steady_gbps, 2),
                "throughput_write_steady_gbps": round(s.storage.throughput_write_steady_gbps, 2)
            }
            result["rationale_storage"] = s.storage.rationale
        
        return result
    
    return {
        "inputs": {
            "model": model.name,
            "server": server.name,
            "storage": storage.name,
            "concurrency": concurrency,
            "effective_context": effective_context,
            "kv_precision": kv_precision
        },
        "storage_profile": {
            "name": storage.name,
            "type": storage.type,
            "capacity_total_tb": storage.capacity_total_tb,
            "usable_capacity_tb": storage.usable_capacity_tb,
            "iops_read_max": storage.iops_read_max,
            "iops_write_max": storage.iops_write_max,
            "throughput_read_mbps": storage.throughput_read_mbps,
            "throughput_write_mbps": storage.throughput_write_mbps,
            "block_size_kb_read": storage.block_size_kb_read,
            "block_size_kb_write": storage.block_size_kb_write,
            "latency_read_ms_p50": storage.latency_read_ms_p50,
            "latency_read_ms_p99": storage.latency_read_ms_p99,
            "latency_write_ms_p50": storage.latency_write_ms_p50,
            "latency_write_ms_p99": storage.latency_write_ms_p99,
            "rack_units_u": storage.rack_units_u,
            "power_kw": storage.power_kw
        },
        "scenarios": {
            "minimum": scenario_to_dict(scenarios["minimum"]),
            "recommended": scenario_to_dict(scenarios["recommended"]),
            "ideal": scenario_to_dict(scenarios["ideal"])
        },
        "warnings": warnings
    }
