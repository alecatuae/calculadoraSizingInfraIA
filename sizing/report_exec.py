"""
Geração de relatório executivo (resumo para terminal).
"""

from typing import Dict
from .calc_scenarios import ScenarioResult


def format_exec_summary(
    model_name: str,
    server_name: str,
    effective_context: int,
    concurrency: int,
    kv_precision: str,
    scenarios: Dict[str, ScenarioResult],
    text_report_path: str,
    json_report_path: str
) -> str:
    """
    Gera resumo executivo para exibição no terminal.
    
    Returns:
        String com resumo formatado
    """
    lines = []
    
    # Cabeçalho
    lines.append("=" * 80)
    lines.append("RESUMO EXECUTIVO - SIZING DE INFERÊNCIA LLM")
    lines.append("=" * 80)
    lines.append("")
    
    lines.append(f"Modelo:              {model_name}")
    lines.append(f"Servidor:            {server_name}")
    lines.append(f"Contexto Efetivo:    {effective_context:,} tokens")
    lines.append(f"Concorrência Alvo:   {concurrency:,} sessões simultâneas")
    lines.append(f"Precisão KV Cache:   {kv_precision.upper()}")
    lines.append("")
    
    # Tabela de cenários
    lines.append("-" * 100)
    header = f"{'Cenário':<20} {'Nós DGX':<10} {'Energia (kW)':<15} {'Rack (U)':<10} {'Sessões/Nó':<12} {'KV/Sessão (GiB)':<18}"
    lines.append(header)
    lines.append("-" * 100)
    
    for key in ["minimum", "recommended", "ideal"]:
        s = scenarios[key]
        row = f"{s.config.name:<20} {s.nodes_final:<10} {s.total_power_kw:<15.1f} {s.total_rack_u:<10} {s.vram.sessions_per_node:<12} {s.vram.vram_per_session_gib:<18.2f}"
        lines.append(row)
    
    lines.append("-" * 100)
    lines.append("")
    
    # Recomendação
    rec = scenarios["recommended"]
    lines.append(
        f"✓ Cenário RECOMENDADO ({rec.nodes_final} nós, {rec.total_power_kw:.1f} kW, {rec.total_rack_u}U) "
        f"atende os requisitos com tolerância a falhas ({rec.config.ha_mode.upper()})."
    )
    lines.append("")
    
    # Paths dos relatórios
    lines.append("=" * 80)
    lines.append("📄 Relatórios completos salvos em:")
    lines.append(f"   • Texto:  {text_report_path}")
    lines.append(f"   • JSON:   {json_report_path}")
    lines.append("")
    
    return "\n".join(lines)
