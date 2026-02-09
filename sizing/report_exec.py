"""
Geração de relatório executivo (resumo para terminal e relatório Markdown executivo).
"""

from typing import Dict
from .calc_scenarios import ScenarioResult
from .models import ModelSpec
from .servers import ServerSpec


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


def format_executive_markdown(
    model: ModelSpec,
    server: ServerSpec,
    scenarios: Dict[str, ScenarioResult],
    concurrency: int,
    effective_context: int,
    kv_precision: str
) -> str:
    """
    Gera relatório executivo completo em Markdown.
    
    Returns:
        String com relatório executivo formatado em Markdown
    """
    lines = []
    
    # Título
    lines.append("# Relatório Executivo - Sizing de Infraestrutura LLM")
    lines.append("")
    lines.append(f"**Modelo:** {model.name}  ")
    lines.append(f"**Servidor:** {server.name}  ")
    lines.append(f"**Data:** {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ")
    lines.append("")
    lines.append("---")
    lines.append("")
    
    # Sumário Executivo
    lines.append("## Sumário Executivo")
    lines.append("")
    lines.append(f"Para sustentar **{concurrency:,} sessões simultâneas** com contexto de **{effective_context:,} tokens** ")
    lines.append(f"utilizando o modelo **{model.name}**, a infraestrutura é dimensionada por **memória GPU (KV cache)**.")
    lines.append("")
    lines.append(f"O principal limitador é o consumo de HBM para armazenar o estado de atenção (KV cache) de cada sessão ativa.")
    lines.append("")
    
    rec = scenarios["recommended"]
    lines.append(f"**Recomendação:** {rec.nodes_final} nós DGX {server.name} ")
    lines.append(f"({rec.total_power_kw:.1f} kW, {rec.total_rack_u}U rack) com tolerância a falhas {rec.config.ha_mode.upper()}.")
    lines.append("")
    lines.append("---")
    lines.append("")
    
    # Cenários Avaliados
    lines.append("## Cenários Avaliados")
    lines.append("")
    lines.append("| Cenário | Objetivo | Tolerância a Falhas | Risco Operacional |")
    lines.append("|---------|----------|---------------------|-------------------|")
    lines.append("| **Mínimo** | Atender no limite | Nenhuma | Alto |")
    lines.append("| **Recomendado** | Produção estável | Falha simples (N+1) | Médio |")
    lines.append("| **Ideal** | Alta resiliência | Falhas múltiplas (N+2) | Baixo |")
    lines.append("")
    lines.append("Avaliar múltiplos cenários é essencial para equilibrar custo de investimento com risco operacional.")
    lines.append("")
    lines.append("---")
    lines.append("")
    
    # Informações do Modelo
    lines.append("## Informações do Modelo Avaliado")
    lines.append("")
    lines.append("| Item | Valor |")
    lines.append("|------|-------|")
    lines.append(f"| Modelo | {model.name} |")
    lines.append(f"| Número de camadas | {model.num_layers} |")
    lines.append(f"| Contexto máximo | {model.max_position_embeddings:,} tokens |")
    lines.append(f"| Padrão de atenção | {model.attention_pattern} |")
    lines.append(f"| Precisão KV cache | {kv_precision.upper()} |")
    lines.append("")
    lines.append(f"O modelo consome memória viva (KV cache) proporcional ao contexto e concorrência.")
    lines.append("")
    lines.append("---")
    lines.append("")
    
    # Consumo Unitário
    lines.append("## Consumo Unitário do Modelo")
    lines.append("")
    lines.append("| Recurso | Consumo por Sessão | Significado Operacional |")
    lines.append("|---------|-------------------|------------------------|")
    lines.append(f"| KV cache | {rec.vram.vram_per_session_gib:.2f} GiB | Memória ocupada enquanto sessão está ativa |")
    lines.append(f"| GPU HBM | {(rec.vram.vram_per_session_gib/rec.vram.hbm_total_gib*100):.1f}% de um nó | Fração da capacidade GPU consumida |")
    lines.append("")
    lines.append("Cada sessão ativa 'reserva' parte do servidor. A soma das reservas define o limite físico do nó.")
    lines.append("")
    lines.append("---")
    lines.append("")
    
    # Resultados por Cenário
    lines.append("## Resultados por Cenário")
    lines.append("")
    
    for key in ["minimum", "recommended", "ideal"]:
        s = scenarios[key]
        lines.append(f"### Cenário {s.config.name}")
        lines.append("")
        lines.append("| Métrica | Valor |")
        lines.append("|---------|-------|")
        lines.append(f"| Nós DGX | {s.nodes_final} |")
        lines.append(f"| Sessões por nó (capacidade) | {s.vram.sessions_per_node} |")
        lines.append(f"| Sessões por nó (operando) | {s.sessions_per_node_effective} |")
        lines.append(f"| KV por sessão | {s.vram.vram_per_session_gib:.2f} GiB |")
        lines.append(f"| VRAM total por nó | {s.vram_total_node_effective_gib:.1f} GiB ({s.hbm_utilization_ratio_effective*100:.1f}% HBM) |")
        lines.append(f"| Energia total | {s.total_power_kw:.1f} kW |")
        lines.append(f"| Espaço em rack | {s.total_rack_u}U |")
        lines.append(f"| Arquitetura HA | {s.config.ha_mode.upper()} |")
        lines.append("")
        
        # Parágrafo executivo
        if key == "minimum":
            lines.append(f"**Análise:** Opera no limite da capacidade sem margem para picos ou falhas. ")
            lines.append(f"Risco operacional **alto** - qualquer indisponibilidade de hardware afeta o serviço diretamente.")
        elif key == "recommended":
            lines.append(f"**Análise:** Equilibra eficiência e resiliência. Suporta picos de até {s.config.peak_headroom_ratio*100:.0f}% ")
            lines.append(f"e tolera falha de 1 nó sem degradação do serviço. **Adequado para produção.**")
        else:  # ideal
            lines.append(f"**Análise:** Máxima resiliência com margem para múltiplas falhas e picos elevados. ")
            lines.append(f"Custo maior, mas risco operacional **mínimo**. Ideal para serviços críticos.")
        lines.append("")
    
    lines.append("---")
    lines.append("")
    
    # Comparação
    lines.append("## Comparação Executiva dos Cenários")
    lines.append("")
    lines.append("| Critério | Mínimo | Recomendado | Ideal |")
    lines.append("|----------|--------|-------------|-------|")
    lines.append(f"| Nós DGX | {scenarios['minimum'].nodes_final} | {scenarios['recommended'].nodes_final} | {scenarios['ideal'].nodes_final} |")
    lines.append(f"| Energia (kW) | {scenarios['minimum'].total_power_kw:.1f} | {scenarios['recommended'].total_power_kw:.1f} | {scenarios['ideal'].total_power_kw:.1f} |")
    lines.append(f"| Rack (U) | {scenarios['minimum'].total_rack_u} | {scenarios['recommended'].total_rack_u} | {scenarios['ideal'].total_rack_u} |")
    lines.append(f"| Tolerância a falhas | Nenhuma | 1 nó | 2 nós |")
    lines.append(f"| Risco operacional | Alto | Médio | Baixo |")
    lines.append("")
    lines.append(f"**Conclusão:** O cenário **RECOMENDADO** oferece o melhor equilíbrio custo-risco para operação em produção.")
    lines.append("")
    lines.append("---")
    lines.append("")
    
    # Recomendação Final
    lines.append("## Recomendação Final")
    lines.append("")
    lines.append(f"Recomenda-se o **cenário RECOMENDADO** com **{rec.nodes_final} nós DGX {server.name}**, que:")
    lines.append("")
    lines.append(f"- Atende os requisitos de capacidade ({concurrency:,} sessões)")
    lines.append(f"- Suporta picos de até {rec.config.peak_headroom_ratio*100:.0f}%")
    lines.append(f"- Tolera falha de 1 nó sem degradação ({rec.config.ha_mode.upper()})")
    lines.append(f"- Consome {rec.total_power_kw:.1f} kW e ocupa {rec.total_rack_u}U de rack")
    lines.append(f"- Mantém risco operacional em nível **aceitável** para produção")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("*Relatório gerado automaticamente pelo sistema de sizing de infraestrutura LLM*")
    lines.append("")
    
    return "\n".join(lines)
