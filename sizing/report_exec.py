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
    lines.append("-" * 155)
    header = f"{'Cenário':<20} {'Nós':<8} {'kW Total':<12} {'Rack Total':<12} {'Storage Recomendado (TB)':<40} {'Sessões/Nó':<12} {'KV/Sessão (GiB)':<18}"
    lines.append(header)
    lines.append("-" * 155)
    
    for key in ["minimum", "recommended", "ideal"]:
        s = scenarios[key]
        if s.storage:
            # Mostrar valor RECOMENDADO (com margem)
            storage_base = s.storage.storage_total_base_tb
            storage_recommended = s.storage.storage_total_recommended_tb
            margin_pct = s.storage.margin_percent * 100
            storage_display = f"{storage_recommended:.1f} (base: {storage_base:.1f} TB + {margin_pct:.0f}%)"
        else:
            storage_display = "N/A"
        
        row = f"{s.config.name:<20} {s.nodes_final:<8} {s.total_power_kw_with_storage:<12.1f} {s.total_rack_u_with_storage:<12} {storage_display:<40} {s.vram.sessions_per_node:<12} {s.vram.vram_per_session_gib:<18.2f}"
        lines.append(row)
    
    lines.append("-" * 155)
    lines.append("")
    
    # Nota sobre margem de capacidade
    if scenarios["recommended"].storage and scenarios["recommended"].storage.margin_applied:
        margin_pct = scenarios["recommended"].storage.margin_percent * 100
        margin_source = scenarios["recommended"].storage.rationale.get("capacity_policy", {}).get("source", "parameters.json")
        lines.append(f"ℹ️  Os valores de storage apresentados já consideram margem adicional de {margin_pct:.0f}% conforme política de capacidade definida em {margin_source}.")
        lines.append("")
    
    # Recomendação
    rec = scenarios["recommended"]
    storage_info = f", {rec.storage.storage_total_recommended_tb:.1f} TB storage" if rec.storage else ""
    lines.append(
        f"✓ Cenário RECOMENDADO ({rec.nodes_final} nós, {rec.total_power_kw_with_storage:.1f} kW total, {rec.total_rack_u_with_storage}U total{storage_info}) "
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
    kv_precision: str,
    storage_name: str = "N/A"
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
    lines.append(f"utilizando o modelo **{model.name}**, a infraestrutura é dimensionada por **memória GPU (KV cache)** e **storage**.")
    lines.append("")
    lines.append(f"O principal limitador de capacidade é o consumo de HBM para armazenar o estado de atenção (KV cache) de cada sessão ativa. ")
    lines.append(f"Storage é dimensionado para operação contínua (pesos do modelo, cache de runtime, logs e auditoria), ")
    lines.append(f"garantindo resiliência, tempo de recuperação e governança operacional.")
    lines.append("")
    
    rec = scenarios["recommended"]
    storage_rec = rec.storage if rec.storage else None
    lines.append(f"**Recomendação:** {rec.nodes_final} nós DGX {server.name} ")
    if storage_rec:
        lines.append(f"({rec.total_power_kw:.1f} kW, {rec.total_rack_u}U rack, {storage_rec.storage_total_recommended_tb:.1f} TB storage) ")
    else:
        lines.append(f"({rec.total_power_kw:.1f} kW, {rec.total_rack_u}U rack) ")
    lines.append(f"com tolerância a falhas {rec.config.ha_mode.upper()}.")
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
        lines.append(f"| **Energia (Compute + Storage)** | **{s.total_power_kw_with_storage:.1f} kW** ({s.total_power_kw:.1f} + {s.storage_power_kw:.1f}) |")
        lines.append(f"| **Rack (Compute + Storage)** | **{s.total_rack_u_with_storage}U** ({s.total_rack_u} + {s.storage_rack_u}) |")
        
        # Storage metrics
        if s.storage:
            st = s.storage
            lines.append(f"| **Storage total** | **{st.storage_total_recommended_tb:.2f} TB** |")
            lines.append(f"| Storage (modelo) | {st.storage_model_recommended_tb:.2f} TB |")
            lines.append(f"| Storage (cache) | {st.storage_cache_recommended_tb:.2f} TB |")
            lines.append(f"| Storage (logs) | {st.storage_logs_recommended_tb:.2f} TB |")
            lines.append(f"| IOPS (pico R/W) | {st.iops_read_peak:,} / {st.iops_write_peak:,} |")
            lines.append(f"| Throughput (pico R/W) | {st.throughput_read_peak_gbps:.1f} / {st.throughput_write_peak_gbps:.1f} GB/s |")
        
        lines.append(f"| Arquitetura HA | {s.config.ha_mode.upper()} |")
        lines.append("")
        
        # Parágrafo executivo
        if key == "minimum":
            lines.append(f"**Análise Computacional:** Opera no limite da capacidade sem margem para picos ou falhas. ")
            lines.append(f"Risco operacional **alto** - qualquer indisponibilidade de hardware afeta o serviço diretamente. ")
            if s.storage:
                lines.append(f"**Análise Storage:** Volumetria recomendada {s.storage.storage_total_recommended_tb:.1f} TB (base: {s.storage.storage_total_base_tb:.1f} TB) para operação steady-state. ")
                lines.append(f"IOPS e throughput dimensionados sem margem. Risco de gargalo em scale-out ou restart simultâneo.")
        elif key == "recommended":
            lines.append(f"**Análise Computacional:** Equilibra eficiência e resiliência. Suporta picos de até {s.config.peak_headroom_ratio*100:.0f}% ")
            lines.append(f"e tolera falha de 1 nó sem degradação do serviço. **Adequado para produção.** ")
            if s.storage:
                lines.append(f"**Análise Storage:** {s.storage.storage_total_recommended_tb:.1f} TB recomendado (base: {s.storage.storage_total_base_tb:.1f} TB) com margem de capacidade. ")
                lines.append(f"IOPS e throughput suportam restart de 25% dos nós + burst de logs. Tempo de recuperação aceitável.")
        else:  # ideal
            lines.append(f"**Análise Computacional:** Máxima resiliência com margem para múltiplas falhas e picos elevados. ")
            lines.append(f"Custo maior, mas risco operacional **mínimo**. Ideal para serviços críticos. ")
            if s.storage:
                lines.append(f"**Análise Storage:** {s.storage.storage_total_recommended_tb:.1f} TB recomendado (base: {s.storage.storage_total_base_tb:.1f} TB) com margem ampla para máxima resiliência. ")
                lines.append(f"IOPS e throughput suportam falhas em cascata. Retenção estendida de logs (90 dias). Máxima resiliência.")
        lines.append("")
    
    lines.append("---")
    lines.append("")
    
    # Comparação
    lines.append("## Comparação Executiva dos Cenários")
    lines.append("")
    lines.append("| Critério | Mínimo | Recomendado | Ideal |")
    lines.append("|----------|--------|-------------|-------|")
    lines.append(f"| Nós DGX | {scenarios['minimum'].nodes_final} | {scenarios['recommended'].nodes_final} | {scenarios['ideal'].nodes_final} |")
    lines.append(f"| Energia Total (kW) | {scenarios['minimum'].total_power_kw_with_storage:.1f} | {scenarios['recommended'].total_power_kw_with_storage:.1f} | {scenarios['ideal'].total_power_kw_with_storage:.1f} |")
    lines.append(f"| Rack Total (U) | {scenarios['minimum'].total_rack_u_with_storage} | {scenarios['recommended'].total_rack_u_with_storage} | {scenarios['ideal'].total_rack_u_with_storage} |")
    
    # Storage comparison
    if scenarios['minimum'].storage and scenarios['recommended'].storage and scenarios['ideal'].storage:
        st_min = scenarios['minimum'].storage
        st_rec = scenarios['recommended'].storage
        st_ideal = scenarios['ideal'].storage
        lines.append(f"| Storage (TB) | {st_min.storage_total_recommended_tb:.1f} | {st_rec.storage_total_recommended_tb:.1f} | {st_ideal.storage_total_recommended_tb:.1f} |")
        lines.append(f"| IOPS pico (R) | {st_min.iops_read_peak:,} | {st_rec.iops_read_peak:,} | {st_ideal.iops_read_peak:,} |")
        lines.append(f"| Throughput pico (R) | {st_min.throughput_read_peak_gbps:.1f} GB/s | {st_rec.throughput_read_peak_gbps:.1f} GB/s | {st_ideal.throughput_read_peak_gbps:.1f} GB/s |")
    
    lines.append(f"| Tolerância a falhas | Nenhuma | 1 nó | 2 nós |")
    lines.append(f"| Risco operacional | Alto | Médio | Baixo |")
    lines.append("")
    lines.append(f"**Conclusão:** O cenário **RECOMENDADO** oferece o melhor equilíbrio custo-risco para operação em produção. ")
    if scenarios['recommended'].storage:
        lines.append(f"Storage subdimensionado compromete resiliência e tempo de recuperação, mesmo com GPUs suficientes.")
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
    
    if storage_rec:
        lines.append(f"- Requer {storage_rec.storage_total_recommended_tb:.1f} TB de storage ({storage_name}, incluindo margem de capacidade)")
        lines.append(f"  - IOPS pico: {storage_rec.iops_read_peak:,} leitura / {storage_rec.iops_write_peak:,} escrita")
        lines.append(f"  - Throughput pico: {storage_rec.throughput_read_peak_gbps:.1f} GB/s leitura / {storage_rec.throughput_write_peak_gbps:.1f} GB/s escrita")
    
    lines.append(f"- Mantém risco operacional em nível **aceitável** para produção")
    lines.append("")
    lines.append("**Governança:** Storage é recurso crítico. Subdimensionamento impacta:")
    lines.append("- Tempo de recuperação (restart lento)")
    lines.append("- Escalabilidade (gargalo em scale-out)")
    lines.append("- Auditoria e conformidade (retenção inadequada de logs)")
    lines.append("")
    lines.append("---")
    lines.append("")
    
    # Glossário Executivo de Termos
    lines.append("## Glossário Executivo de Termos")
    lines.append("")
    lines.append("| Métrica | O que significa | Por que importa para a decisão | Impacto se estiver errado |")
    lines.append("| ------------------------------- | --------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |")
    lines.append("| **Nós DGX** | Quantidade de servidores de IA necessários para atender a carga analisada. | Define investimento em hardware e influencia energia, rack e custo total. | Subdimensionamento causa indisponibilidade; superdimensionamento aumenta custo. |")
    lines.append("| **Sessões por nó (capacidade)** | Número máximo teórico de conversas simultâneas que um servidor suporta. | Indica o limite físico do servidor antes de atingir saturação de memória. | Operar no limite reduz margem para picos e aumenta risco de instabilidade. |")
    lines.append("| **Sessões por nó (operando)** | Número real de sessões em uso no cenário avaliado. | Mostra a folga operacional disponível. | Se muito próximo do limite, o sistema fica vulnerável a picos de uso. |")
    lines.append("| **KV por sessão** | Memória de GPU consumida por cada conversa ativa. | É o principal fator que determina quantas sessões cabem por servidor. | Conversas mais longas aumentam consumo e reduzem capacidade total. |")
    lines.append("| **VRAM total por nó** | Memória total da GPU utilizada pelo modelo, runtime e sessões. | Indica quão próximo o servidor está do limite físico. | Uso excessivo pode causar falhas ou degradação de performance. |")
    lines.append("| **Energia (Compute + Storage)** | Consumo total de energia dos servidores de IA e do storage. | Impacta custo operacional mensal e capacidade elétrica do datacenter. | Subdimensionar pode causar sobrecarga elétrica; superdimensionar eleva custo. |")
    lines.append("| **Rack (Compute + Storage)** | Espaço físico ocupado por servidores e storage no datacenter. | Define viabilidade física de implantação e expansão futura. | Espaço insuficiente limita crescimento. |")
    lines.append("| **Storage total** | Capacidade total de armazenamento necessária para rodar o modelo e sustentar o sistema (modelo + cache + logs). | Representa o espaço mínimo necessário para operar o ambiente com segurança. | Falta de espaço pode impedir inicialização, gravação de logs ou escala do sistema. Recomenda-se dimensionar ~50% acima do mínimo calculado. |")
    lines.append("| **Storage (modelo)** | Espaço necessário para armazenar os arquivos do modelo (pesos e artefatos). | Essencial para subir o sistema e permitir reinicializações rápidas. | Se insuficiente, o sistema pode não iniciar corretamente. Recomenda-se margem adicional. |")
    lines.append("| **Storage (cache)** | Espaço para arquivos temporários e dados intermediários usados na execução. | Garante funcionamento contínuo e estável do ambiente. | Pode gerar falhas ou degradação se o espaço se esgotar. |")
    lines.append("| **Storage (logs)** | Espaço destinado ao armazenamento de logs operacionais e auditoria. | Fundamental para rastreabilidade, análise de incidentes e governança. | Falta de espaço compromete auditoria e diagnóstico de problemas. |")
    lines.append("| **IOPS (pico R/W)** | Número máximo de operações de leitura e escrita por segundo no pico. | Determina se o storage suporta eventos como subida simultânea de múltiplos servidores. | Gargalo de IOPS aumenta tempo de recuperação e escala. |")
    lines.append("| **Throughput (pico R/W)** | Volume máximo de dados transferidos por segundo no pico de uso. | Afeta tempo de carregamento do modelo e recuperação após falhas. | Throughput insuficiente aumenta tempo de indisponibilidade. |")
    lines.append("| **Arquitetura HA** | Nível de tolerância a falhas adotado (ex.: NONE, N+1, N+2). | Define o quanto o sistema continua operando mesmo após falhas de hardware. | Ausência de HA pode causar interrupção total do serviço. |")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("*Relatório gerado automaticamente pelo sistema de sizing de infraestrutura LLM*")
    lines.append("")
    
    return "\n".join(lines)
