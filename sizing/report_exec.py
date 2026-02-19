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
    json_report_path: str,
    sizing_mode: str = "concurrency_driven"
) -> str:
    """
    Gera resumo executivo para exibição no terminal.

    Returns:
        String com resumo formatado
    """
    lines = []

    lines.append("=" * 80)
    lines.append("RESUMO EXECUTIVO - SIZING DE INFRAESTRUTURA PARA INFERENCIA")
    lines.append("=" * 80)
    lines.append("")

    lines.append(f"Modelo:              {model_name}")
    lines.append(f"Servidor:            {server_name}")
    lines.append(f"Contexto Efetivo:    {effective_context:,} tokens")
    lines.append(f"Concorrencia Alvo:   {concurrency:,} sessoes simultaneas")
    lines.append(f"Precisao KV Cache:   {kv_precision.upper()}")
    modo_label = "SLO-Driven" if sizing_mode == "slo_driven" else "Concorrencia-Driven"
    lines.append(f"Modo de Sizing:      {modo_label}")
    lines.append("")

    has_latency = any(scenarios[k].latency is not None for k in scenarios)
    qual_pt = {'excellent': 'Excelente', 'good': 'Bom', 'acceptable': 'Aceitavel', 'slow': 'Lento'}
    status_map = {'OK': '[OK]', 'SLO_MARGINAL': '[MARGINAL]', 'SLO_VIOLATION': '[VIOLADO]'}

    if has_latency:
        lines.append("-" * 195)
        header = (f"{'Cenario':<16} {'Nos':<6} {'kW Total':<10} {'Rack':<8} "
                  f"{'Storage (TB)':<26} {'Sess./No':<10} {'KV/Sess.(GiB)':<15} "
                  f"{'TTFT P50':<22} {'TPOT':<22} {'SLO':<12}")
        lines.append(header)
        lines.append("-" * 195)

        for key in ["minimum", "recommended", "ideal"]:
            s = scenarios[key]
            storage_display = (
                f"{s.storage.storage_total_recommended_tb:.1f} (+{s.storage.margin_percent*100:.0f}%)"
                if s.storage else "N/A"
            )
            la = s.latency
            if la:
                ttft_str = f"{la.ttft_p50_ms:.0f}ms ({qual_pt.get(la.ttft_quality, la.ttft_quality)})"
                if la.ttft_p50_ms >= 99000:
                    ttft_str = "inf (saturado)"
                tpot_str = f"{la.tpot_tokens_per_sec:.1f} tok/s ({qual_pt.get(la.tpot_quality, la.tpot_quality)})"
                slo_str = status_map.get(la.status, la.status)
            else:
                ttft_str = tpot_str = slo_str = "N/A"

            row = (f"{s.config.name:<16} {s.nodes_final:<6} {s.total_power_kw_with_storage:<10.1f} "
                   f"{s.total_rack_u_with_storage:<8} {storage_display:<26} "
                   f"{s.vram.sessions_per_node:<10} {s.vram.vram_per_session_gib:<15.2f} "
                   f"{ttft_str:<22} {tpot_str:<22} {slo_str:<12}")
            lines.append(row)

        lines.append("-" * 195)
    else:
        lines.append("-" * 155)
        header = f"{'Cenario':<20} {'Nos':<8} {'kW Total':<12} {'Rack Total':<12} {'Storage Recomendado (TB)':<40} {'Sess./No':<12} {'KV/Sess.(GiB)':<18}"
        lines.append(header)
        lines.append("-" * 155)

        for key in ["minimum", "recommended", "ideal"]:
            s = scenarios[key]
            storage_display = (
                f"{s.storage.storage_total_recommended_tb:.1f} (base: {s.storage.storage_total_base_tb:.1f} TB + {s.storage.margin_percent*100:.0f}%)"
                if s.storage else "N/A"
            )
            row = f"{s.config.name:<20} {s.nodes_final:<8} {s.total_power_kw_with_storage:<12.1f} {s.total_rack_u_with_storage:<12} {storage_display:<40} {s.vram.sessions_per_node:<12} {s.vram.vram_per_session_gib:<18.2f}"
            lines.append(row)

        lines.append("-" * 155)

    lines.append("")

    # SLO-Driven: exibir concorrência máxima por cenário
    any_slo_cap = any(scenarios[k].slo_capacity is not None for k in scenarios)
    if any_slo_cap and sizing_mode == "slo_driven":
        lines.append("Concorrencia maxima atendivel pelos SLOs:")
        for key in ["minimum", "recommended", "ideal"]:
            sc = scenarios[key].slo_capacity
            if sc:
                viavel = "OK" if sc.is_feasible else "INVIAVEL"
                lines.append(
                    f"  {scenarios[key].config.name:<16}: {sc.max_concurrency_combined:>6} sessoes"
                    f" | Gargalo: {sc.limiting_factor:<10} | Util. max: {sc.util_max_from_ttft*100:.1f}%"
                    f" | [{viavel}]"
                )
        lines.append("")

    # Calibração: mostrar recomendações quando há violação no modo Concorrência-Driven
    any_calib = any(scenarios[k].calibration is not None for k in scenarios)
    if any_calib and sizing_mode == "concurrency_driven":
        lines.append("Calibracao recomendada para atender os SLOs:")
        for key in ["minimum", "recommended", "ideal"]:
            cal = scenarios[key].calibration
            if cal:
                lines.append(
                    f"  {scenarios[key].config.name:<16}: max. c/SLOs = {cal.max_concurrency_current_nodes:>6} sessoes"
                    f" | nos recomendados = {cal.nodes_recommended or 'N/A'}"
                    f" (+{cal.extra_nodes_needed} nos extras)"
                    f" | gargalo: {cal.limiting_factor}"
                )
        lines.append("")

    # Nota sobre margem
    if scenarios["recommended"].storage and scenarios["recommended"].storage.margin_applied:
        margin_pct = scenarios["recommended"].storage.margin_percent * 100
        margin_source = scenarios["recommended"].storage.rationale.get("capacity_policy", {}).get("source", "parameters.json")
        platform_per_server_tb = scenarios["recommended"].storage.platform_per_server_tb
        lines.append(f"[INFO] Storage considera margem adicional de {margin_pct:.0f}% ({margin_source}).")
        lines.append(f"[INFO] Storage total inclui volume estrutural da plataforma ({platform_per_server_tb:.2f} TB/servidor).")
        lines.append("")

    # Recomendação
    rec = scenarios["recommended"]
    storage_info = f", {rec.storage.storage_total_recommended_tb:.1f} TB storage" if rec.storage else ""
    rec_la = rec.latency
    latency_info = ""
    if rec_la:
        ttft_display = f"{rec_la.ttft_p50_ms:.0f}ms" if rec_la.ttft_p50_ms < 99000 else "inf"
        slo_status_txt = {"OK": "[OK]", "SLO_MARGINAL": "[MARGINAL]", "SLO_VIOLATION": "[VIOLADO]"}.get(rec_la.status, "")
        latency_info = f", TTFT {ttft_display} / TPOT {rec_la.tpot_tokens_per_sec:.1f} tok/s {slo_status_txt}"

    lines.append(
        f"Cenario RECOMENDADO ({rec.nodes_final} nos, {rec.total_power_kw_with_storage:.1f} kW total, "
        f"{rec.total_rack_u_with_storage}U total{storage_info}{latency_info}) "
        f"atende os requisitos com tolerancia a falhas ({rec.config.ha_mode.upper()})."
    )
    lines.append("")

    lines.append("=" * 80)
    lines.append("Relatorios completos salvos em:")
    lines.append(f"   Texto:  {text_report_path}")
    lines.append(f"   JSON:   {json_report_path}")
    lines.append("")

    return "\n".join(lines)


def format_executive_markdown(
    model: ModelSpec,
    server: ServerSpec,
    scenarios: Dict[str, ScenarioResult],
    concurrency: int,
    effective_context: int,
    kv_precision: str,
    storage_name: str = "N/A",
    sizing_mode: str = "concurrency_driven"
) -> str:
    """
    Gera relatório executivo completo em Markdown.

    Returns:
        String com relatório executivo formatado em Markdown
    """
    lines = []

    lines.append("# Relatorio Executivo - Sizing de Infraestrutura para Inferencia")
    lines.append("")
    lines.append(f"**Modelo:** {model.name}  ")
    lines.append(f"**Servidor:** {server.name}  ")
    lines.append(f"**Data:** {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ")
    modo_label = "SLO-Driven (latencia guia dimensionamento)" if sizing_mode == "slo_driven" else "Concorrencia-Driven (SLOs implicitos de parameters.json)"
    lines.append(f"**Modo de Sizing:** {modo_label}  ")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Sumario Executivo
    lines.append("## Sumario Executivo")
    lines.append("")

    if sizing_mode == "slo_driven":
        lines.append(f"O dimensionamento e guiado pelos **SLOs de latencia** definidos. "
                     f"A ferramenta calcula a **concorrencia maxima atendivel** por cenario.")
    else:
        lines.append(f"Para sustentar **{concurrency:,} sessoes simultaneas** com contexto de **{effective_context:,} tokens** "
                     f"utilizando o modelo **{model.name}**, a infraestrutura e dimensionada por **memoria GPU (KV cache)** e **storage**.")
    lines.append("")
    lines.append(f"O principal limitador de capacidade e o consumo de HBM para armazenar o estado de atencao (KV cache) de cada sessao ativa. "
                 f"Storage e dimensionado para operacao continua (pesos do modelo, cache de runtime, logs e auditoria), "
                 f"garantindo resiliencia, tempo de recuperacao e governanca operacional.")
    lines.append("")

    rec = scenarios["recommended"]
    storage_rec = rec.storage if rec.storage else None
    rec_la = rec.latency

    # Latencia no sumario
    if rec_la:
        ttft_display = f"{rec_la.ttft_p50_ms:.0f}ms" if rec_la.ttft_p50_ms < 99000 else "inf"
        ttft_qual = {'excellent': 'excelente', 'good': 'bom', 'acceptable': 'aceitavel', 'slow': 'lento'}.get(rec_la.ttft_quality, rec_la.ttft_quality)
        tpot_qual = {'excellent': 'excelente', 'good': 'bom', 'acceptable': 'aceitavel', 'slow': 'lento'}.get(rec_la.tpot_quality, rec_la.tpot_quality)
        slo_text = {
            'OK': 'SLOs de latencia atendidos com margem.',
            'SLO_MARGINAL': 'SLOs de latencia atendidos com margem minima -- monitorar em producao.',
            'SLO_VIOLATION': 'SLOs de latencia **nao atendidos** -- gargalo identificado: ' + rec_la.bottleneck.split(' - ')[0] + '.'
        }.get(rec_la.status, '')
        lines.append(f"O cenario recomendado apresenta **TTFT de {ttft_display}** (qualidade: {ttft_qual}) e "
                     f"**TPOT de {rec_la.tpot_tokens_per_sec:.2f} tok/s** (qualidade: {tpot_qual}). {slo_text}")
        lines.append("")

    # Concorrencia maxima (modo SLO-Driven)
    rec_slo = rec.slo_capacity
    if rec_slo and sizing_mode == "slo_driven":
        if rec_slo.is_feasible:
            lines.append(f"**Concorrencia maxima atendivel (cenario RECOMENDADO):** {rec_slo.max_concurrency_combined:,} sessoes "
                         f"(gargalo: {rec_slo.limiting_factor}).")
        else:
            lines.append(f"**Atencao:** SLO de TTFT inviavel para este hardware -- {rec_slo.infeasibility_reason}")
        lines.append("")

    lines.append(f"**Recomendacao:** {rec.nodes_final} nos DGX {server.name} ")
    if storage_rec:
        lines.append(f"({rec.total_power_kw:.1f} kW, {rec.total_rack_u}U rack, {storage_rec.storage_total_recommended_tb:.1f} TB storage) ")
    else:
        lines.append(f"({rec.total_power_kw:.1f} kW, {rec.total_rack_u}U rack) ")
    lines.append(f"com tolerancia a falhas {rec.config.ha_mode.upper()}.")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Cenarios Avaliados
    lines.append("## Cenarios Avaliados")
    lines.append("")
    lines.append("| Cenario | Objetivo | Tolerancia a Falhas | Risco Operacional |")
    lines.append("|---------|----------|---------------------|-------------------|")
    lines.append("| **Minimo** | Atender no limite | Nenhuma | Alto |")
    lines.append("| **Recomendado** | Producao estavel | Falha simples (N+1) | Medio |")
    lines.append("| **Ideal** | Alta resiliencia | Falhas multiplas (N+2) | Baixo |")
    lines.append("")
    lines.append("Avaliar multiplos cenarios e essencial para equilibrar custo de investimento com risco operacional.")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Informacoes do Modelo
    lines.append("## Informacoes do Modelo Avaliado")
    lines.append("")
    lines.append("| Item | Valor |")
    lines.append("|------|-------|")
    lines.append(f"| Modelo | {model.name} |")
    lines.append(f"| Numero de camadas | {model.num_layers} |")
    lines.append(f"| Contexto maximo | {model.max_position_embeddings:,} tokens |")
    lines.append(f"| Padrao de atencao | {model.attention_pattern} |")
    lines.append(f"| Precisao KV cache | {kv_precision.upper()} |")
    lines.append("")
    lines.append(f"O modelo consome memoria viva (KV cache) proporcional ao contexto e concorrencia.")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Consumo Unitario
    lines.append("## Consumo Unitario do Modelo")
    lines.append("")
    lines.append("| Recurso | Consumo por Sessao | Significado Operacional |")
    lines.append("|---------|-------------------|------------------------|")
    lines.append(f"| KV cache | {rec.vram.vram_per_session_gib:.2f} GiB | Memoria ocupada enquanto sessao esta ativa |")
    lines.append(f"| GPU HBM | {(rec.vram.vram_per_session_gib/rec.vram.hbm_total_gib*100):.1f}% de um no | Fracao da capacidade GPU consumida |")
    lines.append("")
    lines.append("Cada sessao ativa reserva parte do servidor. A soma das reservas define o limite fisico do no.")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Resultados por Cenario
    lines.append("## Resultados por Cenario")
    lines.append("")

    qual_label_md = {'excellent': 'Excelente', 'good': 'Bom', 'acceptable': 'Aceitavel', 'slow': 'Lento'}
    slo_label_md = {'OK': '[OK] Atende', 'SLO_MARGINAL': '[MARGINAL]', 'SLO_VIOLATION': '[VIOLADO]'}

    for key in ["minimum", "recommended", "ideal"]:
        s = scenarios[key]
        lines.append(f"### Cenario {s.config.name}")
        lines.append("")
        lines.append("| Metrica | Valor |")
        lines.append("|---------|-------|")
        lines.append(f"| Nos DGX | {s.nodes_final} |")
        lines.append(f"| Sessoes por no (capacidade) | {s.vram.sessions_per_node} |")
        lines.append(f"| Sessoes por no (operando) | {s.sessions_per_node_effective} |")
        lines.append(f"| KV por sessao | {s.vram.vram_per_session_gib:.2f} GiB |")
        lines.append(f"| VRAM total por no | {s.vram_total_node_effective_gib:.1f} GiB ({s.hbm_utilization_ratio_effective*100:.1f}% HBM) |")
        lines.append(f"| **Energia (Compute + Storage)** | **{s.total_power_kw_with_storage:.1f} kW** ({s.total_power_kw:.1f} + {s.storage_power_kw:.1f}) |")
        lines.append(f"| **Rack (Compute + Storage)** | **{s.total_rack_u_with_storage}U** ({s.total_rack_u} + {s.storage_rack_u}) |")

        if s.storage:
            st = s.storage
            lines.append(f"| **Storage total** | **{st.storage_total_recommended_tb:.2f} TB** |")
            lines.append(f"| Storage (modelo) | {st.storage_model_recommended_tb:.2f} TB |")
            lines.append(f"| Storage (cache) | {st.storage_cache_recommended_tb:.2f} TB |")
            lines.append(f"| Storage (logs) | {st.storage_logs_recommended_tb:.2f} TB |")
            lines.append(f"| IOPS (pico R/W) | {st.iops_read_peak:,} / {st.iops_write_peak:,} |")
            lines.append(f"| Throughput (pico R/W) | {st.throughput_read_peak_gbps:.1f} / {st.throughput_write_peak_gbps:.1f} GB/s |")

        lines.append(f"| Arquitetura HA | {s.config.ha_mode.upper()} |")

        # TTFT/TPOT
        if s.latency:
            la = s.latency
            ttft_val = f"{la.ttft_p50_ms:.0f} ms" if la.ttft_p50_ms < 99000 else "inf (saturado)"
            ttft_p99_val = f"{la.ttft_p99_ms:.0f} ms" if la.ttft_p99_ms < 99000 else "inf (saturado)"
            tpot_val = (f"{la.tpot_tokens_per_sec:.2f} tok/s (ITL: {la.itl_ms_per_token:.0f} ms/token)"
                        if la.itl_ms_per_token < 99000 else f"{la.tpot_tokens_per_sec:.2f} tok/s")
            util_val = f"{la.utilization * 100:.1f}%"
            ttft_qual = qual_label_md.get(la.ttft_quality, la.ttft_quality)
            tpot_qual = qual_label_md.get(la.tpot_quality, la.tpot_quality)
            slo_val = slo_label_md.get(la.status, la.status)
            lines.append(f"| **TTFT P50 (latencia 1o token)** | **{ttft_val}** -- {ttft_qual} |")
            lines.append(f"| TTFT P99 | {ttft_p99_val} |")
            lines.append(f"| **TPOT (velocidade streaming)** | **{tpot_val}** -- {tpot_qual} |")
            lines.append(f"| Utilizacao GPU (queuing) | {util_val} |")
            lines.append(f"| Gargalo | {la.bottleneck.split(' - ')[0]} |")
            lines.append(f"| **Status SLO Latencia** | **{slo_val}** |")

        # SLO capacity (modo SLO-Driven)
        if s.slo_capacity and sizing_mode == "slo_driven":
            sc = s.slo_capacity
            if sc.is_feasible:
                lines.append(f"| **Concorrencia max (SLOs)** | **{sc.max_concurrency_combined:,} sessoes** |")
                lines.append(f"| Max por TTFT | {sc.max_concurrency_from_ttft:,} sessoes (util. max: {sc.util_max_from_ttft*100:.1f}%) |")
                lines.append(f"| Max por TPOT | {sc.max_concurrency_from_tpot:,} sessoes (sess./no max: {sc.sessions_per_node_max_from_tpot}) |")
                lines.append(f"| Fator limitante | {sc.limiting_factor} |")
            else:
                lines.append(f"| **SLO de TTFT** | **INVIAVEL** -- {sc.infeasibility_reason[:80]} |")

        # Calibracao (modo Concorrencia-Driven)
        if s.calibration:
            cal = s.calibration
            lines.append(f"| Concorrencia max c/SLOs atuais | {cal.max_concurrency_current_nodes:,} sessoes |")
            lines.append(f"| Nos recomendados para SLOs | {cal.nodes_recommended or 'N/A'} nos (+{cal.extra_nodes_needed} extras) |")

        lines.append("")

        if key == "minimum":
            lines.append(f"**Analise Computacional:** Opera no limite da capacidade sem margem para picos ou falhas. "
                         f"Risco operacional **alto** - qualquer indisponibilidade de hardware afeta o servico diretamente. ")
            if s.storage:
                lines.append(f"**Analise Storage:** Volumetria recomendada {s.storage.storage_total_recommended_tb:.1f} TB "
                             f"(base: {s.storage.storage_total_base_tb:.1f} TB) para operacao steady-state. "
                             f"IOPS e throughput dimensionados sem margem. Risco de gargalo em scale-out ou restart simultaneo.")
        elif key == "recommended":
            lines.append(f"**Analise Computacional:** Equilibra eficiencia e resiliencia. "
                         f"Suporta picos de ate {s.config.peak_headroom_ratio*100:.0f}% "
                         f"e tolera falha de 1 no sem degradacao do servico. **Adequado para producao.** ")
            if s.storage:
                lines.append(f"**Analise Storage:** {s.storage.storage_total_recommended_tb:.1f} TB recomendado "
                             f"(base: {s.storage.storage_total_base_tb:.1f} TB) com margem de capacidade. "
                             f"IOPS e throughput suportam restart de 25% dos nos + burst de logs. Tempo de recuperacao aceitavel.")
        else:
            lines.append(f"**Analise Computacional:** Maxima resiliencia com margem para multiplas falhas e picos elevados. "
                         f"Custo maior, mas risco operacional **minimo**. Ideal para servicos criticos. ")
            if s.storage:
                lines.append(f"**Analise Storage:** {s.storage.storage_total_recommended_tb:.1f} TB recomendado "
                             f"(base: {s.storage.storage_total_base_tb:.1f} TB) com margem ampla para maxima resiliencia. "
                             f"IOPS e throughput suportam falhas em cascata. Retencao estendida de logs (90 dias). Maxima resiliencia.")
        lines.append("")

    lines.append("---")
    lines.append("")

    # Comparacao Executiva
    lines.append("## Comparacao Executiva dos Cenarios")
    lines.append("")
    lines.append("| Criterio | Minimo | Recomendado | Ideal |")
    lines.append("|----------|--------|-------------|-------|")
    lines.append(f"| Nos DGX | {scenarios['minimum'].nodes_final} | {scenarios['recommended'].nodes_final} | {scenarios['ideal'].nodes_final} |")
    lines.append(f"| Energia Total (kW) | {scenarios['minimum'].total_power_kw_with_storage:.1f} | {scenarios['recommended'].total_power_kw_with_storage:.1f} | {scenarios['ideal'].total_power_kw_with_storage:.1f} |")
    lines.append(f"| Rack Total (U) | {scenarios['minimum'].total_rack_u_with_storage} | {scenarios['recommended'].total_rack_u_with_storage} | {scenarios['ideal'].total_rack_u_with_storage} |")

    if scenarios['minimum'].storage and scenarios['recommended'].storage and scenarios['ideal'].storage:
        st_min = scenarios['minimum'].storage
        st_rec = scenarios['recommended'].storage
        st_ideal = scenarios['ideal'].storage
        lines.append(f"| Storage (TB) | {st_min.storage_total_recommended_tb:.1f} | {st_rec.storage_total_recommended_tb:.1f} | {st_ideal.storage_total_recommended_tb:.1f} |")
        lines.append(f"| IOPS pico (R) | {st_min.iops_read_peak:,} | {st_rec.iops_read_peak:,} | {st_ideal.iops_read_peak:,} |")
        lines.append(f"| Throughput pico (R) | {st_min.throughput_read_peak_gbps:.1f} GB/s | {st_rec.throughput_read_peak_gbps:.1f} GB/s | {st_ideal.throughput_read_peak_gbps:.1f} GB/s |")

    lines.append(f"| Tolerancia a falhas | Nenhuma | 1 no | 2 nos |")
    lines.append(f"| Risco operacional | Alto | Medio | Baixo |")

    if any(scenarios[k].latency is not None for k in scenarios):
        def _ttft_str(k):
            la = scenarios[k].latency
            if la is None:
                return "N/A"
            if la.ttft_p50_ms >= 99000:
                return "inf (saturado)"
            return f"{la.ttft_p50_ms:.0f} ms ({qual_label_md.get(la.ttft_quality, la.ttft_quality)})"

        def _tpot_str(k):
            la = scenarios[k].latency
            if la is None:
                return "N/A"
            return f"{la.tpot_tokens_per_sec:.1f} tok/s ({qual_label_md.get(la.tpot_quality, la.tpot_quality)})"

        def _slo_str(k):
            la = scenarios[k].latency
            if la is None:
                return "N/A"
            return slo_label_md.get(la.status, la.status)

        lines.append(f"| **TTFT P50** | {_ttft_str('minimum')} | {_ttft_str('recommended')} | {_ttft_str('ideal')} |")
        lines.append(f"| **TPOT** | {_tpot_str('minimum')} | {_tpot_str('recommended')} | {_tpot_str('ideal')} |")
        lines.append(f"| Status SLO Latencia | {_slo_str('minimum')} | {_slo_str('recommended')} | {_slo_str('ideal')} |")

    if any(scenarios[k].slo_capacity is not None for k in scenarios):
        def _slo_cap_str(k):
            sc = scenarios[k].slo_capacity
            if sc is None:
                return "N/A"
            if not sc.is_feasible:
                return "INVIAVEL"
            return f"{sc.max_concurrency_combined:,} sessoes"

        lines.append(f"| **Concorrencia max (SLOs)** | {_slo_cap_str('minimum')} | {_slo_cap_str('recommended')} | {_slo_cap_str('ideal')} |")

    lines.append("")
    lines.append(f"**Conclusao:** O cenario **RECOMENDADO** oferece o melhor equilibrio custo-risco para operacao em producao. ")
    if scenarios['recommended'].storage:
        lines.append(f"Storage subdimensionado compromete resiliencia e tempo de recuperacao, mesmo com GPUs suficientes.")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Recomendacao Final
    lines.append("## Recomendacao Final")
    lines.append("")
    lines.append(f"Recomenda-se o **cenario RECOMENDADO** com **{rec.nodes_final} nos DGX {server.name}**, que:")
    lines.append("")
    lines.append(f"- Atende os requisitos de capacidade ({concurrency:,} sessoes)")
    lines.append(f"- Suporta picos de ate {rec.config.peak_headroom_ratio*100:.0f}%")
    lines.append(f"- Tolera falha de 1 no sem degradacao ({rec.config.ha_mode.upper()})")
    lines.append(f"- Consome {rec.total_power_kw:.1f} kW e ocupa {rec.total_rack_u}U de rack")

    if storage_rec:
        lines.append(f"- Requer {storage_rec.storage_total_recommended_tb:.1f} TB de storage ({storage_name}, incluindo margem de capacidade)")
        lines.append(f"  - IOPS pico: {storage_rec.iops_read_peak:,} leitura / {storage_rec.iops_write_peak:,} escrita")
        lines.append(f"  - Throughput pico: {storage_rec.throughput_read_peak_gbps:.1f} GB/s leitura / {storage_rec.throughput_write_peak_gbps:.1f} GB/s escrita")

    lines.append(f"- Mantem risco operacional em nivel **aceitavel** para producao")

    # Calibracao na recomendacao
    rec_cal = rec.calibration
    if rec_cal:
        lines.append("")
        lines.append(f"**Calibracao para SLOs:** Para atender {rec_cal.concurrency_requested:,} sessoes com os SLOs configurados, "
                     f"sao necessarios **{rec_cal.nodes_recommended or 'N/A'} nos DGX** (+{rec_cal.extra_nodes_needed} extras). "
                     f"Com os {rec_cal.nodes_current} nos atuais, o maximo atendivel e "
                     f"**{rec_cal.max_concurrency_current_nodes:,} sessoes**.")

    lines.append("")
    lines.append("**Governanca:** Storage e recurso critico. Subdimensionamento impacta:")
    lines.append("- Tempo de recuperacao (restart lento)")
    lines.append("- Escalabilidade (gargalo em scale-out)")
    lines.append("- Auditoria e conformidade (retencao inadequada de logs)")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Secao de Analise de Latencia
    any_latency = any(scenarios[k].latency is not None for k in scenarios)
    if any_latency:
        lines.append("## Analise de Latencia de Inferencia (TTFT e TPOT)")
        lines.append("")

        first_la = next(scenarios[k].latency for k in ["minimum", "recommended", "ideal"] if scenarios[k].latency)

        if first_la.target_ttft_p50_ms or first_la.target_tpot_tokens_per_sec:
            lines.append("**SLO Definido:**")
            if first_la.target_ttft_p50_ms:
                lines.append(f"- TTFT (Time to First Token) P50: **{first_la.target_ttft_p50_ms}ms**")
            if first_la.target_ttft_p99_ms:
                lines.append(f"- TTFT P99: **{first_la.target_ttft_p99_ms}ms**")
            if first_la.target_tpot_tokens_per_sec:
                lines.append(f"- TPOT minimo: **{first_la.target_tpot_tokens_per_sec:.1f} tokens/s**")
            lines.append("")

        lines.append("| Cenario | TTFT P50 | TPOT | Status | Gargalo | Acao Prioritaria |")
        lines.append("|---------|----------|------|--------|---------|-----------------|")
        for key in ["minimum", "recommended", "ideal"]:
            s = scenarios[key]
            la = s.latency
            if la is None:
                continue
            ttft_txt = f"{la.ttft_p50_ms:.0f}ms" if la.ttft_p50_ms < 99000 else "inf"
            ttft_txt += f" ({qual_label_md.get(la.ttft_quality, la.ttft_quality)})"
            tpot_txt = f"{la.tpot_tokens_per_sec:.1f} tok/s ({qual_label_md.get(la.tpot_quality, la.tpot_quality)})"
            status_txt = {'OK': 'Atende SLO', 'SLO_MARGINAL': 'Marginal', 'SLO_VIOLATION': 'Viola SLO'}.get(la.status, la.status)
            bottleneck_short = la.bottleneck.split(' - ')[0] if ' - ' in la.bottleneck else la.bottleneck[:20]
            rec_short = la.recommendation.strip().split('\n')[0].strip().lstrip('1234567890. ') if la.recommendation else 'N/A'
            if len(rec_short) > 60:
                rec_short = rec_short[:57] + '...'
            scenario_name = {'minimum': 'Minimo', 'recommended': 'Recomendado', 'ideal': 'Ideal'}[key]
            lines.append(f"| {scenario_name} | {ttft_txt} | {tpot_txt} | {status_txt} | {bottleneck_short} | {rec_short} |")

        lines.append("")

        rec_la_detail = scenarios['recommended'].latency
        if rec_la_detail:
            total_latency = rec_la_detail.ttft_p50_ms
            net_pct = rec_la_detail.network_latency_p50_ms / total_latency * 100 if total_latency > 0 else 0
            pref_pct = rec_la_detail.prefill_time_ms / total_latency * 100 if total_latency > 0 else 0
            lines.append("**Breakdown de Latencia TTFT (Cenario Recomendado):**")
            lines.append(f"- Network: {rec_la_detail.network_latency_p50_ms:.0f}ms ({net_pct:.1f}%)")
            lines.append(f"- Prefill: {rec_la_detail.prefill_time_ms:.0f}ms ({pref_pct:.1f}%)")
            if rec_la_detail.queuing_delay_p50_ms < 99000:
                q_pct = rec_la_detail.queuing_delay_p50_ms / total_latency * 100 if total_latency > 0 else 0
                lines.append(f"- Queuing: {rec_la_detail.queuing_delay_p50_ms:.0f}ms ({q_pct:.1f}%)")
            else:
                lines.append("- Queuing: inf (sistema saturado)")
            lines.append(f"- TPOT por sessao: {rec_la_detail.tpot_tokens_per_sec:.2f} tok/s (ITL: {rec_la_detail.itl_ms_per_token:.0f}ms/token)")
            lines.append(f"- Utilizacao: {rec_la_detail.utilization*100:.1f}%")
            lines.append("")
            lines.append(f"**Gargalo Principal:** {rec_la_detail.bottleneck}")
            lines.append("")

        lines.append("---")
        lines.append("")

    # Glossario
    lines.append("## Glossario Executivo de Termos")
    lines.append("")
    lines.append("| Metrica | O que significa | Por que importa para a decisao | Impacto se estiver errado |")
    lines.append("| --- | --- | --- | --- |")
    lines.append("| **Nos DGX** | Quantidade de servidores de IA necessarios para atender a carga analisada. | Define investimento em hardware e influencia energia, rack e custo total. | Subdimensionamento causa indisponibilidade; superdimensionamento aumenta custo. |")
    lines.append("| **Sessoes por no (capacidade)** | Numero maximo teorico de conversas simultaneas que um servidor suporta. | Indica o limite fisico do servidor antes de atingir saturacao de memoria. | Operar no limite reduz margem para picos e aumenta risco de instabilidade. |")
    lines.append("| **Sessoes por no (operando)** | Numero real de sessoes em uso no cenario avaliado. | Mostra a folga operacional disponivel. | Se muito proximo do limite, o sistema fica vulneravel a picos de uso. |")
    lines.append("| **KV por sessao** | Memoria de GPU consumida por cada conversa ativa. | E o principal fator que determina quantas sessoes cabem por servidor. | Conversas mais longas aumentam consumo e reduzem capacidade total. |")
    lines.append("| **VRAM total por no** | Memoria total da GPU utilizada pelo modelo, runtime e sessoes. | Indica quao proximo o servidor esta do limite fisico. | Uso excessivo pode causar falhas ou degradacao de performance. |")
    lines.append("| **Energia (Compute + Storage)** | Consumo total de energia dos servidores de IA e do storage. | Impacta custo operacional mensal e capacidade eletrica do datacenter. | Subdimensionar pode causar sobrecarga eletrica; superdimensionar eleva custo. |")
    lines.append("| **Rack (Compute + Storage)** | Espaco fisico ocupado por servidores e storage no datacenter. | Define viabilidade fisica de implantacao e expansao futura. | Espaco insuficiente limita crescimento. |")
    lines.append("| **Storage total** | Capacidade total de armazenamento necessaria para rodar o modelo e sustentar o sistema (modelo + cache + logs). | Representa o espaco minimo necessario para operar o ambiente com seguranca. | Falta de espaco pode impedir inicializacao, gravacao de logs ou escala do sistema. Recomenda-se dimensionar ~50% acima do minimo calculado. |")
    lines.append("| **Storage (modelo)** | Espaco necessario para armazenar os arquivos do modelo (pesos e artefatos). | Essencial para subir o sistema e permitir reinicializacoes rapidas. | Se insuficiente, o sistema pode nao iniciar corretamente. Recomenda-se margem adicional. |")
    lines.append("| **Storage (cache)** | Espaco para arquivos temporarios e dados intermediarios usados na execucao. | Garante funcionamento continuo e estavel do ambiente. | Pode gerar falhas ou degradacao se o espaco se esgotar. |")
    lines.append("| **Storage (logs)** | Espaco destinado ao armazenamento de logs operacionais e auditoria. | Fundamental para rastreabilidade, analise de incidentes e governanca. | Falta de espaco compromete auditoria e diagnostico de problemas. |")
    lines.append("| **IOPS (pico R/W)** | Numero maximo de operacoes de leitura e escrita por segundo no pico. | Determina se o storage suporta eventos como subida simultanea de multiplos servidores. | Gargalo de IOPS aumenta tempo de recuperacao e escala. |")
    lines.append("| **Throughput (pico R/W)** | Volume maximo de dados transferidos por segundo no pico de uso. | Afeta tempo de carregamento do modelo e recuperacao apos falhas. | Throughput insuficiente aumenta tempo de indisponibilidade. |")
    lines.append("| **Arquitetura HA** | Nivel de tolerancia a falhas adotado (ex.: NONE, N+1, N+2). | Define o quanto o sistema continua operando mesmo apos falhas de hardware. | Ausencia de HA pode causar interrupcao total do servico. |")
    lines.append("| **TTFT** | Tempo ate o primeiro token ser retornado ao usuario (inclui rede, fila e prefill). | Latencia percebida pelo usuario -- define se o sistema parece responsivo. | TTFT alto faz o usuario perceber demora antes de qualquer resposta aparecer. |")
    lines.append("| **TPOT/ITL** | Velocidade de geracao de tokens (tokens/s) ou intervalo entre tokens (ms/token). | Determina a fluidez do streaming -- quantas palavras por minuto o usuario ve. | TPOT baixo torna o streaming lento e perceptivelmente truncado. |")
    lines.append("| **Concorrencia max (SLOs)** | Numero maximo de sessoes simultaneas que a infra pode atender dentro dos SLOs de latencia. | Indica o limite real de uso mantendo qualidade de servico. | Exceder este limite causa TTFT infinito (filas) e TPOT inaceitavel. |")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("*Relatorio gerado automaticamente pelo Calculadora de Sizing de Infraestrutura para Inferencia, desenvolvido pelo time de InfraCore de CLOUD.*")
    lines.append("")

    return "\n".join(lines)
