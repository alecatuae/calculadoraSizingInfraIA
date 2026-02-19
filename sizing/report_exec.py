"""
Geração de relatório executivo (resumo para terminal e relatório Markdown executivo).
"""

from typing import Dict, Optional
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
    sizing_mode: str = "concurrency_driven",
    ttft_input_ms: Optional[int] = None,
    tpot_input_ms: Optional[float] = None,
    concurrency_input: Optional[int] = None,
) -> str:
    """
    Gera resumo executivo para exibição no terminal.

    Seção 8 — Saída no Terminal:
      - Modo usado
      - Entrada e resultado de concurrency/ttft/tpot
      - Número de servidores por cenário
      - Caminhos dos relatórios
    """
    lines = []

    lines.append("=" * 80)
    lines.append("RESUMO — SIZING DE INFRAESTRUTURA PARA INFERENCIA")
    lines.append("=" * 80)
    lines.append("")

    # Modo e entradas
    if sizing_mode == "slo_driven":
        lines.append(f"Modo:                MODO B — Sizing por SLO")
        lines.append(f"Modelo:              {model_name}")
        lines.append(f"Servidor Inferencia: {server_name}")
        lines.append(f"Contexto Efetivo:    {effective_context:,} tokens")
        lines.append(f"Precisao KV Cache:   {kv_precision.upper()}")
        lines.append("")
        lines.append(f"Entradas SLO:")
        lines.append(f"  TTFT alvo:         {ttft_input_ms} ms")
        lines.append(f"  TPOT alvo:         {tpot_input_ms} tok/s")
    else:
        lines.append(f"Modo:                MODO A — Sizing por Concorrencia")
        lines.append(f"Modelo:              {model_name}")
        lines.append(f"Servidor Inferencia: {server_name}")
        lines.append(f"Contexto Efetivo:    {effective_context:,} tokens")
        lines.append(f"Concorrencia:        {concurrency_input:,} sessoes simultaneas")
        lines.append(f"Precisao KV Cache:   {kv_precision.upper()}")

    lines.append("")

    # Tabela de resultados por cenário
    qual_pt = {'excellent': 'Excelente', 'good': 'Bom', 'acceptable': 'Aceitavel', 'slow': 'Lento', None: 'N/A'}
    slo_map = {'OK': '[OK]', 'SLO_MARGINAL': '[MARGINAL]', 'SLO_VIOLATION': '[VIOLADO]', 'NO_SLO': ''}

    if sizing_mode == "slo_driven":
        # Modo B: mostrar concorrência final calculada
        lines.append("-" * 110)
        header = (f"{'Cenario':<16} {'Servidores':<12} {'Concorr. Final':<16} {'TTFT Final':<22} "
                  f"{'TPOT Final':<22} {'kW':<8} {'Rack':<8}")
        lines.append(header)
        lines.append("-" * 110)

        for key in ["minimum", "recommended", "ideal"]:
            s = scenarios[key]
            la = s.latency
            sc = s.slo_capacity

            conc_final = sc.max_concurrency_combined if sc and sc.is_feasible else 0

            if la:
                ttft_str = (f"{la.ttft_p50_ms:.0f}ms"
                            if la.ttft_p50_ms < 99000 else "inf (saturado)")
                ttft_qual = qual_pt.get(la.ttft_quality, la.ttft_quality)
                tpot_str = f"{la.tpot_tokens_per_sec:.1f} tok/s ({qual_pt.get(la.tpot_quality, la.tpot_quality)})"
                ttft_display = f"{ttft_str} ({ttft_qual})"
            else:
                ttft_display = tpot_str = "N/A"

            row = (f"{s.config.name:<16} {s.nodes_final:<12} {conc_final:<16} "
                   f"{ttft_display:<22} {tpot_str:<22} "
                   f"{s.total_power_kw_with_storage:<8.1f} {s.total_rack_u_with_storage:<8}")
            lines.append(row)

        lines.append("-" * 110)

    else:
        # Modo A: mostrar TTFT/TPOT estimados
        lines.append("-" * 110)
        header = (f"{'Cenario':<16} {'Servidores':<12} {'TTFT Estimado':<22} "
                  f"{'TPOT Estimado':<22} {'kW':<8} {'Rack':<8} {'Storage (TB)':<14}")
        lines.append(header)
        lines.append("-" * 110)

        for key in ["minimum", "recommended", "ideal"]:
            s = scenarios[key]
            la = s.latency
            storage_display = (
                f"{s.storage.storage_total_recommended_tb:.1f} (+{s.storage.margin_percent*100:.0f}%)"
                if s.storage else "N/A"
            )

            if la:
                ttft_str = (f"{la.ttft_p50_ms:.0f}ms"
                            if la.ttft_p50_ms < 99000 else "inf")
                ttft_display = f"{ttft_str} ({qual_pt.get(la.ttft_quality, la.ttft_quality)})"
                tpot_display = f"{la.tpot_tokens_per_sec:.1f} tok/s ({qual_pt.get(la.tpot_quality, la.tpot_quality)})"
            else:
                ttft_display = tpot_display = "N/A"

            row = (f"{s.config.name:<16} {s.nodes_final:<12} {ttft_display:<22} "
                   f"{tpot_display:<22} {s.total_power_kw_with_storage:<8.1f} "
                   f"{s.total_rack_u_with_storage:<8} {storage_display:<14}")
            lines.append(row)

        lines.append("-" * 110)

    lines.append("")

    # Nota sobre margem
    if scenarios["recommended"].storage and scenarios["recommended"].storage.margin_applied:
        margin_pct = scenarios["recommended"].storage.margin_percent * 100
        platform_per_server_tb = scenarios["recommended"].storage.platform_per_server_tb
        lines.append(f"[INFO] Storage considera margem adicional de {margin_pct:.0f}% sobre o volume base.")
        lines.append(f"[INFO] Storage inclui volume estrutural da plataforma ({platform_per_server_tb:.2f} TB/servidor).")
        lines.append("")

    # Cenário recomendado
    rec = scenarios["recommended"]
    rec_la = rec.latency
    rec_sc = rec.slo_capacity

    def _fmt_ttft(la_obj):
        if la_obj is None:
            return "N/A"
        return f"{la_obj.ttft_p50_ms:.0f}ms" if la_obj.ttft_p50_ms < 99000 else "inf (saturado)"

    if sizing_mode == "slo_driven" and rec_sc:
        conc_final = rec_sc.max_concurrency_combined
        ttft_ms = _fmt_ttft(rec_la)
        tpot_val = f"{rec_la.tpot_tokens_per_sec:.1f} tok/s" if rec_la else "N/A"
        lines.append(
            f"Cenario RECOMENDADO: {rec.nodes_final} servidor(es) de inferencia | "
            f"{conc_final:,} sessoes dentro dos SLOs | "
            f"TTFT: {ttft_ms} | TPOT: {tpot_val} | "
            f"{rec.total_power_kw_with_storage:.1f} kW | {rec.total_rack_u_with_storage}U"
        )
    else:
        storage_info = f" | {rec.storage.storage_total_recommended_tb:.1f} TB storage" if rec.storage else ""
        if rec_la:
            ttft_ms = _fmt_ttft(rec_la)
            lat_info = f" | TTFT: {ttft_ms} | TPOT: {rec_la.tpot_tokens_per_sec:.1f} tok/s"
        else:
            lat_info = ""
        lines.append(
            f"Cenario RECOMENDADO: {rec.nodes_final} servidor(es) de inferencia | "
            f"{concurrency_input:,} sessoes | "
            f"{rec.total_power_kw_with_storage:.1f} kW | {rec.total_rack_u_with_storage}U"
            f"{storage_info}{lat_info}"
        )

    lines.append(f"Tolerancia a Falhas: {rec.config.ha_mode.upper()}")
    lines.append("")
    lines.append("=" * 80)
    lines.append("Relatorios salvos em:")
    lines.append(f"   Texto:  {text_report_path}")
    lines.append(f"   JSON:   {json_report_path}")
    lines.append("")

    return "\n".join(lines)


def _slo_demand_table(
    sizing_mode: str,
    ttft_input_ms: Optional[int],
    tpot_input_ms: Optional[float],
    concurrency_input: Optional[int],
    scenarios: Dict[str, ScenarioResult],
    scenario_key: str = "recommended"
) -> list:
    """
    Gera a seção 'Parâmetros de Demanda e SLO' como lista de linhas Markdown.

    Para Modo A: entrada=concorrência, resultado=concorrência+TTFT/TPOT estimados.
    Para Modo B: entrada=TTFT+TPOT, resultado=concorrência_final+TTFT/TPOT finais.
    """
    lines = []
    lines.append("## Parâmetros de Demanda e SLO")
    lines.append("")

    # Descrições didáticas
    lines.append("**Concorrência** — Número de requisições/sessões simultâneas atendidas. "
                 "Determina a capacidade operacional e o custo da infraestrutura.")
    lines.append("")
    lines.append("**TTFT (Time To First Token)** — Tempo até o primeiro token ser entregue ao usuário "
                 "(inclui rede, fila e processamento do prompt). "
                 "Define a percepção de responsividade: TTFT alto faz o sistema parecer lento.")
    lines.append("")
    lines.append("**TPOT (Time Per Output Token)** — Velocidade de geração contínua de tokens (tokens/segundo). "
                 "Define a fluidez do streaming: TPOT baixo torna a leitura truncada e lenta.")
    lines.append("")

    s = scenarios[scenario_key]
    la = s.latency
    sc = s.slo_capacity

    def _ttft_display(la_obj):
        if la_obj is None:
            return "N/A"
        if la_obj.ttft_p50_ms >= 99000:
            return "inf (saturado)"
        return f"{la_obj.ttft_p50_ms:.0f} ms"

    if sizing_mode == "slo_driven":
        conc_entrada = "—"
        conc_resultado = str(sc.max_concurrency_combined) + " sessões" if sc and sc.is_feasible else "N/A"
        ttft_entrada = f"{ttft_input_ms} ms"
        tpot_entrada = f"{tpot_input_ms} tok/s"
        ttft_resultado = _ttft_display(la)
        tpot_resultado = f"{la.tpot_tokens_per_sec:.1f} tok/s" if la else "N/A"

        conc_obs = "Calculada pela capacidade da infra dentro dos SLOs"
        ttft_obs = "Tempo total ao primeiro token (rede + fila + prefill)"
        tpot_obs = "Velocidade de geração por sessão ativa"
    else:
        conc_entrada = f"{concurrency_input:,} sessões"
        conc_resultado = f"{concurrency_input:,} sessões"
        ttft_entrada = "—"
        tpot_entrada = "—"
        ttft_resultado = _ttft_display(la)
        tpot_resultado = f"{la.tpot_tokens_per_sec:.1f} tok/s" if la else "N/A"

        conc_obs = "Dimensionamento calculado para esta concorrência"
        ttft_obs = "Estimativa para o cenário recomendado"
        tpot_obs = "Estimativa para o cenário recomendado"

    lines.append("| Parâmetro | Entrada | Resultado (Cenário Recomendado) | Observação Operacional |")
    lines.append("|-----------|---------|----------------------------------|------------------------|")
    lines.append(f"| **Concorrência** | {conc_entrada} | {conc_resultado} | {conc_obs} |")
    lines.append(f"| **TTFT P50** | {ttft_entrada} | {ttft_resultado} | {ttft_obs} |")
    lines.append(f"| **TPOT** | {tpot_entrada} | {tpot_resultado} | {tpot_obs} |")
    lines.append("")

    return lines


def format_executive_markdown(
    model: ModelSpec,
    server: ServerSpec,
    scenarios: Dict[str, ScenarioResult],
    concurrency: int,
    effective_context: int,
    kv_precision: str,
    storage_name: str = "N/A",
    sizing_mode: str = "concurrency_driven",
    ttft_input_ms: Optional[int] = None,
    tpot_input_ms: Optional[float] = None,
    concurrency_input: Optional[int] = None,
) -> str:
    """
    Gera relatório executivo completo em Markdown.
    """
    lines = []

    lines.append("# Relatorio Executivo — Sizing de Infraestrutura para Inferencia")
    lines.append("")
    lines.append(f"**Modelo:** {model.name}  ")
    lines.append(f"**Servidor de Inferencia:** {server.name}  ")
    lines.append(f"**Data:** {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ")
    if sizing_mode == "slo_driven":
        modo_label = f"MODO B — Sizing por SLO (TTFT={ttft_input_ms}ms / TPOT={tpot_input_ms} tok/s)"
    else:
        modo_label = f"MODO A — Sizing por Concorrencia ({concurrency_input:,} sessoes)"
    lines.append(f"**Modo de Sizing:** {modo_label}  ")
    lines.append("")
    lines.append("---")
    lines.append("")

    # ── Parâmetros de Demanda e SLO ──────────────────────────────────────────
    lines.extend(_slo_demand_table(
        sizing_mode=sizing_mode,
        ttft_input_ms=ttft_input_ms,
        tpot_input_ms=tpot_input_ms,
        concurrency_input=concurrency_input,
        scenarios=scenarios,
        scenario_key="recommended"
    ))
    lines.append("---")
    lines.append("")

    # ── Sumário Executivo ─────────────────────────────────────────────────────
    lines.append("## Sumario Executivo")
    lines.append("")

    rec = scenarios["recommended"]
    storage_rec = rec.storage if rec.storage else None
    rec_la = rec.latency
    rec_sc = rec.slo_capacity

    if sizing_mode == "slo_driven":
        conc_final = rec_sc.max_concurrency_combined if rec_sc else 0
        lines.append(
            f"O dimensionamento e guiado pelos **SLOs de latencia** definidos "
            f"(TTFT ≤ {ttft_input_ms}ms / TPOT ≥ {tpot_input_ms} tok/s). "
            f"A infraestrutura e calculada para maximizar a **concorrencia atendivel** dentro dessas metas."
        )
        lines.append("")
        lines.append(
            f"O cenario recomendado suporta **{conc_final:,} sessoes simultaneas** "
            f"com {rec.nodes_final} servidor(es) de inferencia {server.name}."
        )
    else:
        lines.append(
            f"Para sustentar **{concurrency_input:,} sessoes simultaneas** com contexto de "
            f"**{effective_context:,} tokens** utilizando o modelo **{model.name}**, "
            f"a infraestrutura e dimensionada por **memoria GPU (KV cache)** e **storage**."
        )
    lines.append("")

    lines.append(
        f"O principal limitador de capacidade e o consumo de HBM para armazenar o estado de atencao "
        f"(KV cache) de cada sessao ativa. Storage e dimensionado para operacao continua (pesos do modelo, "
        f"cache de runtime, logs e auditoria), garantindo resiliencia, tempo de recuperacao e governanca operacional."
    )
    lines.append("")

    if rec_la:
        ttft_display = f"{rec_la.ttft_p50_ms:.0f}ms" if rec_la.ttft_p50_ms < 99000 else "inf"
        ttft_qual = {'excellent': 'excelente', 'good': 'bom', 'acceptable': 'aceitavel', 'slow': 'lento'}.get(
            rec_la.ttft_quality, rec_la.ttft_quality)
        tpot_qual = {'excellent': 'excelente', 'good': 'bom', 'acceptable': 'aceitavel', 'slow': 'lento'}.get(
            rec_la.tpot_quality, rec_la.tpot_quality)
        slo_text = {
            'OK': 'SLOs de latencia atendidos com margem.',
            'SLO_MARGINAL': 'SLOs de latencia atendidos com margem minima.',
            'NO_SLO': 'Latencias estimadas (sem SLO definido).'
        }.get(rec_la.status, '')
        lines.append(
            f"O cenario recomendado apresenta **TTFT de {ttft_display}** (qualidade: {ttft_qual}) e "
            f"**TPOT de {rec_la.tpot_tokens_per_sec:.2f} tok/s** (qualidade: {tpot_qual}). {slo_text}"
        )
        lines.append("")

    lines.append(
        f"**Recomendacao:** {rec.nodes_final} servidor(es) de inferencia {server.name} "
    )
    if storage_rec:
        lines.append(
            f"({rec.total_power_kw:.1f} kW, {rec.total_rack_u}U rack, "
            f"{storage_rec.storage_total_recommended_tb:.1f} TB storage) "
        )
    else:
        lines.append(f"({rec.total_power_kw:.1f} kW, {rec.total_rack_u}U rack) ")
    lines.append(f"com tolerancia a falhas {rec.config.ha_mode.upper()}.")
    lines.append("")
    lines.append("---")
    lines.append("")

    # ── Cenários Avaliados ────────────────────────────────────────────────────
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

    # ── Informações do Modelo ─────────────────────────────────────────────────
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
    lines.append("O modelo consome memoria viva (KV cache) proporcional ao contexto e concorrencia.")
    lines.append("")
    lines.append("---")
    lines.append("")

    # ── Consumo Unitário ──────────────────────────────────────────────────────
    lines.append("## Consumo Unitario do Modelo")
    lines.append("")
    lines.append("| Recurso | Consumo por Sessao | Significado Operacional |")
    lines.append("|---------|-------------------|------------------------|")
    lines.append(f"| KV cache | {rec.vram.vram_per_session_gib:.2f} GiB | Memoria ocupada enquanto sessao esta ativa |")
    lines.append(
        f"| GPU HBM | {(rec.vram.vram_per_session_gib/rec.vram.hbm_total_gib*100):.1f}% de um no | "
        f"Fracao da capacidade GPU consumida |"
    )
    lines.append("")
    lines.append("Cada sessao ativa reserva parte do servidor. A soma das reservas define o limite fisico do no.")
    lines.append("")
    lines.append("---")
    lines.append("")

    # ── Resultados por Cenário ────────────────────────────────────────────────
    lines.append("## Resultados por Cenario")
    lines.append("")

    qual_label_md = {'excellent': 'Excelente', 'good': 'Bom', 'acceptable': 'Aceitavel', 'slow': 'Lento'}
    slo_label_md = {'OK': '[OK] Atende', 'SLO_MARGINAL': '[MARGINAL]', 'NO_SLO': '[Estimativa]'}

    for key in ["minimum", "recommended", "ideal"]:
        s = scenarios[key]
        lines.append(f"### Cenario {s.config.name}")
        lines.append("")
        lines.append("| Metrica | Valor |")
        lines.append("|---------|-------|")
        lines.append(f"| Servidores de Inferencia | {s.nodes_final} |")
        lines.append(f"| Sessoes por servidor (capacidade) | {s.vram.sessions_per_node} |")
        lines.append(f"| Sessoes por servidor (operando) | {s.sessions_per_node_effective} |")
        lines.append(f"| KV por sessao | {s.vram.vram_per_session_gib:.2f} GiB |")
        lines.append(
            f"| VRAM total por servidor | "
            f"{s.vram_total_node_effective_gib:.1f} GiB ({s.hbm_utilization_ratio_effective*100:.1f}% HBM) |"
        )
        lines.append(
            f"| **Energia (Compute + Storage)** | "
            f"**{s.total_power_kw_with_storage:.1f} kW** ({s.total_power_kw:.1f} + {s.storage_power_kw:.1f}) |"
        )
        lines.append(
            f"| **Rack (Compute + Storage)** | "
            f"**{s.total_rack_u_with_storage}U** ({s.total_rack_u} + {s.storage_rack_u}) |"
        )

        if s.storage:
            st = s.storage
            lines.append(f"| **Storage total** | **{st.storage_total_recommended_tb:.2f} TB** |")
            lines.append(f"| Storage (modelo) | {st.storage_model_recommended_tb:.2f} TB |")
            lines.append(f"| Storage (cache) | {st.storage_cache_recommended_tb:.2f} TB |")
            lines.append(f"| Storage (logs) | {st.storage_logs_recommended_tb:.2f} TB |")
            lines.append(f"| IOPS (pico R/W) | {st.iops_read_peak:,} / {st.iops_write_peak:,} |")
            lines.append(
                f"| Throughput (pico R/W) | "
                f"{st.throughput_read_peak_gbps:.1f} / {st.throughput_write_peak_gbps:.1f} GB/s |"
            )

        lines.append(f"| Arquitetura HA | {s.config.ha_mode.upper()} |")

        # TTFT/TPOT
        if s.latency:
            la = s.latency
            ttft_val = f"{la.ttft_p50_ms:.0f} ms" if la.ttft_p50_ms < 99000 else "inf (saturado)"
            ttft_p99_val = f"{la.ttft_p99_ms:.0f} ms" if la.ttft_p99_ms < 99000 else "inf (saturado)"
            tpot_val = (
                f"{la.tpot_tokens_per_sec:.2f} tok/s (ITL: {la.itl_ms_per_token:.0f} ms/token)"
                if la.itl_ms_per_token < 99000 else f"{la.tpot_tokens_per_sec:.2f} tok/s"
            )
            util_val = f"{la.utilization * 100:.1f}%"
            ttft_qual = qual_label_md.get(la.ttft_quality, la.ttft_quality)
            tpot_qual = qual_label_md.get(la.tpot_quality, la.tpot_quality)
            slo_val = slo_label_md.get(la.status, la.status)
            lines.append(f"| **TTFT P50 (latencia 1o token)** | **{ttft_val}** — {ttft_qual} |")
            lines.append(f"| TTFT P99 | {ttft_p99_val} |")
            lines.append(f"| **TPOT (velocidade streaming)** | **{tpot_val}** — {tpot_qual} |")
            lines.append(f"| Utilizacao GPU (queuing) | {util_val} |")
            lines.append(f"| Gargalo | {la.bottleneck.split(' - ')[0]} |")
            if la.status != 'NO_SLO':
                lines.append(f"| **Status SLO Latencia** | **{slo_val}** |")

        # Capacidade máxima por SLO (Modo B)
        if s.slo_capacity and sizing_mode == "slo_driven":
            sc = s.slo_capacity
            if sc.is_feasible:
                lines.append(f"| **Concorrencia maxima (SLOs)** | **{sc.max_concurrency_combined:,} sessoes** |")
                lines.append(
                    f"| Max por TTFT | "
                    f"{sc.max_concurrency_from_ttft:,} sessoes (util. max: {sc.util_max_from_ttft*100:.1f}%) |"
                )
                lines.append(
                    f"| Max por TPOT | "
                    f"{sc.max_concurrency_from_tpot:,} sessoes (sess./servidor max: {sc.sessions_per_node_max_from_tpot}) |"
                )
                lines.append(f"| Fator limitante | {sc.limiting_factor} |")

        lines.append("")

        if key == "minimum":
            lines.append(
                f"**Analise Computacional:** Opera no limite da capacidade sem margem para picos ou falhas. "
                f"Risco operacional **alto** — qualquer indisponibilidade de hardware afeta o servico diretamente."
            )
            if s.storage:
                lines.append(
                    f"**Analise Storage:** {s.storage.storage_total_recommended_tb:.1f} TB recomendado "
                    f"(base: {s.storage.storage_total_base_tb:.1f} TB). "
                    f"IOPS e throughput dimensionados sem margem."
                )
        elif key == "recommended":
            lines.append(
                f"**Analise Computacional:** Equilibra eficiencia e resiliencia. "
                f"Suporta picos de ate {s.config.peak_headroom_ratio*100:.0f}% "
                f"e tolera falha de 1 servidor sem degradacao. **Adequado para producao.**"
            )
            if s.storage:
                lines.append(
                    f"**Analise Storage:** {s.storage.storage_total_recommended_tb:.1f} TB recomendado "
                    f"(base: {s.storage.storage_total_base_tb:.1f} TB) com margem de capacidade. "
                    f"Tempo de recuperacao aceitavel."
                )
        else:
            lines.append(
                f"**Analise Computacional:** Maxima resiliencia com margem para multiplas falhas. "
                f"Risco operacional **minimo**. Ideal para servicos criticos."
            )
            if s.storage:
                lines.append(
                    f"**Analise Storage:** {s.storage.storage_total_recommended_tb:.1f} TB recomendado "
                    f"(base: {s.storage.storage_total_base_tb:.1f} TB) com margem ampla. "
                    f"Maxima resiliencia."
                )
        lines.append("")

    lines.append("---")
    lines.append("")

    # ── Comparação Executiva ──────────────────────────────────────────────────
    lines.append("## Comparacao Executiva dos Cenarios")
    lines.append("")
    lines.append("| Criterio | Minimo | Recomendado | Ideal |")
    lines.append("|----------|--------|-------------|-------|")
    lines.append(
        f"| Servidores de Inferencia | "
        f"{scenarios['minimum'].nodes_final} | "
        f"{scenarios['recommended'].nodes_final} | "
        f"{scenarios['ideal'].nodes_final} |"
    )
    lines.append(
        f"| Energia Total (kW) | "
        f"{scenarios['minimum'].total_power_kw_with_storage:.1f} | "
        f"{scenarios['recommended'].total_power_kw_with_storage:.1f} | "
        f"{scenarios['ideal'].total_power_kw_with_storage:.1f} |"
    )
    lines.append(
        f"| Rack Total (U) | "
        f"{scenarios['minimum'].total_rack_u_with_storage} | "
        f"{scenarios['recommended'].total_rack_u_with_storage} | "
        f"{scenarios['ideal'].total_rack_u_with_storage} |"
    )

    if all(scenarios[k].storage for k in ['minimum', 'recommended', 'ideal']):
        st_min = scenarios['minimum'].storage
        st_rec = scenarios['recommended'].storage
        st_ideal = scenarios['ideal'].storage
        lines.append(
            f"| Storage (TB) | "
            f"{st_min.storage_total_recommended_tb:.1f} | "
            f"{st_rec.storage_total_recommended_tb:.1f} | "
            f"{st_ideal.storage_total_recommended_tb:.1f} |"
        )
        lines.append(
            f"| IOPS pico (R) | "
            f"{st_min.iops_read_peak:,} | "
            f"{st_rec.iops_read_peak:,} | "
            f"{st_ideal.iops_read_peak:,} |"
        )

    lines.append(f"| Tolerancia a falhas | Nenhuma | 1 servidor | 2 servidores |")
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

        lines.append(
            f"| **TTFT P50** | {_ttft_str('minimum')} | {_ttft_str('recommended')} | {_ttft_str('ideal')} |"
        )
        lines.append(
            f"| **TPOT** | {_tpot_str('minimum')} | {_tpot_str('recommended')} | {_tpot_str('ideal')} |"
        )

    if sizing_mode == "slo_driven":
        def _slo_cap_str(k):
            sc = scenarios[k].slo_capacity
            if sc is None:
                return "N/A"
            if not sc.is_feasible:
                return "INVIAVEL"
            return f"{sc.max_concurrency_combined:,} sessoes"

        lines.append(
            f"| **Concorrencia maxima (SLOs)** | "
            f"{_slo_cap_str('minimum')} | "
            f"{_slo_cap_str('recommended')} | "
            f"{_slo_cap_str('ideal')} |"
        )

    lines.append("")
    lines.append(
        f"**Conclusao:** O cenario **RECOMENDADO** oferece o melhor equilibrio custo-risco para operacao em producao."
    )
    if scenarios['recommended'].storage:
        lines.append(
            "Storage subdimensionado compromete resiliencia e tempo de recuperacao, mesmo com GPUs suficientes."
        )
    lines.append("")
    lines.append("---")
    lines.append("")

    # ── Recomendação Final ────────────────────────────────────────────────────
    lines.append("## Recomendacao Final")
    lines.append("")
    lines.append(
        f"Recomenda-se o **cenario RECOMENDADO** com "
        f"**{rec.nodes_final} servidor(es) de inferencia {server.name}**, que:"
    )
    lines.append("")
    if sizing_mode == "slo_driven" and rec_sc:
        lines.append(f"- Atende os SLOs de latencia (TTFT ≤ {ttft_input_ms}ms / TPOT ≥ {tpot_input_ms} tok/s)")
        lines.append(f"- Suporta **{rec_sc.max_concurrency_combined:,} sessoes simultaneas** dentro dos SLOs")
    else:
        lines.append(f"- Atende os requisitos de capacidade ({concurrency_input:,} sessoes)")
    lines.append(f"- Suporta picos de ate {rec.config.peak_headroom_ratio*100:.0f}%")
    lines.append(f"- Tolera falha de 1 servidor sem degradacao ({rec.config.ha_mode.upper()})")
    lines.append(f"- Consome {rec.total_power_kw:.1f} kW e ocupa {rec.total_rack_u}U de rack")

    if storage_rec:
        lines.append(
            f"- Requer {storage_rec.storage_total_recommended_tb:.1f} TB de storage "
            f"({storage_name}, incluindo margem de capacidade)"
        )
        lines.append(
            f"  - IOPS pico: {storage_rec.iops_read_peak:,} leitura / {storage_rec.iops_write_peak:,} escrita"
        )
        lines.append(
            f"  - Throughput pico: {storage_rec.throughput_read_peak_gbps:.1f} GB/s leitura / "
            f"{storage_rec.throughput_write_peak_gbps:.1f} GB/s escrita"
        )

    lines.append(f"- Mantem risco operacional em nivel **aceitavel** para producao")
    lines.append("")

    lines.append("**Governanca:** Storage e recurso critico. Subdimensionamento impacta:")
    lines.append("- Tempo de recuperacao (restart lento)")
    lines.append("- Escalabilidade (gargalo em scale-out)")
    lines.append("- Auditoria e conformidade (retencao inadequada de logs)")
    lines.append("")
    lines.append("---")
    lines.append("")

    # ── Análise de Latência ───────────────────────────────────────────────────
    any_latency = any(scenarios[k].latency is not None for k in scenarios)
    if any_latency:
        lines.append("## Analise de Latencia de Inferencia (TTFT e TPOT)")
        lines.append("")

        first_la = next(
            (scenarios[k].latency for k in ["minimum", "recommended", "ideal"] if scenarios[k].latency),
            None
        )

        if first_la and sizing_mode == "slo_driven":
            lines.append("**SLO Definido:**")
            lines.append(f"- TTFT (Time to First Token) P50: **{ttft_input_ms}ms**")
            lines.append(f"- TPOT minimo: **{tpot_input_ms:.1f} tokens/s**")
            lines.append("")

        lines.append("| Cenario | Servidores | Conc. Final | TTFT P50 | TPOT | Gargalo |")
        lines.append("|---------|------------|-------------|----------|------|---------|")
        for key in ["minimum", "recommended", "ideal"]:
            s = scenarios[key]
            la = s.latency
            sc = s.slo_capacity
            if la is None:
                continue
            ttft_txt = f"{la.ttft_p50_ms:.0f}ms" if la.ttft_p50_ms < 99000 else "inf"
            ttft_txt += f" ({qual_label_md.get(la.ttft_quality, la.ttft_quality)})"
            tpot_txt = f"{la.tpot_tokens_per_sec:.1f} tok/s ({qual_label_md.get(la.tpot_quality, la.tpot_quality)})"
            conc_txt = str(sc.max_concurrency_combined) if (sc and sizing_mode == "slo_driven") else (
                str(concurrency_input) if concurrency_input else "N/A"
            )
            bottleneck_short = la.bottleneck.split(' - ')[0] if ' - ' in la.bottleneck else la.bottleneck[:30]
            scenario_name = {'minimum': 'Minimo', 'recommended': 'Recomendado', 'ideal': 'Ideal'}[key]
            lines.append(
                f"| {scenario_name} | {s.nodes_final} | {conc_txt} | {ttft_txt} | {tpot_txt} | {bottleneck_short} |"
            )

        lines.append("")

        rec_la_detail = scenarios['recommended'].latency
        if rec_la_detail:
            total_latency = rec_la_detail.ttft_p50_ms
            lines.append("**Breakdown de Latencia TTFT (Cenario Recomendado):**")
            if total_latency > 0 and total_latency < 99000:
                net_pct = rec_la_detail.network_latency_p50_ms / total_latency * 100
                pref_pct = rec_la_detail.prefill_time_ms / total_latency * 100
                lines.append(f"- Network: {rec_la_detail.network_latency_p50_ms:.0f}ms ({net_pct:.1f}%)")
                lines.append(f"- Prefill: {rec_la_detail.prefill_time_ms:.0f}ms ({pref_pct:.1f}%)")
                if rec_la_detail.queuing_delay_p50_ms < 99000:
                    q_pct = rec_la_detail.queuing_delay_p50_ms / total_latency * 100
                    lines.append(
                        f"- Queuing: {rec_la_detail.queuing_delay_p50_ms:.0f}ms ({q_pct:.1f}%)"
                    )
                else:
                    lines.append("- Queuing: inf (sistema saturado)")
            lines.append(
                f"- TPOT por sessao: {rec_la_detail.tpot_tokens_per_sec:.2f} tok/s "
                f"(ITL: {rec_la_detail.itl_ms_per_token:.0f}ms/token)"
            )
            lines.append(f"- Utilizacao: {rec_la_detail.utilization*100:.1f}%")
            lines.append("")
            lines.append(f"**Gargalo Principal:** {rec_la_detail.bottleneck}")
            lines.append("")

        lines.append("---")
        lines.append("")

    # ── Glossário ─────────────────────────────────────────────────────────────
    lines.append("## Glossario Executivo de Termos")
    lines.append("")
    lines.append(
        "| Metrica | O que significa | Por que importa | Impacto se subdimensionado |"
    )
    lines.append("| --- | --- | --- | --- |")
    lines.append(
        "| **Servidores de Inferencia** | Quantidade de servidores necessarios para atender a carga. | "
        "Define investimento em hardware e influencia energia, rack e custo total. | "
        "Subdimensionamento causa indisponibilidade. |"
    )
    lines.append(
        "| **Sessoes por servidor** | Numero de conversas simultaneas que um servidor suporta. | "
        "Indica o limite fisico antes de atingir saturacao de memoria. | "
        "Operar no limite aumenta risco de instabilidade. |"
    )
    lines.append(
        "| **KV por sessao** | Memoria de GPU consumida por cada conversa ativa. | "
        "Principal fator que determina quantas sessoes cabem por servidor. | "
        "Conversas mais longas aumentam consumo e reduzem capacidade. |"
    )
    lines.append(
        "| **Energia (Compute + Storage)** | Consumo total de energia dos servidores e storage. | "
        "Impacta custo operacional mensal e capacidade eletrica do datacenter. | "
        "Subdimensionar pode causar sobrecarga eletrica. |"
    )
    lines.append(
        "| **Rack (Compute + Storage)** | Espaco fisico no datacenter. | "
        "Define viabilidade fisica de implantacao. | "
        "Espaco insuficiente limita crescimento. |"
    )
    lines.append(
        "| **Storage total** | Capacidade necessaria para modelo + cache + logs. | "
        "Espaco minimo para operar o ambiente. | "
        "Falta de espaco pode impedir inicializacao ou escala. |"
    )
    lines.append(
        "| **TTFT** | Tempo ate o primeiro token (rede + fila + prefill). | "
        "Latencia percebida pelo usuario — define se o sistema parece responsivo. | "
        "TTFT alto faz o usuario perceber demora antes de qualquer resposta. |"
    )
    lines.append(
        "| **TPOT** | Velocidade de geracao de tokens (tokens/segundo). | "
        "Determina a fluidez do streaming. | "
        "TPOT baixo torna o streaming lento e perceptivelmente truncado. |"
    )
    lines.append(
        "| **Concorrencia maxima (SLOs)** | Sessoes simultaneas atendidas dentro dos SLOs de latencia. | "
        "Indica o limite real de uso mantendo qualidade de servico. | "
        "Exceder este limite causa TTFT infinito (filas) e TPOT inaceitavel. |"
    )
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append(
        "*Relatorio gerado automaticamente pela Calculadora de Sizing de Infraestrutura para Inferencia.*"
    )
    lines.append("")

    return "\n".join(lines)
