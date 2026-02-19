"""
Geração de relatório completo (técnico detalhado).
"""

from typing import Dict, Any, List, Optional
from .calc_scenarios import ScenarioResult
from .models import ModelSpec
from .servers import ServerSpec
from .storage import StorageProfile
from .calc_response_time import LatencyAnalysis, latency_analysis_to_dict


def format_full_report(
    model: ModelSpec,
    server: ServerSpec,
    storage: StorageProfile,
    scenarios: Dict[str, ScenarioResult],
    concurrency: int,
    effective_context: int,
    kv_precision: str,
    warnings: List[str],
    sizing_mode: str = "concurrency_driven"
) -> str:
    """
    Gera relatório completo em texto.
    
    Returns:
        String com relatório formatado
    """
    lines = []
    
    # Cabeçalho
    lines.append("=" * 100)
    lines.append("RELATÓRIO COMPLETO DE SIZING - INFRAESTRUTURA PARA INFERÊNCIA")
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
    
    modo_label = "SLO-Driven (latencia guia dimensionamento)" if sizing_mode == "slo_driven" else "Concorrencia-Driven (SLOs implicitos de parameters.json)"
    lines.append(f"Modo de Sizing: {modo_label}")
    lines.append(f"Concorrencia Alvo: {concurrency:,} sessoes")
    lines.append(f"Contexto Efetivo: {effective_context:,} tokens")
    lines.append(f"Precisao KV: {kv_precision}")
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
    lines.append(f"  • Throughput leitura: {storage.throughput_read_mbps:.1f} MB/s ({storage.throughput_read_mbps/1024:.2f} GB/s)")
    lines.append(f"  • Throughput escrita: {storage.throughput_write_mbps:.1f} MB/s ({storage.throughput_write_mbps/1024:.2f} GB/s)")
    lines.append(f"  • Latência leitura (p50/p99): {storage.latency_read_ms_p50:.2f} / {storage.latency_read_ms_p99:.2f} ms")
    lines.append(f"  • Latência escrita (p50/p99): {storage.latency_write_ms_p50:.2f} / {storage.latency_write_ms_p99:.2f} ms")
    if storage.rack_units_u > 0 or storage.power_kw > 0:
        lines.append(f"  • Consumo físico: {storage.rack_units_u}U rack, {storage.power_kw:.1f} kW")
    lines.append("")
    
    # Seção 2.6: Política de Margem de Capacidade
    lines.append("┌" + "─" * 98 + "┐")
    lines.append("│" + " SEÇÃO 2.6: POLÍTICA DE MARGEM DE CAPACIDADE".ljust(98) + "│")
    lines.append("└" + "─" * 98 + "┘")
    lines.append("")
    
    # Obter informações da política do primeiro cenário
    capacity_policy_info = rec.storage.rationale.get("capacity_policy", {})
    margin_pct = capacity_policy_info.get("margin_percent", 0) * 100
    margin_source = capacity_policy_info.get("source", "parameters.json")
    margin_notes = capacity_policy_info.get("notes", "")
    target_load_time = capacity_policy_info.get("target_load_time_sec", 60.0)
    
    lines.append(f"Política Ativa: {margin_source}")
    lines.append(f"Margem Aplicada: {margin_pct:.0f}%")
    lines.append(f"Tempo Alvo de Carga: {target_load_time:.0f} segundos")
    lines.append(f"Justificativa da Margem: {margin_notes}")
    lines.append("")
    lines.append(f"Observação: O tempo alvo de {target_load_time:.0f}s define o tempo máximo aceitável para carregar")
    lines.append(f"os pesos do modelo durante restart/scale-out, impactando o cálculo de throughput pico de storage.")
    lines.append("")
    
    lines.append("TABELA DE APLICAÇÃO DE MARGEM (Cenário RECOMENDADO):")
    lines.append("")
    lines.append(f"{'Métrica':<30} {'Valor Base (TB)':<18} {'Margem (%)':<15} {'Valor Recomendado (TB)':<25}")
    lines.append("-" * 88)
    lines.append(f"{'Storage (modelo)':<30} {rec.storage.storage_model_base_tb:<18.2f} {margin_pct:<15.0f} {rec.storage.storage_model_recommended_tb:<25.2f}")
    lines.append(f"{'Storage (cache)':<30} {rec.storage.storage_cache_base_tb:<18.2f} {margin_pct:<15.0f} {rec.storage.storage_cache_recommended_tb:<25.2f}")
    lines.append(f"{'Storage (logs)':<30} {rec.storage.storage_logs_base_tb:<18.2f} {margin_pct:<15.0f} {rec.storage.storage_logs_recommended_tb:<25.2f}")
    lines.append(f"{'Storage (operacional)':<30} {rec.storage.storage_operational_base_tb:<18.2f} {margin_pct:<15.0f} {rec.storage.storage_operational_recommended_tb:<25.2f}")
    lines.append("-" * 88)
    lines.append(f"{'TOTAL':<30} {rec.storage.storage_total_base_tb:<18.2f} {margin_pct:<15.0f} {rec.storage.storage_total_recommended_tb:<25.2f}")
    lines.append("")
    
    lines.append("RACIONAL OPERACIONAL:")
    lines.append(f"  • Fórmula: Valor Recomendado = Valor Base × (1 + {margin_pct/100:.2f})")
    lines.append(f"  • Origem: {margin_source}")
    lines.append("  • Justificativa Operacional:")
    lines.append("    - Crescimento orgânico da plataforma sem reengenharia")
    lines.append("    - Retenção adicional de logs para auditoria e análise post-mortem")
    lines.append("    - Expansão futura de capacidade sem pressão operacional")
    lines.append("    - Redução de risco de subdimensionamento e indisponibilidade")
    lines.append("    - Margem de segurança para eventos não previstos (cascading failures, warmup concorrente)")
    lines.append("")
    lines.append("NOTA EXECUTIVA:")
    lines.append(f"  O valor BASE ({rec.storage.storage_total_base_tb:.2f} TB) representa o dimensionamento técnico mínimo.")
    lines.append(f"  O valor RECOMENDADO ({rec.storage.storage_total_recommended_tb:.2f} TB) incorpora margem estratégica de {margin_pct:.0f}% para resiliência operacional.")
    lines.append("  Esta margem é governada por política organizacional e pode ser ajustada via --capacity-margin CLI ou parameters.json.")
    lines.append("")
    
    # Seção 2.7: Volume da Plataforma por Servidor
    lines.append("┌" + "─" * 98 + "┐")
    lines.append("│" + " SEÇÃO 2.7: VOLUME DA PLATAFORMA POR SERVIDOR".ljust(98) + "│")
    lines.append("└" + "─" * 98 + "┘")
    lines.append("")
    
    # Obter breakdown da plataforma do primeiro cenário
    platform_rationale = rec.storage.rationale.get("platform_storage", {})
    platform_inputs = platform_rationale.get("inputs", {})
    
    lines.append(f"Volume Estrutural: {rec.storage.platform_per_server_gb:.0f} GB/servidor ({rec.storage.platform_per_server_tb:.2f} TB/servidor)")
    lines.append(f"Fonte: {platform_rationale.get('inputs', {}).get('num_nodes', 'N/A')} nós × {rec.storage.platform_per_server_tb:.2f} TB = {rec.storage.platform_volume_total_tb:.2f} TB total")
    lines.append("")
    
    lines.append("TABELA DE BREAKDOWN POR COMPONENTE:")
    lines.append("")
    lines.append(f"{'Componente':<40} {'Volume/Servidor (GB)':<22} {'Volume Total (TB)':<20} {'Observação':<20}")
    lines.append("-" * 102)
    
    os_gb = platform_inputs.get("os_installation_gb", 0)
    ai_enterprise_gb = platform_inputs.get("nvidia_ai_enterprise_gb", 0)
    container_runtime_gb = platform_inputs.get("container_runtime_gb", 0)
    engines_gb = platform_inputs.get("model_runtime_engines_gb", 0)
    deps_gb = platform_inputs.get("platform_dependencies_gb", 0)
    config_gb = platform_inputs.get("config_and_metadata_gb", 0)
    num_nodes = rec.nodes_final
    
    lines.append(f"{'Sistema Operacional':<40} {os_gb:<22.0f} {os_gb*num_nodes/1024:<20.2f} {'Ubuntu/RHEL+drivers':<20}")
    lines.append(f"{'NVIDIA AI Enterprise':<40} {ai_enterprise_gb:<22.0f} {ai_enterprise_gb*num_nodes/1024:<20.2f} {'CUDA,cuDNN,TensorRT':<20}")
    lines.append(f"{'Runtime de Containers':<40} {container_runtime_gb:<22.0f} {container_runtime_gb*num_nodes/1024:<20.2f} {'Docker,K8s,imagens':<20}")
    lines.append(f"{'Engines de Inferência':<40} {engines_gb:<22.0f} {engines_gb*num_nodes/1024:<20.2f} {'TensorRT-LLM,vLLM,NIM':<20}")
    lines.append(f"{'Dependências da Plataforma':<40} {deps_gb:<22.0f} {deps_gb*num_nodes/1024:<20.2f} {'Python,NCCL,ML libs':<20}")
    lines.append(f"{'Configuração e Metadados':<40} {config_gb:<22.0f} {config_gb*num_nodes/1024:<20.2f} {'Helm,certs,secrets':<20}")
    lines.append("-" * 102)
    lines.append(f"{'TOTAL por servidor':<40} {rec.storage.platform_per_server_gb:<22.0f} {rec.storage.platform_volume_total_tb:<20.2f} {'':<20}")
    lines.append("")
    
    lines.append(f"TOTAL PLATAFORMA (todos os {num_nodes} nós): {rec.storage.platform_volume_total_tb:.2f} TB")
    lines.append("")
    
    lines.append("NOTA OPERACIONAL:")
    lines.append(platform_rationale.get("operational_meaning", "Volume estrutural fixo da plataforma de IA."))
    lines.append("")
    
    # Seção 2.8: Análise de Latência (TTFT/TPOT) - apenas se SLOs foram definidos
    any_latency = any(scenarios[k].latency is not None for k in scenarios)
    if any_latency:
        lines.append("┌" + "─" * 98 + "┐")
        lines.append("│" + " SEÇÃO 2.8: ANÁLISE DE LATÊNCIA DE INFERÊNCIA (TTFT E TPOT)".ljust(98) + "│")
        lines.append("└" + "─" * 98 + "┘")
        lines.append("")

        # Cabeçalho SLO (uma vez só)
        first_la = next(scenarios[k].latency for k in ["minimum", "recommended", "ideal"] if scenarios[k].latency)
        from .calc_response_time import load_latency_benchmarks
        benchmarks = load_latency_benchmarks()

        lines.append("SLO Definido:")
        if first_la.target_ttft_p50_ms:
            lines.append(f"  • TTFT P50: {first_la.target_ttft_p50_ms} ms ({first_la.target_ttft_p50_ms/1000:.1f}s)")
        if first_la.target_ttft_p99_ms:
            lines.append(f"  • TTFT P99: {first_la.target_ttft_p99_ms} ms ({first_la.target_ttft_p99_ms/1000:.1f}s)")
        if first_la.target_tpot_tokens_per_sec:
            lines.append(f"  • TPOT mínimo: {first_la.target_tpot_tokens_per_sec:.1f} tokens/segundo")
        lines.append("")

        lines.append("Premissas:")
        lines.append(f"  • Tokens de entrada (média): {first_la.avg_input_tokens:,}")
        lines.append(f"  • Tokens de saída (média): {first_la.avg_output_tokens}")
        lines.append(f"  • Network Latency P50: {first_la.network_latency_p50_ms:.0f} ms")
        lines.append(f"  • Network Latency P99: {first_la.network_latency_p99_ms:.0f} ms")
        lines.append(f"  • Fonte prefill: {first_la.source_prefill}")
        lines.append(f"  • Fonte decode: {first_la.source_decode}")
        lines.append("")

        ttft_exc = benchmarks.get('ttft_excellent_ms', 500)
        ttft_acc = benchmarks.get('ttft_acceptable_ms', 2000)
        tpot_exc = benchmarks.get('tpot_excellent_tokens_per_sec', 10)
        tpot_acc = benchmarks.get('tpot_acceptable_tokens_per_sec', 6)
        lines.append("Benchmarks da Indústria:")
        lines.append(f"  • TTFT: Excelente < {ttft_exc}ms | Aceitável: {ttft_exc}-{ttft_acc}ms | Lento > {ttft_acc}ms")
        lines.append(f"  • TPOT: Excelente > {tpot_exc} tok/s | Aceitável: {tpot_acc}-{tpot_exc} tok/s | Lento < {tpot_acc} tok/s")
        lines.append("")

        # Por cenário
        scenario_label_map = {
            "minimum": "MÍNIMO", "recommended": "RECOMENDADO", "ideal": "IDEAL"
        }
        status_icon = {'OK': '[OK]', 'SLO_MARGINAL': '[MARGINAL]', 'SLO_VIOLATION': '[VIOLADO]', 'NO_SLO': '[SEM SLO]'}
        qual_label = {
            'excellent': 'EXCELENTE', 'good': 'BOM', 'acceptable': 'ACEITAVEL', 'slow': 'LENTO'
        }
        util_label = lambda u: (
            'CRÍTICO (risco de saturação)' if u >= 0.90 else
            'ALTO' if u >= 0.80 else
            'ACEITÁVEL' if u >= 0.60 else
            'IDEAL'
        )

        for key in ["minimum", "recommended", "ideal"]:
            la = scenarios[key].latency
            if la is None:
                continue
            lines.append("─" * 84)
            lines.append(f"CENÁRIO: {scenario_label_map[key]}")
            lines.append("─" * 84)
            lines.append("")

            # TTFT
            lines.append("TTFT (Time to First Token):")
            lines.append(f"  - Network Latency:      {la.network_latency_p50_ms:>8.0f} ms")
            if la.queuing_delay_p50_ms >= 99000:
                lines.append(f"  - Queuing Delay P50:    {'inf (saturado)':>14}")
            else:
                lines.append(f"  - Queuing Delay P50:    {la.queuing_delay_p50_ms:>8.0f} ms")
            lines.append(f"  - Prefill Time:         {la.prefill_time_ms:>8.0f} ms")
            lines.append(f"  - {'─'*29}")
            if la.target_ttft_p50_ms:
                slo_tag = '[OK]' if la.ttft_p50_ok else '[VIOLADO]'
                margin_txt = f"+{abs(la.ttft_p50_margin_percent):.1f}% margem" if la.ttft_p50_ok else f"+{abs(la.ttft_p50_margin_percent):.1f}% acima do SLO"
                lines.append(f"  - TTFT P50:             {la.ttft_p50_ms:>8.0f} ms  {slo_tag} {margin_txt}")
            else:
                lines.append(f"  - TTFT P50:             {la.ttft_p50_ms:>8.0f} ms  (sem SLO definido)")
            if la.target_ttft_p99_ms:
                slo_tag = '[OK]' if la.ttft_p99_ok else '[VIOLADO]'
                lines.append(f"  - TTFT P99:             {la.ttft_p99_ms:>8.0f} ms  {slo_tag} (SLO: {la.target_ttft_p99_ms}ms)")
            lines.append("")
            lines.append(f"Status TTFT: {'[OK] SLO ATENDIDO' if la.ttft_p50_ok else '[VIOLADO] SLO NAO ATENDIDO'}")
            lines.append(f"Classificacao TTFT: {qual_label.get(la.ttft_quality, la.ttft_quality.upper())} -- {_ttft_qual_desc(la.ttft_quality, benchmarks)}")
            lines.append("")

            # TPOT
            lines.append("TPOT (Time Per Output Token):")
            lines.append(f"  - Throughput decode (no):  {la.decode_throughput:>8.0f} tokens/s")
            lines.append(f"  - Sessoes ativas por no:   {scenarios[key].sessions_per_node_effective:>8} (efetivas)")
            lines.append(f"  - {'─'*35}")
            if la.target_tpot_tokens_per_sec:
                slo_tag = '[OK]' if la.tpot_ok else '[VIOLADO]'
                margin_txt = f"+{abs(la.tpot_margin_percent):.1f}% acima do minimo" if la.tpot_ok else f"{abs(la.tpot_margin_percent):.1f}% abaixo do SLO"
                lines.append(f"  - TPOT por sessao:         {la.tpot_tokens_per_sec:>8.2f} tok/s  {slo_tag} {margin_txt}")
            else:
                lines.append(f"  - TPOT por sessao:         {la.tpot_tokens_per_sec:>8.2f} tok/s  (sem SLO definido)")
            lines.append(f"  - ITL (ms/token):          {la.itl_ms_per_token:>8.0f} ms/token")
            lines.append("")
            lines.append(f"Status TPOT: {'[OK] SLO ATENDIDO' if la.tpot_ok else '[VIOLADO] SLO NAO ATENDIDO'}")
            lines.append(f"Classificacao TPOT: {qual_label.get(la.tpot_quality, la.tpot_quality.upper())} -- {_tpot_qual_desc(la.tpot_quality, benchmarks)}")
            lines.append("")

            lines.append(f"Utilização: {la.utilization*100:.1f}% ({util_label(la.utilization)})")
            lines.append(f"Gargalo Principal: {la.bottleneck}")
            lines.append("")
            lines.append(f"Recomendação:")
            for rec_line in la.recommendation.split('\n'):
                lines.append(f"  {rec_line}")
            lines.append("")

        # Racional de cálculo TTFT/TPOT
        lines.append("═" * 84)
        lines.append("RACIONAL DE CÁLCULO: TTFT E TPOT")
        lines.append("═" * 84)
        lines.append("")
        lines.append(f"{'Componente':<30} {'Fórmula':<35} {'Fonte':<35}")
        lines.append("-" * 100)
        lines.append(f"{'Network Latency':<30} {'network_latency_p50_ms':<35} {'parameters.json':<35}")
        lines.append(f"{'avg_output_tokens':<30} {'avg_output_tokens':<35} {'parameters.json':<35}")
        lines.append(f"{'Prefill Time':<30} {'(input_tokens/prefill_thr)*1000':<35} {'models.json → performance':<35}")
        lines.append(f"{'num_input_tokens':<30} {'effective_context / 2':<35} {'CLI --effective-context':<35}")
        lines.append(f"{'Queuing Delay':<30} {'(ρ/(1-ρ)) × SvcTime × factor':<35} {'parameters.json (queuing_factor_*)':<35}")
        lines.append(f"{'max_utilization':<30} {'threshold de saturação':<35} {'parameters.json':<35}")
        lines.append(f"{'TTFT':<30} {'network + queuing + prefill':<35} {'(derivado)':<35}")
        lines.append(f"{'Decode Throughput':<30} {'decode_tokens_per_sec_<gpu>':<35} {'models.json → performance':<35}")
        lines.append(f"{'TPOT':<30} {'decode_thr / sessions_per_node':<35} {'models.json + sizing':<35}")
        lines.append(f"{'ITL':<30} {'1000 / TPOT':<35} {'(derivado)':<35}")
        lines.append(f"{'Benchmarks':<30} {'latency_benchmarks.*':<35} {'parameters.json':<35}")
        lines.append("")

    # Seção 2.9: Capacidade Máxima por SLO / Calibração
    any_slo_capacity = any(scenarios[k].slo_capacity is not None for k in scenarios)
    any_calibration = any(scenarios[k].calibration is not None for k in scenarios)

    if any_slo_capacity:
        lines.append("┌" + "─" * 98 + "┐")
        if sizing_mode == "slo_driven":
            lines.append("│" + " SECAO 2.9: CAPACIDADE MAXIMA POR SLO DE LATENCIA (MODO SLO-DRIVEN)".ljust(98) + "│")
        else:
            lines.append("│" + " SECAO 2.9: CALIBRACAO RECOMENDADA (SLOs IMPLICITOS)".ljust(98) + "│")
        lines.append("└" + "─" * 98 + "┘")
        lines.append("")

        first_slo = next(scenarios[k].slo_capacity for k in scenarios if scenarios[k].slo_capacity)
        lines.append("FORMULA (SIZING REVERSO):")
        lines.append(f"  queuing_budget = TTFT_SLO - rede_p50 - prefill_time")
        lines.append(f"  util_max       = queuing_budget / (service_time x qf_p50 + queuing_budget)")
        lines.append(f"  max_conc_TTFT  = floor(util_max x num_nos x sessoes/no)")
        lines.append(f"  sess_max/no    = floor(decode_thr / TPOT_min)")
        lines.append(f"  max_conc_TPOT  = sess_max/no x num_nos")
        lines.append(f"  max_conc       = min(max_conc_TTFT, max_conc_TPOT)")
        lines.append("")
        lines.append(f"  Prefill time calculado: {first_slo.prefill_time_ms:.0f} ms")
        lines.append("")

        lines.append(f"{'Cenario':<18} {'Max TTFT (sess)':<18} {'Max TPOT (sess)':<18} {'Max Final (sess)':<18} {'Gargalo':<12} {'Viavel?':<10}")
        lines.append("-" * 94)
        for k in ["minimum", "recommended", "ideal"]:
            sc = scenarios[k].slo_capacity
            if sc:
                viavel = "Sim" if sc.is_feasible else "Nao"
                lines.append(f"{scenarios[k].config.name:<18} {sc.max_concurrency_from_ttft:<18} {sc.max_concurrency_from_tpot:<18} {sc.max_concurrency_combined:<18} {sc.limiting_factor:<12} {viavel:<10}")
        lines.append("")

        if any_calibration or sizing_mode == "concurrency_driven":
            lines.append("CALIBRACAO RECOMENDADA:")
            lines.append(f"{'Cenario':<18} {'Solicitado':<14} {'Max c/SLOs':<14} {'Nos Atuais':<12} {'Nos Recom.':<12} {'Nos Extras':<12}")
            lines.append("-" * 82)
            for k in ["minimum", "recommended", "ideal"]:
                cal = scenarios[k].calibration
                sc = scenarios[k].slo_capacity
                if sc:
                    nodes_rec = cal.nodes_recommended if cal else scenarios[k].nodes_final
                    extra = cal.extra_nodes_needed if cal else 0
                    lines.append(f"{scenarios[k].config.name:<18} {scenarios[k].calibration.concurrency_requested if cal else 'N/A':<14} {sc.max_concurrency_combined:<14} {scenarios[k].nodes_final:<12} {nodes_rec or 'N/A':<12} {extra:<12}")
            lines.append("")

    # Seção 3: Resultados por Cenário
    lines.append("┌" + "─" * 98 + "┐")
    lines.append("│" + " SECAO 3: RESULTADOS POR CENARIO".ljust(98) + "│")
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
            margin_pct = s.storage.margin_percent * 100
            lines.append(f"  • Volumetria total (BASE): {s.storage.storage_total_base_tb:.2f} TB")
            lines.append(f"  • Volumetria total (RECOMENDADA): {s.storage.storage_total_recommended_tb:.2f} TB (base + {margin_pct:.0f}%)")
            lines.append(f"    - Modelo (base/recomendado): {s.storage.storage_model_base_tb:.2f} / {s.storage.storage_model_recommended_tb:.2f} TB")
            lines.append(f"    - Cache (base/recomendado): {s.storage.storage_cache_base_tb:.2f} / {s.storage.storage_cache_recommended_tb:.2f} TB")
            lines.append(f"    - Logs (base/recomendado): {s.storage.storage_logs_base_tb:.2f} / {s.storage.storage_logs_recommended_tb:.2f} TB")
            lines.append(f"    - Operacional (base/recomendado): {s.storage.storage_operational_base_tb:.2f} / {s.storage.storage_operational_recommended_tb:.2f} TB")
            lines.append(f"    - Plataforma (base/recomendado): {s.storage.platform_volume_total_tb:.2f} / {s.storage.platform_volume_recommended_tb:.2f} TB ({s.storage.platform_per_server_tb:.2f} TB/servidor × {s.nodes_final} nós)")
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


def _ttft_qual_desc(quality: str, benchmarks: dict) -> str:
    """Retorna descrição textual da qualidade TTFT."""
    exc = benchmarks.get('ttft_excellent_ms', 500)
    acc = benchmarks.get('ttft_acceptable_ms', 2000)
    if quality == 'excellent':
        return f"< {exc}ms — experiência interativa, sem percepção de delay"
    elif quality == 'good':
        return f"< {benchmarks.get('ttft_good_ms', 1000)}ms — responsivo, delay mínimo"
    elif quality == 'acceptable':
        return f"até {acc}ms — padrão da indústria"
    else:
        return f"> {acc}ms — usuário percebe demora significativa"


def _tpot_qual_desc(quality: str, benchmarks: dict) -> str:
    """Retorna descrição textual da qualidade TPOT."""
    exc = benchmarks.get('tpot_excellent_tokens_per_sec', 10)
    acc = benchmarks.get('tpot_acceptable_tokens_per_sec', 6)
    if quality == 'excellent':
        return f"> {exc} tok/s — streaming fluido (~{int(exc*0.6)} palavras/min)"
    elif quality == 'good':
        return f"{benchmarks.get('tpot_good_tokens_per_sec', 8)}-{exc} tok/s — streaming adequado"
    elif quality == 'acceptable':
        return f"{acc}-{benchmarks.get('tpot_good_tokens_per_sec', 8)} tok/s — mínimo para produção"
    else:
        return f"< {acc} tok/s — streaming lento, prejudica UX"


def format_json_report(
    model: ModelSpec,
    server: ServerSpec,
    storage: StorageProfile,
    scenarios: Dict[str, ScenarioResult],
    concurrency: int,
    effective_context: int,
    kv_precision: str,
    warnings: List[str],
    sizing_mode: str = "concurrency_driven"
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
                # Valores BASE (técnicos)
                "storage_model_base_tb": round(s.storage.storage_model_base_tb, 3),
                "storage_cache_base_tb": round(s.storage.storage_cache_base_tb, 3),
                "storage_logs_base_tb": round(s.storage.storage_logs_base_tb, 3),
                "storage_operational_base_tb": round(s.storage.storage_operational_base_tb, 3),
                "platform_volume_total_tb": round(s.storage.platform_volume_total_tb, 3),
                "storage_total_base_tb": round(s.storage.storage_total_base_tb, 3),
                # Valores RECOMENDADOS (estratégicos com margem)
                "storage_model_recommended_tb": round(s.storage.storage_model_recommended_tb, 3),
                "storage_cache_recommended_tb": round(s.storage.storage_cache_recommended_tb, 3),
                "storage_logs_recommended_tb": round(s.storage.storage_logs_recommended_tb, 3),
                "storage_operational_recommended_tb": round(s.storage.storage_operational_recommended_tb, 3),
                "platform_volume_recommended_tb": round(s.storage.platform_volume_recommended_tb, 3),
                "storage_total_recommended_tb": round(s.storage.storage_total_recommended_tb, 3),
                # Detalhamento da plataforma
                "platform_per_server_gb": round(s.storage.platform_per_server_gb, 2),
                "platform_per_server_tb": round(s.storage.platform_per_server_tb, 3),
                # Margem aplicada
                "margin_applied": s.storage.margin_applied,
                "margin_percent": round(s.storage.margin_percent, 3),
                # IOPS e Throughput
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
        
        # Adicionar análise de latência se disponível
        if s.latency is not None:
            result["latency_analysis"] = latency_analysis_to_dict(s.latency)

        # Adicionar capacidade máxima por SLO se disponível
        if s.slo_capacity is not None:
            sc = s.slo_capacity
            result["slo_capacity"] = {
                "max_concurrency_from_ttft": sc.max_concurrency_from_ttft,
                "max_concurrency_from_tpot": sc.max_concurrency_from_tpot,
                "max_concurrency_combined": sc.max_concurrency_combined,
                "limiting_factor": sc.limiting_factor,
                "util_max_from_ttft": round(sc.util_max_from_ttft, 4),
                "sessions_per_node_max_from_tpot": sc.sessions_per_node_max_from_tpot,
                "prefill_time_ms": round(sc.prefill_time_ms, 1),
                "queuing_budget_ms": round(sc.queuing_budget_ms, 1),
                "is_feasible": sc.is_feasible,
                "infeasibility_reason": sc.infeasibility_reason
            }

        # Adicionar calibração se disponível
        if s.calibration is not None:
            cal = s.calibration
            result["calibration"] = {
                "nodes_current": cal.nodes_current,
                "nodes_recommended": cal.nodes_recommended,
                "max_concurrency_current_nodes": cal.max_concurrency_current_nodes,
                "concurrency_requested": cal.concurrency_requested,
                "limiting_factor": cal.limiting_factor,
                "extra_nodes_needed": cal.extra_nodes_needed
            }

        return result
    
    return {
        "inputs": {
            "model": model.name,
            "server": server.name,
            "storage": storage.name,
            "concurrency": concurrency,
            "effective_context": effective_context,
            "kv_precision": kv_precision,
            "sizing_mode": sizing_mode
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
        "capacity_policy": {
            "margin_percent": scenarios["recommended"].storage.margin_percent if scenarios["recommended"].storage else 0.0,
            "target_load_time_sec": scenarios["recommended"].storage.rationale.get("capacity_policy", {}).get("target_load_time_sec", 60.0) if scenarios["recommended"].storage else 60.0,
            "applied_to": [
                "storage_total",
                "storage_model",
                "storage_cache",
                "storage_logs",
                "storage_operational"
            ],
            "source": scenarios["recommended"].storage.rationale.get("capacity_policy", {}).get("source", "parameters.json") if scenarios["recommended"].storage else "N/A",
            "notes": scenarios["recommended"].storage.rationale.get("capacity_policy", {}).get("notes", "") if scenarios["recommended"].storage else ""
        },
        "platform_storage": {
            "per_server_gb": round(scenarios["recommended"].storage.platform_per_server_gb, 2) if scenarios["recommended"].storage else 0.0,
            "per_server_tb": round(scenarios["recommended"].storage.platform_per_server_tb, 3) if scenarios["recommended"].storage else 0.0,
            "total_tb_recommended": round(scenarios["recommended"].storage.platform_volume_total_tb, 3) if scenarios["recommended"].storage else 0.0,
            "source": "platform_storage_profile.json",
            "breakdown": scenarios["recommended"].storage.rationale.get("platform_storage", {}).get("inputs", {}) if scenarios["recommended"].storage else {}
        },
        "scenarios": {
            "minimum": scenario_to_dict(scenarios["minimum"]),
            "recommended": scenario_to_dict(scenarios["recommended"]),
            "ideal": scenario_to_dict(scenarios["ideal"])
        },
        "warnings": warnings
    }
